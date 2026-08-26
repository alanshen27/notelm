#!/usr/bin/env python3
"""RunPod training lifecycle for notelm (stdlib only).

Provision a single RTX A5000 secure-cloud pod, sync the repo, train the
transformer on POP909, pull checkpoints back, and terminate. Costs are
guarded by --budget and --max-hours; the pod is terminated on breach.

Read-only commands (check/status/logs) never spend money.

Usage:
  python3 scripts/runpod_train.py check
  python3 scripts/runpod_train.py full [--epochs 60] [--budget 25] [--max-hours 12]
  python3 scripts/runpod_train.py create | sync | setup | train | monitor |
                                  download | terminate | ssh-cmd

API key: RUNPOD_API_KEY env var or .env in the repo root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
STATE_FILE = ROOT / ".runpod_pod.json"
SSH_KEY = Path.home() / ".ssh" / "notelm_runpod"

GPU_TYPE = "NVIDIA RTX A5000"  # 24 GB, ~$0.27/hr secure cloud (closest to A10G)
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME = "notelm-train"
REMOTE_DIR = "/workspace/notelm"

SYNC_EXCLUDES = [
    ".git", ".venv", "data", "outputs", "logs", "checkpoints",
    "src/checkpoints", "checkpoints_deprecated", "ui/node_modules",
    "ui/dist", "__pycache__", ".env", "*.zip", ".runpod_pod.json",
]

DEFAULT_TRAIN_ARGS = "--model transformer --dataset pop909 --tokenizer event"

# Named training recipes (run as one chained command in the remote tmux session).
RECIPES = {
    # Straight POP909 training, event tokenizer.
    "pop": [
        "python -u train.py --model transformer --dataset pop909 "
        "--tokenizer event --epochs {epochs}",
    ],
    # Pretrain on full MAESTRO, fine-tune on POP909 (event), then POP909 REMI.
    "pretrain-finetune": [
        "python -u train.py --model transformer --dataset maestro_full "
        "--tokenizer event --epochs 20",
        "cp checkpoints/transformer/event/weights.pt "
        "checkpoints/transformer/event/maestro-pretrain.pt",
        "python -u train.py --model transformer --dataset pop909 "
        "--tokenizer event --epochs {epochs} --lr 1e-4 "
        "--weights checkpoints/transformer/event/maestro-pretrain.pt --start-epoch 0",
        "python -u train.py --model transformer --dataset pop909 "
        "--tokenizer remi --epochs {epochs}",
    ],
}


def _api_key() -> str:
    import os

    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith("RUNPOD_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not key:
        sys.exit("RUNPOD_API_KEY not set (env var or .env in repo root)")
    return key


def _request(method: str, url: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {_api_key()}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "notelm-runpod-train/1.0")
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=60) as resp:
            text = resp.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        sys.exit(f"RunPod API {method} {url} failed ({e.code}): {detail}")


def _graphql(query: str) -> dict:
    return _request("POST", GRAPHQL, {"query": query})


def _state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _pod_id() -> str:
    pod_id = _state().get("pod_id")
    if not pod_id:
        sys.exit("No pod recorded. Run `create` first (state: .runpod_pod.json)")
    return pod_id


def _get_pod(pod_id: str) -> dict:
    return _request("GET", f"{API}/pods/{pod_id}")


def _ensure_ssh_key() -> str:
    if not SSH_KEY.exists():
        print(f"Generating dedicated SSH key at {SSH_KEY}")
        SSH_KEY.parent.mkdir(mode=0o700, exist_ok=True)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "notelm-runpod",
             "-f", str(SSH_KEY)],
            check=True,
        )
    return (SSH_KEY.with_suffix(".pub")).read_text().strip()


def _ssh_endpoint(pod: dict) -> tuple[str, int]:
    ip = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22")
    if not ip or not port:
        raise RuntimeError("SSH endpoint not ready (no public IP / port 22 mapping)")
    return ip, int(port)


def _ssh_base(pod: dict) -> list[str]:
    ip, port = _ssh_endpoint(pod)
    return [
        "ssh", "-i", str(SSH_KEY), "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
        f"root@{ip}",
    ]


def _ssh_run(pod: dict, remote_cmd: str, *, check: bool = True) -> int:
    proc = subprocess.run(_ssh_base(pod) + [remote_cmd])
    if check and proc.returncode != 0:
        sys.exit(f"Remote command failed ({proc.returncode}): {remote_cmd}")
    return proc.returncode


def _cost_so_far(state: dict, pod: dict) -> float:
    created = state.get("created_at", time.time())
    hours = (time.time() - created) / 3600
    return hours * float(pod.get("costPerHr") or 0.0)


# ---------------------------------------------------------------- commands


def cmd_check(_args) -> None:
    """Read-only: verify key, show balance and existing pods."""
    balance = None
    try:
        result = _graphql("query { myself { clientBalance spendLimit } }")
        myself = (result.get("data") or {}).get("myself") or {}
        balance = myself.get("clientBalance")
    except SystemExit as e:
        print(f"(balance lookup unavailable: {e})")

    pods = _request("GET", f"{API}/pods")
    print(f"API key OK. Balance: ${balance if balance is not None else '?'}")
    items = pods if isinstance(pods, list) else pods.get("pods", [])
    if not items:
        print("No pods running (nothing is billing).")
    for p in items:
        print(
            f"  pod {p.get('id')}  {p.get('name')}  "
            f"status={p.get('desiredStatus')}  ${p.get('costPerHr')}/hr"
        )

    a5000_hours = None
    if isinstance(balance, (int, float)) and balance > 0:
        a5000_hours = balance / 0.27
        print(f"Balance buys ~{a5000_hours:.0f} h of RTX A5000 secure cloud.")
    elif balance == 0:
        print("Balance is $0 — fund the account before running `create`/`full`.")


def cmd_create(args) -> None:
    if _state().get("pod_id"):
        sys.exit(f"Pod already recorded in {STATE_FILE}; terminate it first.")
    pubkey = _ensure_ssh_key()
    body = {
        "name": POD_NAME,
        "imageName": args.image,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "containerDiskInGb": 40,
        "volumeInGb": 40,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pubkey},
    }
    print(f"Creating pod: 1x {args.gpu}, secure cloud, image {args.image}")
    pod = _request("POST", f"{API}/pods", body)
    pod_id = pod["id"]
    _save_state({
        "pod_id": pod_id,
        "created_at": time.time(),
        "cost_per_hr": pod.get("costPerHr"),
    })
    print(f"Pod created: {pod_id} (${pod.get('costPerHr')}/hr). Waiting for SSH...")
    cmd_wait(args)


def cmd_wait(_args) -> None:
    pod_id = _pod_id()
    deadline = time.time() + 600
    while time.time() < deadline:
        pod = _get_pod(pod_id)
        try:
            ip, port = _ssh_endpoint(pod)
        except RuntimeError:
            print(f"  status={pod.get('desiredStatus')} — waiting for network...")
            time.sleep(15)
            continue
        rc = subprocess.run(
            _ssh_base(pod) + ["echo ok"], capture_output=True
        ).returncode
        if rc == 0:
            print(f"SSH ready: root@{ip} -p {port}")
            return
        print(f"  ssh not accepting yet (root@{ip}:{port}) — retrying...")
        time.sleep(15)
    sys.exit("Timed out waiting for pod SSH (10 min). Check RunPod console.")


def cmd_status(_args) -> None:
    state = _state()
    pod = _get_pod(_pod_id())
    print(f"pod {pod['id']}  status={pod.get('desiredStatus')}  ${pod.get('costPerHr')}/hr")
    try:
        ip, port = _ssh_endpoint(pod)
        print(f"ssh -i {SSH_KEY} -p {port} root@{ip}")
    except RuntimeError as e:
        print(e)
    print(f"est. spend so far: ${_cost_so_far(state, pod):.2f}")


def cmd_ssh_cmd(_args) -> None:
    pod = _get_pod(_pod_id())
    ip, port = _ssh_endpoint(pod)
    print(f"ssh -i {SSH_KEY} -p {port} -o StrictHostKeyChecking=no root@{ip}")


def cmd_sync(_args) -> None:
    pod = _get_pod(_pod_id())
    excludes = " ".join(f"--exclude='{e}'" for e in SYNC_EXCLUDES)
    tar_cmd = f"tar czf - {excludes} -C '{ROOT}' ."
    remote = f"mkdir -p {REMOTE_DIR} && tar xzf - -C {REMOTE_DIR}"
    print(f"Syncing repo -> {REMOTE_DIR} ...")
    ssh = " ".join(f"'{p}'" if " " in p else p for p in _ssh_base(pod))
    rc = subprocess.run(f"{tar_cmd} | {ssh} \"{remote}\"", shell=True).returncode
    if rc != 0:
        sys.exit(f"Sync failed ({rc})")
    print("Sync done.")


def cmd_setup(_args) -> None:
    pod = _get_pod(_pod_id())
    print("Running remote setup (uv + CUDA torch + POP909)...")
    _ssh_run(
        pod,
        f"cd {REMOTE_DIR} && chmod +x scripts/*.sh && "
        f"./scripts/setup.sh --system --fetch-pop909 --cuda",
    )
    print("Remote setup done.")


def cmd_train(args) -> None:
    pod = _get_pod(_pod_id())
    if args.recipe:
        steps = [s.format(epochs=args.epochs) for s in RECIPES[args.recipe]]
        chain = " && ".join(steps)
        remote = (
            f"cd {REMOTE_DIR} && mkdir -p logs && "
            f"tmux new-session -d -s notelm-train "
            f"\"source .venv/bin/activate && cd src && ({chain}) 2>&1 "
            f"| tee -a ../logs/train-recipe.log\""
        )
        print(f"Launching recipe '{args.recipe}' ({len(steps)} steps) in tmux")
        _ssh_run(pod, remote)
    else:
        train_args = args.train_args or f"{DEFAULT_TRAIN_ARGS} --epochs {args.epochs}"
        print(f"Launching training in tmux: train.py {train_args}")
        _ssh_run(pod, f"cd {REMOTE_DIR} && ./scripts/train_tmux.sh {train_args}")
    print("Training started. Follow with: python3 scripts/runpod_train.py monitor")


def cmd_logs(_args) -> None:
    pod = _get_pod(_pod_id())
    _ssh_run(
        pod,
        f"tail -n 40 {REMOTE_DIR}/logs/train-*.log 2>/dev/null "
        f"|| echo 'no logs yet'",
        check=False,
    )


def _training_alive(pod: dict) -> bool:
    rc = subprocess.run(
        _ssh_base(pod) + ["tmux has-session -t notelm-train 2>/dev/null"],
        capture_output=True,
    ).returncode
    return rc == 0


def cmd_monitor(args) -> None:
    """Poll until training ends; enforce budget/max-hours; then download+terminate."""
    state = _state()
    pod_id = _pod_id()
    print(
        f"Monitoring pod {pod_id} (budget ${args.budget:.2f}, "
        f"max {args.max_hours:.1f} h). Ctrl-C detaches without terminating."
    )
    while True:
        pod = _get_pod(pod_id)
        cost = _cost_so_far(state, pod)
        hours = (time.time() - state.get("created_at", time.time())) / 3600
        alive = _training_alive(pod)
        subprocess.run(
            _ssh_base(pod)
            + [f"grep -h 'val loss' {REMOTE_DIR}/logs/train-*.log 2>/dev/null | tail -3"],
        )
        print(f"  [{hours:.2f} h, est ${cost:.2f}] training {'running' if alive else 'DONE'}")

        if not alive:
            print("Training finished.")
            break
        if cost >= args.budget or hours >= args.max_hours:
            print(f"BUDGET GUARD tripped (${cost:.2f} / {hours:.1f} h) — stopping training.")
            _ssh_run(pod, "tmux kill-session -t notelm-train", check=False)
            break
        time.sleep(args.poll)

    cmd_download(args)
    cmd_terminate(args)


def cmd_download(_args) -> None:
    pod = _get_pod(_pod_id())
    dest = ROOT / "src"
    dest.mkdir(exist_ok=True)
    print(f"Downloading checkpoints -> {dest}/checkpoints ...")
    ssh = " ".join(f"'{p}'" if " " in p else p for p in _ssh_base(pod))
    remote = (
        f"cd {REMOTE_DIR}/src 2>/dev/null || cd {REMOTE_DIR}; "
        f"tar czf - checkpoints logs 2>/dev/null || tar czf - checkpoints"
    )
    rc = subprocess.run(f"{ssh} \"{remote}\" | tar xzf - -C '{dest}'", shell=True).returncode
    if rc != 0:
        sys.exit(f"Download failed ({rc}) — pod NOT terminated, retry `download`.")
    print("Checkpoints downloaded.")


def cmd_terminate(_args) -> None:
    state = _state()
    pod_id = _pod_id()
    pod = _get_pod(pod_id)
    print(f"Terminating pod {pod_id} (est. total ${_cost_so_far(state, pod):.2f})")
    _request("DELETE", f"{API}/pods/{pod_id}")
    STATE_FILE.unlink(missing_ok=True)
    print("Pod deleted — billing stopped.")


def cmd_full(args) -> None:
    cmd_check(args)
    cmd_create(args)
    try:
        cmd_sync(args)
        cmd_setup(args)
        cmd_train(args)
        cmd_monitor(args)  # monitor downloads + terminates on completion
    except (Exception, KeyboardInterrupt):
        print("\nERROR during full run — terminating pod to stop billing.")
        try:
            cmd_terminate(args)
        finally:
            raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=[
        "check", "create", "wait", "status", "ssh-cmd", "sync", "setup",
        "train", "logs", "monitor", "download", "terminate", "full",
    ])
    p.add_argument("--gpu", default=GPU_TYPE, help=f"GPU type id (default: {GPU_TYPE})")
    p.add_argument("--image", default=IMAGE, help="Docker image for the pod")
    p.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    p.add_argument("--train-args", help="Override full train.py argument string")
    p.add_argument("--recipe", choices=sorted(RECIPES),
                   help="Named multi-step training recipe (recommended: pretrain-finetune)")
    p.add_argument("--budget", type=float, default=25.0,
                   help="Hard USD cap for the run (default: 25)")
    p.add_argument("--max-hours", type=float, default=20.0,
                   help="Hard wall-clock cap in hours (default: 20)")
    p.add_argument("--poll", type=int, default=120,
                   help="Monitor poll interval seconds (default: 120)")
    args = p.parse_args()

    handlers = {
        "check": cmd_check, "create": cmd_create, "wait": cmd_wait,
        "status": cmd_status, "ssh-cmd": cmd_ssh_cmd, "sync": cmd_sync,
        "setup": cmd_setup, "train": cmd_train, "logs": cmd_logs,
        "monitor": cmd_monitor, "download": cmd_download,
        "terminate": cmd_terminate, "full": cmd_full,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
