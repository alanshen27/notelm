#!/usr/bin/env python3
"""RunPod training lifecycle for notelm (stdlib only).

Provision a single RTX A5000 secure-cloud pod, sync the repo, train the
transformer on POP909, pull checkpoints back, and terminate. Costs are
guarded by --budget and --max-hours; the pod is terminated on breach.

Read-only commands (check/status/logs) never spend money.

Usage:
  python3 scripts/runpod_train.py check
  python3 scripts/runpod_train.py full [--epochs 60] [--budget 25] [--max-hours 12]
  python3 scripts/runpod_train.py create | sync | setup | fetch-gigamidi |
                                  train | monitor | download | terminate | ssh-cmd

API key: RUNPOD_API_KEY env var or .env in the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
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
POD_NAME = "notelm-train"

CANON_GPU_FALLBACKS = [
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 3090",
]
H100_FALLBACKS = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
]

GPU_TYPE = "NVIDIA RTX A5000"  # 24 GB, ~$0.27/hr secure cloud (closest to A10G)
# Tried in order if the requested type is sold out. Stay in the 16–48 GB band.
GPU_FALLBACKS = [
    "NVIDIA RTX A5000",
    "NVIDIA RTX A4500",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 3090",
]
CLOUD_FALLBACKS = ("SECURE", "COMMUNITY")
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME = "notelm-train"
REMOTE_DIR = "/workspace/notelm"

SYNC_EXCLUDES = [
    ".git", ".venv", "data", "outputs", "logs", "checkpoints",
    "src/checkpoints", "checkpoints_deprecated", "ui/node_modules",
    "ui/dist", "node_modules", "apps/*/node_modules", "apps/*/dist",
    "__pycache__", ".env", "*.zip", ".runpod_pod.json", ".runpod_pod_canon.json",
    ".DS_Store", "._*",
]

DEFAULT_TRAIN_ARGS = "--dataset pop --tokenizer remi"

# Named training recipes (run as one chained command in the remote tmux session).
RECIPES = {
    "pop": [
        "python -u train.py --dataset pop --tokenizer remi --epochs {epochs}",
    ],
    "pop909": [
        "python -u train.py --dataset pop909 --tokenizer remi --epochs {epochs}",
    ],
    "pretrain-finetune": [
        "python -u train.py --dataset pretrain --tokenizer remi --epochs 40",
        "python -u ../scripts/select_best_checkpoint.py "
        "--log ../logs/train-recipe.log "
        "--ckpt-dir checkpoints/transformer/remi "
        "--dest checkpoints/transformer/remi/pretrain.pt "
        "--also checkpoints/transformer/remi/weights.pt",
        "python -u train.py --dataset pop --tokenizer remi --epochs {epochs} "
        "--lr 1e-4 --weights checkpoints/transformer/remi/pretrain.pt --start-epoch 0",
    ],
    "event-pretrain-finetune": [
        "python -u train.py --dataset pretrain --tokenizer event --epochs 20",
        "python -u ../scripts/select_best_checkpoint.py "
        "--log ../logs/train-recipe.log "
        "--ckpt-dir checkpoints/transformer/event "
        "--dest checkpoints/transformer/event/pretrain.pt "
        "--also checkpoints/transformer/event/weights.pt",
        "python -u train.py --dataset pop --tokenizer event --epochs {epochs} "
        "--lr 1e-4 --weights checkpoints/transformer/event/pretrain.pt --start-epoch 0",
    ],
    "canon": [
        "python -u train.py --model canon --tokenizer remi --dataset instruments --epochs 40",
        "python -u ../scripts/select_best_checkpoint.py "
        "--log ../logs/canon-pretrain.log "
        "--ckpt-dir checkpoints/canon/remi "
        "--dest checkpoints/canon/remi/pretrain.pt "
        "--also checkpoints/canon/remi/weights.pt",
        "python -u train.py --model canon --tokenizer remi --dataset piano --epochs {epochs} "
        "--lr 1e-4 --weights checkpoints/canon/remi/pretrain.pt --start-epoch 0",
        "python -u ../scripts/select_best_checkpoint.py "
        "--log ../logs/canon-finetune.log "
        "--ckpt-dir checkpoints/canon/remi "
        "--dest checkpoints/canon/remi/canon.pt "
        "--also checkpoints/canon/remi/weights.pt",
    ],
}


def _dotenv_value(key: str) -> str:
    val = os.environ.get(key, "")
    if val:
        return val.strip().strip("'\"")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            name, _, rest = raw.partition("=")
            if name.strip() == key:
                return rest.strip().strip("'\"")
    return ""


def _api_key() -> str:
    key = _dotenv_value("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY not set (env var or .env in repo root)")
    return key


class ApiError(RuntimeError):
    def __init__(self, code: int, detail: str):
        super().__init__(f"RunPod API failed ({code}): {detail}")
        self.code = code
        self.detail = detail


def _unavailable(err: BaseException) -> bool:
    text = str(err).lower()
    return "no instances currently available" in text or "not available" in text


def _request(
    method: str, url: str, body: dict | None = None, *, fatal: bool = True
) -> dict | list:
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
        err = ApiError(e.code, detail)
        if fatal:
            sys.exit(f"RunPod API {method} {url} failed ({e.code}): {detail}")
        raise err from e


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
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=12",
        "-o", "TCPKeepAlive=yes",
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


def apply_slot(slot: str) -> None:
    """Keep the etude 4090 in .runpod_pod.json; canon uses .runpod_pod_canon.json."""
    global STATE_FILE, POD_NAME
    name = (slot or "default").strip() or "default"
    if name in ("default", "train", "etude"):
        STATE_FILE = ROOT / ".runpod_pod.json"
        POD_NAME = "notelm-train"
    else:
        STATE_FILE = ROOT / f".runpod_pod_{name}.json"
        POD_NAME = f"notelm-{name}"


def _gpu_queue(requested: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    extras = GPU_FALLBACKS
    if "h100" in POD_NAME:
        extras = H100_FALLBACKS
    elif "canon" in POD_NAME:
        extras = CANON_GPU_FALLBACKS
    for name in (requested, *extras):
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def cmd_create(args) -> None:
    if _state().get("pod_id"):
        sys.exit(f"Pod already recorded in {STATE_FILE}; terminate it first.")
    pubkey = _ensure_ssh_key()
    gpus = _gpu_queue(args.gpu)
    clouds = list(CLOUD_FALLBACKS)
    want = max(1, int(getattr(args, "gpu_count", 1) or 1))
    counts = [want] if want == 1 else [want, 1]
    print(f"Creating pod, image {args.image}")
    print(f"  GPU order: {', '.join(gpus)}")
    print(f"  clouds: {', '.join(clouds)}")
    print(f"  gpuCount try: {counts}")

    last_err: BaseException | None = None
    pod: dict | None = None
    attempts: list[tuple[str, list[str], str]] = []
    for cloud in clouds:
        attempts.append((cloud, gpus, "availability"))
    for cloud in clouds:
        for gpu in gpus:
            attempts.append((cloud, [gpu], "custom"))

    seen_keys: set[tuple[str, tuple[str, ...], str, int]] = set()
    for gpu_count in counts:
        for cloud, ids, priority in attempts:
            key = (cloud, tuple(ids), priority, gpu_count)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            label = ids[0] if len(ids) == 1 else f"{len(ids)} types"
            print(f"  trying {gpu_count}x {label} on {cloud} ({priority})...")
            disk = 80 if ("canon" in POD_NAME or "h100" in POD_NAME) else 40
            vol = 80 if ("canon" in POD_NAME or "h100" in POD_NAME) else 40
            body = {
                "name": POD_NAME,
                "imageName": args.image,
                "cloudType": cloud,
                "computeType": "GPU",
                "gpuTypeIds": ids,
                "gpuTypePriority": priority,
                "gpuCount": gpu_count,
                "containerDiskInGb": disk,
                "volumeInGb": vol,
                "volumeMountPath": "/workspace",
                "ports": ["22/tcp"],
                "env": {"PUBLIC_KEY": pubkey},
            }
            try:
                pod = _request("POST", f"{API}/pods", body, fatal=False)  # type: ignore[assignment]
                break
            except ApiError as e:
                last_err = e
                if _unavailable(e) or (len(ids) > 1 and e.code in (400, 500)):
                    print(f"    skip ({e.code}): {e.detail[:180]}")
                    continue
                sys.exit(f"RunPod API POST {API}/pods failed ({e.code}): {e.detail}")
        if isinstance(pod, dict) and pod.get("id"):
            break

    if not isinstance(pod, dict) or not pod.get("id"):
        sys.exit(
            "No GPU stock on Secure or Community for: "
            + ", ".join(gpus)
            + (f"\nLast error: {last_err}" if last_err else "")
        )

    machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    gpu_name = (
        (machine or {}).get("gpuTypeId")
        or pod.get("gpuTypeId")
        or pod.get("gpu")
        or "?"
    )
    _save_state({
        "pod_id": pod["id"],
        "created_at": time.time(),
        "cost_per_hr": pod.get("costPerHr"),
        "gpu": gpu_name,
        "cloud": pod.get("cloudType"),
    })
    print(
        f"Pod created: {pod['id']}  ${pod.get('costPerHr')}/hr  "
        f"gpu={gpu_name}  cloud={pod.get('cloudType') or '?'}"
    )
    print("Waiting for SSH...")
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
    # COPYFILE_DISABLE: skip macOS AppleDouble (._*) forks. Remote GNU tar
    # cannot chown to the Mac uid inside a RunPod container.
    tar_cmd = (
        f"COPYFILE_DISABLE=1 tar czf - {excludes} -C '{ROOT}' ."
    )
    remote = (
        f"mkdir -p {REMOTE_DIR} && "
        f"tar --no-same-owner --no-same-permissions -xzf - -C {REMOTE_DIR}"
    )
    print(f"Syncing repo -> {REMOTE_DIR} ...")
    ssh = " ".join(f"'{p}'" if " " in p else p for p in _ssh_base(pod))
    rc = subprocess.run(f"{tar_cmd} | {ssh} \"{remote}\"", shell=True).returncode
    if rc != 0:
        sys.exit(f"Sync failed ({rc})")
    print("Sync done.")


def cmd_setup(_args) -> None:
    """Run setup.sh in tmux so a laptop/SSH timeout does not SIGHUP uv."""
    pod = _get_pod(_pod_id())
    print("Installing tmux on the pod (fresh images do not have it)...")
    _ssh_run(
        pod,
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq && apt-get install -y -qq tmux curl unzip ca-certificates git",
    )
    print("Running remote setup in tmux (uv + CUDA torch + POP909 + extra MIDI)...")
    _ssh_run(
        pod,
        f"cd {REMOTE_DIR} && chmod +x scripts/*.sh && mkdir -p logs && "
        f"(tmux kill-session -t notelm-setup 2>/dev/null || true) && "
        f"tmux new-session -d -s notelm-setup "
        f"'cd {REMOTE_DIR} && ./scripts/setup.sh --system --fetch-pop909 --fetch-extra "
        f"--fetch-maestro --cuda "
        f">> logs/setup.log 2>&1; echo $? > logs/setup.exit'",
    )
    print(f"  log: {REMOTE_DIR}/logs/setup.log")
    while True:
        alive = (
            subprocess.run(
                _ssh_base(pod) + ["tmux has-session -t notelm-setup 2>/dev/null"],
                capture_output=True,
            ).returncode
            == 0
        )
        subprocess.run(
            _ssh_base(pod)
            + [f"tail -n 5 {REMOTE_DIR}/logs/setup.log 2>/dev/null || true"],
        )
        if not alive:
            break
        print("  setup still running (this can take 10–20 min; SSH idle is ok)...")
        time.sleep(20)

    probe = subprocess.run(
        _ssh_base(pod) + [f"cat {REMOTE_DIR}/logs/setup.exit"],
        capture_output=True,
        text=True,
    )
    code = (probe.stdout or "").strip()
    if code != "0":
        sys.exit(f"Remote setup failed (exit {code or probe.returncode}). See logs/setup.log")
    print("Remote setup done.")


def cmd_fetch_gigamidi(args) -> None:
    """Stream a piano-ish GigaMIDI subset onto the pod (tmux, CPU/network only)."""
    token = _dotenv_value("HF_TOKEN") or _dotenv_value("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN missing. Accept terms at "
            "https://huggingface.co/datasets/Metacreation/GigaMIDI "
            "then add HF_TOKEN to .env"
        )
    max_files = int(args.max_files)
    pod = _get_pod(_pod_id())
    print("Syncing fetch_datasets.py to the pod...")
    ssh = " ".join(f"'{p}'" if " " in p else p for p in _ssh_base(pod))
    rc = subprocess.run(
        f"COPYFILE_DISABLE=1 tar czf - -C '{ROOT}' scripts/fetch_datasets.py | "
        f"{ssh} \"tar --no-same-owner -xzf - -C {REMOTE_DIR}\"",
        shell=True,
    ).returncode
    if rc != 0:
        sys.exit(f"Sync of fetch script failed ({rc})")
    print(f"Pushing Hugging Face token (not logged) and fetching up to {max_files:,} MIDI files...")
    _ssh_run(pod, f"mkdir -p {REMOTE_DIR}/logs")
    push = subprocess.run(
        _ssh_base(pod)
        + ["bash", "-c", "cat > /workspace/notelm/.hf_token && chmod 600 /workspace/notelm/.hf_token"],
        input=token.encode(),
        capture_output=True,
    )
    if push.returncode != 0:
        sys.exit(
            f"Could not write remote HF token ({push.returncode}): "
            f"{push.stderr.decode(errors='replace')}"
        )
    job = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"cd {REMOTE_DIR}\n"
        "source .venv/bin/activate\n"
        "export PATH=\"/root/.local/bin:$HOME/.local/bin:$PATH\"\n"
        "if command -v uv >/dev/null 2>&1; then uv pip install datasets huggingface_hub; "
        "else python -m pip install datasets huggingface_hub; fi\n"
        f'export HF_TOKEN="$(cat {REMOTE_DIR}/.hf_token)"\n'
        "export HF_ENDPOINT=https://huggingface.co\n"
        f"exec python -u scripts/fetch_datasets.py gigamidi --max-files {max_files}\n"
    )
    put = subprocess.run(
        _ssh_base(pod)
        + ["bash", "-c", f"cat > {REMOTE_DIR}/logs/gigamidi-job.sh && chmod +x {REMOTE_DIR}/logs/gigamidi-job.sh"],
        input=job.encode(),
        capture_output=True,
    )
    if put.returncode != 0:
        sys.exit(f"Could not write fetch job ({put.returncode}): {put.stderr.decode(errors='replace')}")
    log_path = f"{REMOTE_DIR}/logs/gigamidi-fetch.log"
    _ssh_run(pod, "tmux kill-session -t notelm-gigamidi 2>/dev/null || true", check=False)
    _ssh_run(
        pod,
        f"tmux new-session -d -s notelm-gigamidi "
        f"'bash {REMOTE_DIR}/logs/gigamidi-job.sh >> {log_path} 2>&1; "
        f"echo $? > {REMOTE_DIR}/logs/gigamidi-fetch.exit'",
    )
    print("  tmux: notelm-gigamidi")
    print(f"  log:  {log_path}")
    print("Waiting for GigaMIDI fetch to finish (do not start train yet)...")
    while True:
        alive = (
            subprocess.run(
                _ssh_base(pod) + ["tmux has-session -t notelm-gigamidi 2>/dev/null"],
                capture_output=True,
            ).returncode
            == 0
        )
        subprocess.run(
            _ssh_base(pod) + [f"tail -n 8 {log_path} 2>/dev/null || true"],
        )
        n = subprocess.run(
            _ssh_base(pod)
            + [
                f"find {REMOTE_DIR}/data/GigaMIDI -iname '*.mid' -o -iname '*.midi' "
                f"2>/dev/null | wc -l"
            ],
            capture_output=True,
            text=True,
        )
        count = (n.stdout or "").strip()
        print(f"  GigaMIDI files on disk: {count or '?'}")
        if not alive:
            break
        time.sleep(30)
    probe = subprocess.run(
        _ssh_base(pod) + [f"cat {REMOTE_DIR}/logs/gigamidi-fetch.exit"],
        capture_output=True,
        text=True,
    )
    code = (probe.stdout or "").strip()
    if code != "0":
        sys.exit(
            f"GigaMIDI fetch failed (exit {code or probe.returncode}). "
            f"See {log_path}"
        )
    print("GigaMIDI fetch done.")


def cmd_train(args) -> None:
    pod = _get_pod(_pod_id())
    _ssh_run(
        pod,
        f"test -x {REMOTE_DIR}/.venv/bin/python || {{ "
        f"echo 'No venv at {REMOTE_DIR}/.venv — run setup first (and check it succeeded).'; "
        f"exit 1; }}",
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = f"{REMOTE_DIR}/logs/train-{stamp}.log"
    _ssh_run(pod, "tmux kill-session -t notelm-train 2>/dev/null || true", check=False)
    if args.recipe:
        steps = [s.format(epochs=args.epochs) for s in RECIPES[args.recipe]]
        chain = " && ".join(steps)
        print(f"Launching recipe '{args.recipe}' ({len(steps)} steps) in tmux")
        _ssh_run(
            pod,
            f"cd {REMOTE_DIR} && mkdir -p logs && "
            f"tmux new-session -d -s notelm-train "
            f"\"source .venv/bin/activate && cd src && ({chain}) "
            f"2>&1 | tee {log_path} {REMOTE_DIR}/logs/train-recipe.log\"",
        )
    else:
        train_args = args.train_args or f"{DEFAULT_TRAIN_ARGS} --epochs {args.epochs}"
        print(f"Launching training in tmux: train.py {train_args}")
        _ssh_run(pod, f"cd {REMOTE_DIR} && ./scripts/train_tmux.sh {train_args}")
    time.sleep(4)
    if not _training_alive(pod):
        print("tmux died immediately. Last log lines:")
        _ssh_run(
            pod,
            f"tail -n 40 {log_path} {REMOTE_DIR}/logs/train-recipe.log 2>/dev/null || true",
            check=False,
        )
        sys.exit(
            "Training did not stay up. Sync new code first:\n"
            "  python3 scripts/runpod_train.py sync\n"
            "then train again."
        )
    st = _state()
    st["train_started_at"] = time.time()
    _save_state(st)
    print("Training started. Follow with: python3 scripts/runpod_train.py monitor")


def cmd_logs(_args) -> None:
    pod = _get_pod(_pod_id())
    logs = f"{REMOTE_DIR}/logs"
    newest = f"latest=$(ls -t {logs}/train-*.log 2>/dev/null | head -1)"
    _ssh_run(
        pod,
        f"{newest}; "
        f"echo '=== tmux ==='; "
        f"if tmux has-session -t notelm-train 2>/dev/null; then "
        f"echo 'notelm-train: running'; "
        f"else echo 'notelm-train: DEAD (job exited — check log tail)'; fi; "
        f"echo; echo \"=== log: ${{latest:-none}} ===\"; "
        f"if [ -n \"$latest\" ]; then "
        f"grep -hE 'Begin training|Epoch |train loss|Dataset ready|skipped |Traceback|Error' "
        f"\"$latest\" 2>/dev/null | tail -n 25; "
        f"echo; echo '=== log tail ==='; tail -n 12 \"$latest\"; "
        f"fi; "
        f"echo; echo '=== live tmux (last lines) ==='; "
        f"if tmux has-session -t notelm-train 2>/dev/null; then "
        f"tmux capture-pane -t notelm-train -p | tail -n 12; "
        f"else echo '(no session)'; fi",
        check=False,
    )


def _training_alive(pod: dict) -> bool:
    rc = subprocess.run(
        _ssh_base(pod) + ["tmux has-session -t notelm-train 2>/dev/null"],
        capture_output=True,
    ).returncode
    return rc == 0


def cmd_monitor(args) -> None:
    """Poll until training ends; enforce budget/max-hours. Never deletes the pod."""
    state = _state()
    pod_id = _pod_id()
    print(
        f"Monitoring pod {pod_id} (budget ${args.budget:.2f}, "
        f"max {args.max_hours:.1f} h). Ctrl-C detaches without terminating."
    )
    started = time.time()
    saw_running = False
    while True:
        pod = _get_pod(pod_id)
        cost = _cost_so_far(state, pod)
        train_t0 = state.get("train_started_at") or started
        hours = (time.time() - train_t0) / 3600
        billed_h = (time.time() - state.get("created_at", time.time())) / 3600
        alive = _training_alive(pod)
        subprocess.run(
            _ssh_base(pod)
            + [f"grep -h 'val loss' {REMOTE_DIR}/logs/train-*.log 2>/dev/null | tail -3"],
        )
        print(
            f"  [train {hours:.2f} h, pod up {billed_h:.2f} h, est ${cost:.2f}] "
            f"training {'running' if alive else 'DONE'}"
        )

        if alive:
            saw_running = True
        if not alive:
            if not saw_running and (time.time() - started) < 90:
                print(
                    "Training session died immediately — setup/train likely failed.\n"
                    "Pod is still running (billing). Inspect with:\n"
                    "  python3 scripts/runpod_train.py logs\n"
                    "  python3 scripts/runpod_train.py status\n"
                    "Stop billing:  python3 scripts/runpod_train.py terminate"
                )
                sys.exit(1)
            print("Training finished.")
            break
        if cost >= args.budget or hours >= args.max_hours:
            print(f"BUDGET GUARD tripped (${cost:.2f} / {hours:.1f} h) — stopping training.")
            _ssh_run(pod, "tmux kill-session -t notelm-train", check=False)
            break
        time.sleep(args.poll)

    cmd_download(args)
    print(
        "Pod is still running (billing). Checkpoints pull attempted.\n"
        "Stop billing only when you mean it:\n"
        "  python3 scripts/runpod_train.py terminate"
    )


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
        cmd_monitor(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Pod is still running (billing).\n"
            "  python3 scripts/runpod_train.py status\n"
            "  python3 scripts/runpod_train.py terminate"
        )
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=[
        "check", "create", "wait", "status", "ssh-cmd", "sync", "setup",
        "fetch-gigamidi", "train", "logs", "monitor", "download", "terminate", "full",
    ])
    p.add_argument("--gpu", default=GPU_TYPE, help=f"GPU type id (default: {GPU_TYPE})")
    p.add_argument("--image", default=IMAGE, help="Docker image for the pod")
    p.add_argument("--epochs", type=int, default=40, help="Fine-tune epochs for recipes (default: 40)")
    p.add_argument("--train-args", help="Override full train.py argument string")
    p.add_argument("--recipe", choices=sorted(RECIPES),
                   help="Named multi-step training recipe (recommended: pretrain-finetune)")
    p.add_argument("--budget", type=float, default=200.0,
                   help="Hard USD cap for the run (default: 200)")
    p.add_argument("--max-hours", type=float, default=48.0,
                   help="Hard wall-clock cap from training start, not pod create (default: 48)")
    p.add_argument("--poll", type=int, default=120,
                   help="Monitor poll interval seconds (default: 120)")
    p.add_argument(
        "--max-files",
        type=int,
        default=50_000,
        help="GigaMIDI piano-ish file cap for fetch-gigamidi (default 50000)",
    )
    p.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="GPUs per pod (default 1; try 2 for H100 DDP, falls back to 1)",
    )
    p.add_argument(
        "--slot",
        default="default",
        help="Pod state file slot (default=.runpod_pod.json; canon=.runpod_pod_canon.json)",
    )
    args = p.parse_args()
    apply_slot(args.slot)
    if "h100" in POD_NAME and args.gpu == GPU_TYPE:
        args.gpu = "NVIDIA H100 80GB HBM3"
    elif "canon" in POD_NAME and args.gpu == GPU_TYPE:
        args.gpu = "NVIDIA RTX A6000"

    handlers = {
        "check": cmd_check, "create": cmd_create, "wait": cmd_wait,
        "status": cmd_status, "ssh-cmd": cmd_ssh_cmd, "sync": cmd_sync,
        "setup": cmd_setup, "fetch-gigamidi": cmd_fetch_gigamidi,
        "train": cmd_train, "logs": cmd_logs,
        "monitor": cmd_monitor, "download": cmd_download,
        "terminate": cmd_terminate, "full": cmd_full,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
