#!/usr/bin/env python3
"""Copy the lowest-val-loss epoch checkpoint (not the latest).

Parses `Epoch N/M | train loss: … | val loss: … | saved PATH` lines. Ties
go to the earlier epoch.

`--watch` keeps rewriting dest until the pop fine-tune has loaded weights, so
a later `cp weights.pt pretrain.pt` cannot sneak the last epoch back in.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+train loss:\s+([\d.]+)\s+\|\s+"
    r"val loss:\s+([\d.]+)\s+\|\s+saved\s+(\S+\.pt)"
)
BEGIN_RE = re.compile(r"Begin training:")
DATASET_RE = re.compile(r"Dataset ready:")
TRAIN_HDR_RE = re.compile(r"Training  model=")
LOADED_RE = re.compile(r"Loaded .+/pretrain\.pt")


def _strip(text: str) -> str:
    return ANSI.sub("", text)


def parse_first_run(text: str) -> list[dict]:
    """Epochs from the first training job in the log (pretrain)."""
    body = _strip(text)
    begins = [m.start() for m in BEGIN_RE.finditer(body)]
    chunk = body if not begins else body[begins[0] : (begins[1] if len(begins) > 1 else len(body))]
    rows = []
    for m in EPOCH_RE.finditer(chunk):
        rows.append(
            {
                "epoch": int(m.group(1)),
                "total": int(m.group(2)),
                "train": float(m.group(3)),
                "val": float(m.group(4)),
                "saved": m.group(5),
            }
        )
    return rows


def resolve_ckpt(saved: str, ckpt_dir: Path) -> Path:
    p = Path(saved)
    if p.is_file():
        return p
    src = ckpt_dir.parent.parent.parent  # …/src
    for root in (Path.cwd(), src, ckpt_dir):
        cand = p if p.is_absolute() else root / p
        if cand.is_file():
            return cand
    m = re.search(r"epoch-(\d+)", saved)
    if m:
        pts = sorted((ckpt_dir / f"epoch-{m.group(1)}").glob("*.pt"))
        if pts:
            return pts[-1]
    raise FileNotFoundError(saved)


def pick_best(rows: list[dict]) -> dict:
    return min(rows, key=lambda r: (r["val"], r["epoch"]))


def copy_best(rows: list[dict], ckpt_dir: Path, dests: list[Path]) -> dict:
    best = pick_best(rows)
    src = resolve_ckpt(best["saved"], ckpt_dir)
    written: list[Path] = []
    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() == src.resolve():
            continue
        if dest.is_file() and filecmp.cmp(src, dest, shallow=True):
            continue
        shutil.copy2(src, dest)
        written.append(dest)
    meta = {
        "epoch": best["epoch"],
        "total": best["total"],
        "train": best["train"],
        "val": best["val"],
        "src": str(src),
        "dests": [str(d) for d in dests],
    }
    sidecar = dests[0].with_name("best.json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    if written:
        print(
            f"best val {best['val']:.4f} at epoch {best['epoch']}/{best['total']} "
            f"({src}) -> {', '.join(str(d) for d in written)}",
            flush=True,
        )
    return best


def _self_test() -> None:
    sample = """
Begin training: transformer (25.3M params)
Epoch 17/40 | train loss: 1.1061 | val loss: 1.4135 | saved checkpoints/transformer/remi/epoch-17/a.pt
Epoch 18/40 | train loss: 1.0928 | val loss: 1.4015 | saved checkpoints/transformer/remi/epoch-18/b.pt
Epoch 24/40 | train loss: 1.0284 | val loss: 1.4279 | saved checkpoints/transformer/remi/epoch-24/c.pt
Begin training: transformer (25.3M params)
Epoch 1/40 | train loss: 9.9999 | val loss: 0.0001 | saved checkpoints/transformer/remi/epoch-1/sneak.pt
"""
    rows = parse_first_run(sample)
    best = pick_best(rows)
    assert [r["epoch"] for r in rows] == [17, 18, 24], rows
    assert best["epoch"] == 18 and best["val"] == 1.4015, best
    print("self-test ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path)
    ap.add_argument("--ckpt-dir", type=Path)
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--also", type=Path, action="append", default=[])
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.log is None or args.ckpt_dir is None or args.dest is None:
        ap.error("--log, --ckpt-dir, and --dest are required")

    dests = [args.dest, *args.also]
    if not args.log.is_file():
        logs = sorted(args.log.parent.glob("train-*.log"), key=lambda p: p.stat().st_mtime)
        if logs:
            args.log = logs[-1]

    def run_once() -> tuple[list[dict], str]:
        text = args.log.read_text(errors="replace") if args.log.is_file() else ""
        return parse_first_run(text), text

    if not args.watch:
        rows, _ = run_once()
        if not rows:
            print("No epoch lines in log.", file=sys.stderr)
            return 1
        copy_best(rows, args.ckpt_dir, dests)
        return 0

    last_stamp = None
    finetune_at: float | None = None
    while True:
        rows, text = run_once()
        body = _strip(text)
        pretrain_done = bool(rows) and any(r["epoch"] >= r["total"] for r in rows)
        if rows:
            best = pick_best(rows)
            stamp = (best["epoch"], best["val"], best["saved"])
            if stamp != last_stamp or pretrain_done:
                copy_best(rows, args.ckpt_dir, dests)
                last_stamp = stamp
        n_dataset = len(DATASET_RE.findall(body))
        n_begin = len(BEGIN_RE.findall(body))
        n_hdr = len(TRAIN_HDR_RE.findall(body))
        finetune_loading = n_dataset >= 2 or n_hdr >= 2 or n_begin >= 2 or bool(LOADED_RE.search(body))
        if finetune_loading:
            if rows:
                copy_best(rows, args.ckpt_dir, dests)
            if finetune_at is None:
                finetune_at = time.time()
                print("Finetune is loading — holding best pretrain in place.", flush=True)
            if time.time() - finetune_at >= 90:
                print("Chooser done.", flush=True)
                return 0
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
