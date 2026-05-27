import argparse
import os
import re
import time
from pathlib import Path

import torch

from inference import resolve_checkpoint, _search_roots
from models.lstm import LSTM
from utils.data import load_datasets, tokenizer
from utils.notify import notify_training_complete

WEIGHTS_PATH = "weights.pt"
PAD_ID = tokenizer.token_to_id["PAD"]


def _nvidia_gpu_present() -> bool:
    import shutil

    return shutil.which("nvidia-smi") is not None


def _warn_cuda_driver_mismatch() -> None:
    if torch.cuda.is_available() or not _nvidia_gpu_present():
        return
    print(
        "\nWARNING: nvidia-smi works but PyTorch cannot use CUDA (training on CPU).\n"
        "Common cause: PyPI torch on Linux bundles CUDA 13; your driver may be 12.x.\n"
        "Fix on this machine:  ./scripts/fix_cuda_torch.sh\n"
        "  (needs torch+*cu124*, not plain PyPI torch with CUDA 13 libs)\n"
    )


def _training_config(device: str) -> dict:
    """Tune batch size from available device memory."""
    cpus = os.cpu_count() or 1
    if device == "cuda":
        gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gb >= 35:
            return {"batch_size": 64, "accum_steps": 1, "num_workers": min(8, cpus)}
        if gb >= 20:
            return {"batch_size": 32, "accum_steps": 1, "num_workers": min(6, cpus)}
        if gb >= 12:
            return {"batch_size": 16, "accum_steps": 1, "num_workers": min(4, cpus)}
        return {"batch_size": 8, "accum_steps": 2, "num_workers": min(4, cpus)}
    if device == "mps":
        return {"batch_size": 4, "accum_steps": 4, "num_workers": 0}
    return {"batch_size": 2, "accum_steps": 8, "num_workers": min(2, cpus)}


def _infer_start_epoch(weights: Path) -> int | None:
    """Epoch folder epoch-N means N epochs done; resume at 0-based index N."""
    for part in weights.parts:
        m = re.fullmatch(r"epoch-(\d+)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def resolve_init_weights(spec: str) -> Path:
    """Path to .pt, or shorthand like epoch-40 / 40 (latest checkpoint in that folder)."""
    raw = spec.strip()
    m = re.fullmatch(r"(?:epoch-)?(\d+)", raw, re.IGNORECASE)
    if m:
        n = m.group(1)
        for root in _search_roots():
            folder = root / "checkpoints" / f"epoch-{n}"
            if not folder.is_dir():
                continue
            pts = sorted(folder.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                return pts[0].resolve()
        raise FileNotFoundError(
            f"No .pt checkpoint found under checkpoints/epoch-{n}/ "
            f"(searched: {', '.join(str(r) for r in _search_roots())})"
        )
    return resolve_checkpoint(raw)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train notelm LSTM (optionally resume from a checkpoint).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python train.py --epoch 40          # latest .pt in checkpoints/epoch-40/, continue as epoch 41
  python train.py -w epoch-40         # same
  python train.py -w checkpoints/epoch-40/20260527-132949.pt
  python train.py --epoch 40 --epochs 320
""",
    )
    p.add_argument(
        "--epoch",
        "-e",
        type=int,
        metavar="N",
        help="Resume: load latest weights from checkpoints/epoch-N/ and continue numbering",
    )
    p.add_argument(
        "--weights",
        "-w",
        metavar="PATH",
        help="Initial weights: .pt path, or epoch-N / N (same as --epoch N)",
    )
    p.add_argument(
        "--start-epoch",
        type=int,
        metavar="N",
        help="0-based epoch index to resume from (default: same as completed epoch folder N)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=320,
        help="Stop after this many epochs total (default: 320)",
    )
    args = p.parse_args()
    if args.epoch is not None and args.weights:
        p.error("use --epoch or --weights, not both")
    if args.epoch is not None:
        args.weights = str(args.epoch)
    return args


def main():
    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _warn_cuda_driver_mismatch()
    train_cfg = _training_config(device)

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        print(f"Using device: {device} ({props.name}, {vram_gb:.0f} GB)")
    else:
        print("Using device:", device)

    print(
        f"Training config: batch_size={train_cfg['batch_size']}, "
        f"accum_steps={train_cfg['accum_steps']} "
        f"(effective {train_cfg['batch_size'] * train_cfg['accum_steps']}), "
        f"num_workers={train_cfg['num_workers']}"
    )

    start_epoch = 0
    init_weights: Path | None = None
    if args.weights:
        init_weights = resolve_init_weights(args.weights)
        start_epoch = (
            args.start_epoch
            if args.start_epoch is not None
            else _infer_start_epoch(init_weights)
        )
        if start_epoch is None:
            start_epoch = 0
            print(
                f"Loaded weights from {init_weights} (could not infer epoch; "
                f"pass --start-epoch N to set checkpoint numbering)"
            )
        else:
            print(
                f"Resume from epoch {start_epoch + 1} using weights: {init_weights}"
            )
    elif args.start_epoch is not None:
        raise SystemExit("--start-epoch requires --weights")

    if args.epochs <= start_epoch:
        raise SystemExit(
            f"--epochs {args.epochs} must be greater than start epoch index {start_epoch}"
        )

    train_dataset, val_dataset = load_datasets()

    start = time.time()
    try:
        model = LSTM(
            train_dataset,
            val_dataset,
            tokenizer.vocab_size,
            device,
            PAD_ID,
            batch_size=train_cfg["batch_size"],
            accum_steps=train_cfg["accum_steps"],
            num_workers=train_cfg["num_workers"],
        ).to(device)

        if init_weights is not None:
            state = torch.load(init_weights, map_location=device, weights_only=True)
            model.load_state_dict(state)

        model.fit(epochs=args.epochs, start_epoch=start_epoch)

        torch.save(model.state_dict(), WEIGHTS_PATH)

        notify_training_complete(
            success=True,
            epochs=args.epochs,
            device=device,
            elapsed_s=time.time() - start,
            weights_path=WEIGHTS_PATH,
        )
    except Exception as exc:
        notify_training_complete(
            success=False,
            epochs=args.epochs,
            device=device,
            elapsed_s=time.time() - start,
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
