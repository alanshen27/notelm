import argparse
import os
import re
import time
from pathlib import Path

import torch

from inference import resolve_checkpoint, _search_roots
from models.lstm import LSTM
from utils.checkpoints import (
    MODEL_NAMES,
    checkpoint_dir,
    default_model,
    epoch_dir,
    legacy_tokenizer_dir,
    normalize_model,
    weights_path as final_weights_path,
)
from utils.data import TRAIN_TOKENIZERS, load_datasets
from utils.notify import notify_training_complete
from utils.tokenizers import TOKENIZER_NAMES


def _nvidia_gpu_present() -> bool:
    import shutil

    return shutil.which("nvidia-smi") is not None


def _warn_cuda_driver_mismatch() -> None:
    if torch.cuda.is_available() or not _nvidia_gpu_present():
        return
    print(
        "\nWARNING: nvidia-smi works but PyTorch cannot use CUDA (training on CPU).\n"
        "Fix on this machine:  ./scripts/fix_cuda_torch.sh\n"
    )


def _training_config(device: str) -> dict:
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


def _paths_for_run(model_name: str, tokenizer_name: str) -> tuple[Path, Path]:
    ckpt = checkpoint_dir(model_name, tokenizer_name)
    ckpt.mkdir(parents=True, exist_ok=True)
    return final_weights_path(model_name, tokenizer_name), ckpt


def _infer_start_epoch(weights: Path) -> int | None:
    for part in weights.parts:
        m = re.fullmatch(r"epoch-(\d+)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def resolve_init_weights(spec: str, model_name: str, tokenizer_name: str) -> Path:
    raw = spec.strip()
    m = re.fullmatch(r"(?:epoch-)?(\d+)", raw, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        for root in _search_roots():
            for folder in (
                root / epoch_dir(model_name, tokenizer_name, n),
                root / legacy_tokenizer_dir(tokenizer_name) / f"epoch-{n}",
                root / "checkpoints" / f"epoch-{n}",
            ):
                if not folder.is_dir():
                    continue
                pts = sorted(folder.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if pts:
                    return pts[0].resolve()
        raise FileNotFoundError(
            f"No checkpoint for {model_name}/{tokenizer_name} epoch-{n} "
            f"(also checked legacy layouts)"
        )
    return resolve_checkpoint(raw)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train notelm (one or all tokenizers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
models:     {", ".join(MODEL_NAMES)}
tokenizers: {", ".join(TOKENIZER_NAMES)}

checkpoints -> checkpoints/{{model}}/{{tokenizer}}/epoch-N/

examples:
  python train.py --model lstm --tokenizer remi
  python train.py --all-tokenizers
  python train.py --model lstm --tokenizer raw --epoch 40
""",
    )
    p.add_argument(
        "--model",
        "-m",
        choices=MODEL_NAMES,
        default=default_model(),
        help="Architecture (default: lstm, or NOTELM_MODEL)",
    )
    p.add_argument(
        "--tokenizer",
        "-t",
        choices=TOKENIZER_NAMES,
        help="Input representation (default: event)",
    )
    p.add_argument(
        "--all-tokenizers",
        action="store_true",
        help=f"Train each tokenizer for this model: {', '.join(TRAIN_TOKENIZERS)}",
    )
    p.add_argument(
        "--only",
        metavar="NAMES",
        help="With --all-tokenizers, comma-separated subset",
    )
    p.add_argument("--epoch", "-e", type=int, metavar="N", help="Resume from epoch-N")
    p.add_argument("--weights", "-w", metavar="PATH", help="Initial .pt weights")
    p.add_argument("--start-epoch", type=int, metavar="N", help="0-based resume index")
    p.add_argument("--epochs", type=int, default=320, help="Total epochs (default: 320)")
    args = p.parse_args()

    if args.epoch is not None and args.weights:
        p.error("use --epoch or --weights, not both")
    if args.epoch is not None:
        args.weights = str(args.epoch)
    if args.all_tokenizers and args.tokenizer:
        p.error("use --tokenizer or --all-tokenizers, not both")
    if args.only and not args.all_tokenizers:
        p.error("--only requires --all-tokenizers")

    return args


def _tokenizer_list(args) -> list[str]:
    if args.all_tokenizers:
        if args.only:
            names = [s.strip().lower() for s in args.only.split(",") if s.strip()]
            bad = [n for n in names if n not in TOKENIZER_NAMES]
            if bad:
                raise SystemExit(f"Unknown tokenizer(s): {bad}")
            return names
        return list(TRAIN_TOKENIZERS)
    return [args.tokenizer or "event"]


def _build_model(model_name: str, train_dataset, val_dataset, tokenizer, device, pad_id, train_cfg, ckpt_root):
    model_name = normalize_model(model_name)
    if model_name == "lstm":
        return LSTM(
            train_dataset,
            val_dataset,
            tokenizer.vocab_size,
            device,
            pad_id,
            batch_size=train_cfg["batch_size"],
            accum_steps=train_cfg["accum_steps"],
            num_workers=train_cfg["num_workers"],
            checkpoint_dir=str(ckpt_root),
        )
    raise NotImplementedError(
        f"Model {model_name!r} is not implemented yet. "
        f"Checkpoints would live under checkpoints/{model_name}/{{tokenizer}}/"
    )


def train_one(
    model_name: str,
    tokenizer_name: str,
    *,
    device: str,
    train_cfg: dict,
    epochs: int,
    weights_spec: str | None,
    start_epoch_override: int | None,
) -> None:
    model_name = normalize_model(model_name)
    out_weights, ckpt_root = _paths_for_run(model_name, tokenizer_name)
    train_dataset, val_dataset, tokenizer = load_datasets(tokenizer_name)
    pad_id = tokenizer.token_to_id["PAD"]

    start_epoch = 0
    init_weights: Path | None = None
    if weights_spec:
        init_weights = resolve_init_weights(weights_spec, model_name, tokenizer_name)
        start_epoch = (
            start_epoch_override
            if start_epoch_override is not None
            else _infer_start_epoch(init_weights)
        )
        if start_epoch is None:
            start_epoch = 0
            print(f"[{model_name}/{tokenizer_name}] Loaded {init_weights} (start_epoch=0)")
        else:
            print(
                f"[{model_name}/{tokenizer_name}] Resume epoch {start_epoch + 1} "
                f"from {init_weights}"
            )

    if epochs <= start_epoch:
        raise SystemExit(
            f"[{model_name}/{tokenizer_name}] --epochs {epochs} must be > start {start_epoch}"
        )

    print(f"\n{'=' * 60}")
    print(f"Training  model={model_name}  tokenizer={tokenizer_name}")
    print(f"Checkpoints -> {ckpt_root.resolve()}/")
    print(f"{'=' * 60}")

    model = _build_model(
        model_name, train_dataset, val_dataset, tokenizer, device, pad_id, train_cfg, ckpt_root
    ).to(device)

    if init_weights is not None:
        state = torch.load(init_weights, map_location=device, weights_only=True)
        model.load_state_dict(state)

    model.fit(epochs=epochs, start_epoch=start_epoch)
    torch.save(model.state_dict(), out_weights)
    print(f"[{model_name}/{tokenizer_name}] saved {out_weights.resolve()}")


def main():
    args = parse_args()
    model_name = normalize_model(args.model)
    names = _tokenizer_list(args)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _warn_cuda_driver_mismatch()
    train_cfg = _training_config(device)

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"Using device: cuda ({props.name}, {props.total_memory / 1024**3:.0f} GB)")
    else:
        print("Using device:", device)

    print(
        f"Model: {model_name} | tokenizers: {', '.join(names)} | "
        f"batch={train_cfg['batch_size']} accum={train_cfg['accum_steps']}"
    )

    start = time.time()
    weights_arg = args.weights if len(names) == 1 else None
    if args.weights and len(names) > 1:
        print("Note: --weights/--epoch only applied when training a single tokenizer.")

    try:
        for i, tok in enumerate(names):
            if i > 0 and device == "cuda":
                torch.cuda.empty_cache()
            train_one(
                model_name,
                tok,
                device=device,
                train_cfg=train_cfg,
                epochs=args.epochs,
                weights_spec=weights_arg,
                start_epoch_override=args.start_epoch,
            )

        notify_training_complete(
            success=True,
            epochs=args.epochs,
            device=device,
            elapsed_s=time.time() - start,
            weights_path=f"checkpoints/{model_name}/{{tokenizer}}/weights.pt",
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
