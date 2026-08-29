import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path


def _require_venv_python() -> None:
    """Fail fast when `python` is not the project venv (common on SSH boxes)."""
    root = Path(__file__).resolve().parents[1]
    venv_py = root / ".venv" / "bin" / "python"
    if venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
        print(
            f"Wrong Python: {sys.executable}\n"
            f"Project venv: {venv_py}\n\n"
            "Fix:\n"
            f"  cd {root} && source .venv/bin/activate && cd src && python train.py ...\n"
            f"  ./scripts/train.sh --dataset pop --epochs 40",
            file=sys.stderr,
        )
        raise SystemExit(1)


_require_venv_python()

import torch

from inference import resolve_checkpoint, _search_roots
from models.transformer import Transformer
from models.canon import Canon, DEC_MAX_LEN, ENC_MAX_LEN
from utils.checkpoints import (
    CHECKPOINT_CODENAMES,
    MODEL_NAMES,
    checkpoint_dir,
    default_model,
    epoch_dir,
    legacy_tokenizer_dir,
    normalize_codename,
    normalize_model,
    weights_path as final_weights_path,
)
from utils.data import DATASET_NAMES, load_datasets, load_span_datasets, normalize_dataset
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


def _resolve_device(requested: str | None = None) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise SystemExit("MPS requested but not available")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    elif not torch.cuda.is_available():
        return "mps" if torch.backends.mps.is_available() else "cpu"

    if torch.cuda.is_available() and requested in (None, "cuda"):
        try:
            torch.randn(2, device="cuda")
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "no kernel image" in str(e).lower():
                cap = torch.cuda.get_device_capability(0)
                name = torch.cuda.get_device_name(0)
                raise SystemExit(
                    f"\nCUDA kernel error on {name} (cap {cap}).\n"
                    "Blackwell / RTX 50 / RTX PRO needs PyTorch cu128, not cu124.\n\n"
                    "Fix:\n"
                    "  cd /notelm && ./scripts/fix_cuda_torch.sh\n"
                ) from e
            raise
        return "cuda"

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _init_distributed() -> None:
    if "LOCAL_RANK" not in os.environ:
        return
    import torch.distributed as dist

    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)


def _training_config(device: str, model_name: str = "transformer") -> dict:
    cpus = os.cpu_count() or 1
    canon = model_name == "canon"
    if device == "cuda":
        gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if canon:
            if gb >= 70:
                return {"batch_size": 32, "accum_steps": 1, "num_workers": min(8, cpus)}
            if gb >= 40:
                return {"batch_size": 16, "accum_steps": 2, "num_workers": min(8, cpus)}
            if gb >= 20:
                return {"batch_size": 8, "accum_steps": 2, "num_workers": min(6, cpus)}
            return {"batch_size": 4, "accum_steps": 4, "num_workers": min(4, cpus)}
        if gb >= 35:
            return {"batch_size": 64, "accum_steps": 1, "num_workers": min(8, cpus)}
        if gb >= 20:
            return {"batch_size": 32, "accum_steps": 1, "num_workers": min(6, cpus)}
        if gb >= 12:
            return {"batch_size": 16, "accum_steps": 1, "num_workers": min(4, cpus)}
        return {"batch_size": 8, "accum_steps": 2, "num_workers": min(4, cpus)}
    if device == "mps":
        return {"batch_size": 2 if canon else 4, "accum_steps": 4, "num_workers": 0}
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
        description="Train the notelm Transformer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
tokenizers: {", ".join(TOKENIZER_NAMES)}
datasets:   {", ".join(DATASET_NAMES)}

examples:
  python train.py --dataset pop909 --tokenizer event --epochs 40
  python train.py --dataset pop --epochs 40 --codename etude
  python train.py --dataset maestro_full --epochs 20
  python train.py --model canon --dataset instruments --tokenizer remi --epochs 40
  python train.py --model canon --dataset piano --tokenizer remi --epochs 40 --codename canon
""",
    )
    p.add_argument(
        "--dataset",
        "-d",
        choices=DATASET_NAMES,
        default=None,
        help="Training corpus (default: pop909, or NOTELM_DATASET)",
    )
    p.add_argument(
        "--model",
        "-m",
        choices=MODEL_NAMES,
        default=default_model(),
        help="Architecture (transformer causal LM, or canon enc-dec infill)",
    )
    p.add_argument(
        "--tokenizer",
        "-t",
        choices=TOKENIZER_NAMES,
        default="remi",
        help="Input representation (default: remi)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Device (default: auto with CUDA kernel smoke test)",
    )
    p.add_argument("--epoch", "-e", type=int, metavar="N", help="Resume from epoch-N")
    p.add_argument("--weights", "-w", metavar="PATH", help="Initial .pt weights")
    p.add_argument("--start-epoch", type=int, metavar="N", help="0-based resume index")
    p.add_argument("--epochs", type=int, default=40, help="Total epochs (default: 40)")
    p.add_argument(
        "--seq-len",
        type=int,
        metavar="N",
        help="Override window length (default: 4096 event / 2048 remi)",
    )
    p.add_argument(
        "--limit-files",
        type=int,
        metavar="N",
        help="Use only N files (smoke tests)",
    )
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Encode MIDI to data/token_cache and exit (no GPU train)",
    )
    p.add_argument(
        "--lr",
        type=float,
        metavar="LR",
        help="Learning rate override (e.g. 1e-4 for fine-tuning)",
    )
    p.add_argument(
        "--codename",
        metavar="NAME",
        choices=CHECKPOINT_CODENAMES,
        help=(
            "Also copy best weights to {name}.pt "
            f"({', '.join(CHECKPOINT_CODENAMES)})"
        ),
    )
    args = p.parse_args()

    if args.epoch is not None and args.weights:
        p.error("use --epoch or --weights, not both")
    if args.epoch is not None:
        args.weights = str(args.epoch)
    return args


def train_one(
    model_name: str,
    tokenizer_name: str,
    *,
    device: str,
    train_cfg: dict,
    epochs: int,
    weights_spec: str | None,
    start_epoch_override: int | None,
    dataset: str | None = None,
    seq_len: int | None = None,
    limit_files: int | None = None,
    codename: str | None = None,
    cache_only: bool = False,
) -> None:
    model_name = normalize_model(model_name)
    if model_name == "canon" and tokenizer_name != "remi":
        raise SystemExit("canon requires --tokenizer remi (bar spans)")
    out_weights, ckpt_root = _paths_for_run(model_name, tokenizer_name)
    if model_name == "canon":
        train_dataset, val_dataset, tokenizer = load_span_datasets(
            tokenizer_name,
            dataset=dataset,
            seq_len=seq_len,
            limit_files=limit_files,
            enc_len=ENC_MAX_LEN,
            dec_len=DEC_MAX_LEN,
        )
    else:
        train_dataset, val_dataset, tokenizer = load_datasets(
            tokenizer_name, dataset=dataset, seq_len=seq_len, limit_files=limit_files
        )
    if cache_only:
        print(f"Token cache ready for {model_name}/{tokenizer_name} dataset={dataset}")
        return
    pad_id = tokenizer.token_to_id["PAD"]
    effective_seq_len = getattr(train_dataset, "seq_len", None) or (
        train_dataset.source.seq_len if hasattr(train_dataset, "source") else 2048
    )

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

    if model_name == "canon":
        model = Canon(
            train_dataset,
            val_dataset,
            tokenizer.vocab_size,
            device,
            pad_id,
            tokenizer=tokenizer,
            batch_size=train_cfg["batch_size"],
            accum_steps=train_cfg["accum_steps"],
            num_workers=train_cfg["num_workers"],
            checkpoint_dir=str(ckpt_root),
            lr=train_cfg.get("lr"),
            enc_max_len=ENC_MAX_LEN,
            dec_max_len=DEC_MAX_LEN,
        ).to(device)
    else:
        model = Transformer(
            train_dataset,
            val_dataset,
            tokenizer.vocab_size,
            device,
            pad_id,
            tokenizer=tokenizer,
            batch_size=train_cfg["batch_size"],
            accum_steps=train_cfg["accum_steps"],
            num_workers=train_cfg["num_workers"],
            checkpoint_dir=str(ckpt_root),
            max_len=effective_seq_len,
            lr=train_cfg.get("lr"),
        ).to(device)

    if init_weights is not None:
        state = torch.load(init_weights, map_location=device, weights_only=True)
        # New emotion rows sit at the end of the vocab; copy overlapping weights.
        current = model.state_dict()
        for key, tensor in state.items():
            if key in current and current[key].shape == tensor.shape:
                current[key] = tensor
            elif key in current and current[key].ndim == tensor.ndim:
                slices = tuple(slice(0, min(a, b)) for a, b in zip(current[key].shape, tensor.shape))
                current[key][slices] = tensor[slices]
        model.load_state_dict(current)

    if torch.distributed.is_initialized():
        local = int(os.environ["LOCAL_RANK"])
        model.ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], output_device=local
        )

    model.fit(epochs=epochs, start_epoch=start_epoch)
    rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if rank0:
        torch.save(model.state_dict(), out_weights)
        print(f"[{model_name}/{tokenizer_name}] saved {out_weights.resolve()}")
        if codename:
            named = ckpt_root / f"{normalize_codename(codename)}.pt"
            shutil.copy2(out_weights, named)
            print(f"[{model_name}/{tokenizer_name}] saved {named.resolve()}")


def main():
    args = parse_args()
    model_name = normalize_model(args.model)
    dataset = normalize_dataset(args.dataset)

    if args.cache_only:
        print(f"Cache-only encode: model={model_name} dataset={dataset} tokenizer={args.tokenizer}")
        train_one(
            model_name,
            args.tokenizer,
            device="cpu",
            train_cfg={"batch_size": 1, "accum_steps": 1, "num_workers": 0},
            epochs=1,
            weights_spec=None,
            start_epoch_override=None,
            dataset=dataset,
            seq_len=args.seq_len,
            limit_files=args.limit_files,
            cache_only=True,
        )
        return

    _init_distributed()
    device = _resolve_device(None if args.device == "auto" else args.device)
    _warn_cuda_driver_mismatch()
    train_cfg = _training_config(device, model_name)
    if args.lr:
        train_cfg["lr"] = args.lr

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"Using device: cuda ({props.name}, {props.total_memory / 1024**3:.0f} GB)")
    else:
        print("Using device:", device)

    print(
        f"Model: {model_name} | dataset: {dataset} | tokenizer: {args.tokenizer} | "
        f"batch={train_cfg['batch_size']} accum={train_cfg['accum_steps']}"
    )

    start = time.time()
    try:
        train_one(
            model_name,
            args.tokenizer,
            device=device,
            train_cfg=train_cfg,
            epochs=args.epochs,
            weights_spec=args.weights,
            start_epoch_override=args.start_epoch,
            dataset=dataset,
            seq_len=args.seq_len,
            limit_files=args.limit_files,
            codename=args.codename,
        )
        notify_training_complete(
            success=True,
            epochs=args.epochs,
            device=device,
            elapsed_s=time.time() - start,
            weights_path=f"checkpoints/{model_name}/{args.tokenizer}/weights.pt",
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
