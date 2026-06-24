"""Checkpoint layout: checkpoints/{model}/{tokenizer}/ — models never share a tree."""

from __future__ import annotations

import os
from pathlib import Path

from utils.tokenizers import TOKENIZER_NAMES

CHECKPOINTS_ROOT = Path("checkpoints")
MODEL_NAMES = ("lstm", "transformer")
DEFAULT_MODEL = "lstm"


def normalize_model(model_name: str) -> str:
    name = (model_name or DEFAULT_MODEL).strip().lower()
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}")
    return name


def normalize_tokenizer(tokenizer_name: str) -> str:
    name = tokenizer_name.strip().lower()
    if name not in TOKENIZER_NAMES:
        raise ValueError(f"Unknown tokenizer {name!r}")
    return name


def checkpoint_dir(model_name: str, tokenizer_name: str) -> Path:
    """e.g. checkpoints/lstm/event/"""
    return CHECKPOINTS_ROOT / normalize_model(model_name) / normalize_tokenizer(tokenizer_name)


def epoch_dir(model_name: str, tokenizer_name: str, epoch_1based: int) -> Path:
    """e.g. checkpoints/lstm/event/epoch-41/"""
    return checkpoint_dir(model_name, tokenizer_name) / f"epoch-{epoch_1based}"


def weights_path(model_name: str, tokenizer_name: str) -> Path:
    """e.g. checkpoints/lstm/event/weights.pt"""
    return checkpoint_dir(model_name, tokenizer_name) / "weights.pt"


def legacy_weights_path(tokenizer_name: str) -> Path:
    """Old: weights-event.pt in cwd."""
    return Path(f"weights-{tokenizer_name}.pt")


def legacy_tokenizer_dir(tokenizer_name: str) -> Path:
    """Old: checkpoints/event/ (no model level)."""
    return CHECKPOINTS_ROOT / normalize_tokenizer(tokenizer_name)


def default_model() -> str:
    return normalize_model(os.environ.get("NOTELM_MODEL", DEFAULT_MODEL))


def infer_model_from_path(path: Path) -> str:
    """Parse model from checkpoints/{model}/..."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "checkpoints" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt in MODEL_NAMES:
                return nxt
            if nxt in TOKENIZER_NAMES:
                return DEFAULT_MODEL  # legacy checkpoints/event/
            if nxt.startswith("epoch-"):
                return DEFAULT_MODEL  # legacy checkpoints/epoch-40/
    return DEFAULT_MODEL


def infer_tokenizer_from_path(path: Path) -> str:
    """Parse tokenizer from checkpoints/{model}/{tokenizer}/..."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part != "checkpoints" or i + 1 >= len(parts):
            continue
        nxt = parts[i + 1]
        if nxt in MODEL_NAMES and i + 2 < len(parts):
            candidate = parts[i + 2]
            if candidate in TOKENIZER_NAMES:
                return candidate
        if nxt in TOKENIZER_NAMES:
            return nxt
    return "event"
