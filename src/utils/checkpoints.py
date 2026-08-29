"""Checkpoint layout: checkpoints/transformer/{tokenizer}/"""

from __future__ import annotations

import os
from pathlib import Path

from utils.tokenizers import TOKENIZER_NAMES

CHECKPOINTS_ROOT = Path("checkpoints")
MODEL_NAMES = ("transformer", "canon")
DEFAULT_MODEL = "transformer"
# Filenames for named snapshots from 2026-08-29 on. Product names (Prelude, …)
# stay on the site; these are the checkpoint code names.
CHECKPOINT_CODENAMES = ("invention", "etude", "prelude", "chaconne", "canon", "sinfonia")


def normalize_model(model_name: str) -> str:
    name = (model_name or DEFAULT_MODEL).strip().lower()
    if name == "lstm":
        name = DEFAULT_MODEL
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}")
    return name


def normalize_tokenizer(tokenizer_name: str) -> str:
    name = tokenizer_name.strip().lower()
    if name not in TOKENIZER_NAMES:
        raise ValueError(f"Unknown tokenizer {name!r}")
    return name


def normalize_codename(codename: str) -> str:
    name = (codename or "").strip().lower()
    if name not in CHECKPOINT_CODENAMES:
        raise ValueError(
            f"Unknown checkpoint code name {name!r}. "
            f"Choose from: {', '.join(CHECKPOINT_CODENAMES)}"
        )
    return name


def checkpoint_dir(model_name: str, tokenizer_name: str) -> Path:
    return CHECKPOINTS_ROOT / normalize_model(model_name) / normalize_tokenizer(tokenizer_name)


def epoch_dir(model_name: str, tokenizer_name: str, epoch_1based: int) -> Path:
    return checkpoint_dir(model_name, tokenizer_name) / f"epoch-{epoch_1based}"


def weights_path(model_name: str, tokenizer_name: str) -> Path:
    return checkpoint_dir(model_name, tokenizer_name) / "weights.pt"


def legacy_weights_path(tokenizer_name: str) -> Path:
    return Path(f"weights-{tokenizer_name}.pt")


def legacy_tokenizer_dir(tokenizer_name: str) -> Path:
    return CHECKPOINTS_ROOT / normalize_tokenizer(tokenizer_name)


def default_model() -> str:
    return normalize_model(os.environ.get("NOTELM_MODEL", DEFAULT_MODEL))


def infer_model_from_path(path: Path) -> str:
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "checkpoints" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt in MODEL_NAMES:
                return nxt
    return DEFAULT_MODEL


def infer_tokenizer_from_path(path: Path) -> str:
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
    return "remi"
