"""Musically weighted next-token loss: pitch + meter matter more than velocity."""

from __future__ import annotations

import torch
import torch.nn.functional as F


PC_AUX = 0.5


def family_weight(token: str) -> float:
    if token.startswith("Pitch_") or token.startswith("NOTE_ON_"):
        return 2.5
    if token.startswith("Position_") or token in ("Bar",) or token.startswith("Bar_"):
        return 2.0
    if token.startswith("TIME_SHIFT_"):
        return 1.8
    if token.startswith("NOTE_OFF_") or token.startswith("Duration_"):
        return 1.3
    if token.startswith("Velocity_") or token.startswith("VELOCITY_"):
        return 0.35
    if token.startswith("EMOTION_") or token.startswith("INST_"):
        return 0.8
    return 1.0


def class_weights(tokenizer, vocab_size: int, device: torch.device) -> torch.Tensor:
    w = torch.ones(vocab_size, device=device)
    for tok, idx in tokenizer.token_to_id.items():
        if 0 <= idx < vocab_size:
            w[idx] = family_weight(tok)
    return w


def pitch_lookups(
    tokenizer, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """pc_of_id[v] = pitch class or -1. pc_member[12, vocab] bool mask."""
    pc_of_id = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    pc_member = torch.zeros(12, vocab_size, dtype=torch.bool, device=device)
    for tok, idx in tokenizer.token_to_id.items():
        if idx < 0 or idx >= vocab_size:
            continue
        pitch = None
        if tok.startswith("Pitch_"):
            pitch = int(tok.rsplit("_", 1)[-1])
        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.rsplit("_", 1)[-1])
        if pitch is None:
            continue
        pc = pitch % 12
        pc_of_id[idx] = pc
        pc_member[pc, idx] = True
    return pc_of_id, pc_member


def pitch_class_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pc_of_id: torch.Tensor,
    pc_member: torch.Tensor,
) -> torch.Tensor:
    """12-way chroma CE on pitch tokens so C vs C# is extra-expensive."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    targets = pc_of_id[flat_labels]
    mask = targets >= 0
    if not mask.any():
        return logits.new_zeros(())
    row = flat_logits[mask]
    parts = []
    for c in range(12):
        ids = pc_member[c]
        if not bool(ids.any()):
            parts.append(row.new_full((row.size(0),), -1e4))
        else:
            parts.append(torch.logsumexp(row[:, ids], dim=-1))
    stacked = torch.stack(parts, dim=-1)
    return F.cross_entropy(stacked, targets[mask])
