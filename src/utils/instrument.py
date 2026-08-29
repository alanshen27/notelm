"""GM-family instrument tags. Prefix tokens (INST_piano, …) condition continue/fill."""

from __future__ import annotations

INSTRUMENT_TAGS = (
    "none",
    "piano",
    "guitar",
    "bass",
    "strings",
    "ensemble",
    "organ",
    "brass",
    "reed",
    "pipe",
    "synth",
    "perc",
    "drums",
    "mixed",
)
INSTRUMENT_TOKEN_PREFIX = "INST_"
_TAG_SET = set(INSTRUMENT_TAGS)


def instrument_token(tag: str) -> str:
    key = (tag or "none").strip().lower()
    if key not in _TAG_SET:
        key = "none"
    return f"{INSTRUMENT_TOKEN_PREFIX}{key}"


def family_for_program(program: int, *, is_drum: bool = False) -> str:
    if is_drum:
        return "drums"
    p = int(program)
    if 0 <= p <= 7:
        return "piano"
    if 8 <= p <= 15 or 112 <= p <= 119:
        return "perc"
    if 16 <= p <= 23:
        return "organ"
    if 24 <= p <= 31:
        return "guitar"
    if 32 <= p <= 39:
        return "bass"
    if 40 <= p <= 47:
        return "strings"
    if 48 <= p <= 55:
        return "ensemble"
    if 56 <= p <= 63:
        return "brass"
    if 64 <= p <= 71:
        return "reed"
    if 72 <= p <= 79:
        return "pipe"
    if 80 <= p <= 103:
        return "synth"
    return "mixed"


def instrument_from_pretty(midi) -> str:
    """Majority GM family by note count. Mixed if no family has ≥70% of pitched notes."""
    counts: dict[str, int] = {}
    for inst in midi.instruments:
        n = len(inst.notes)
        if n <= 0:
            continue
        fam = family_for_program(inst.program, is_drum=bool(inst.is_drum))
        counts[fam] = counts.get(fam, 0) + n
    if not counts:
        return "none"
    pitched = {k: v for k, v in counts.items() if k != "drums"}
    use = pitched or counts
    top, n = max(use.items(), key=lambda kv: kv[1])
    total = sum(use.values())
    if total > 0 and n / total >= 0.7:
        return top
    return "mixed"
