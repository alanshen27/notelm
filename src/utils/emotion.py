"""EMOPIA-style valence–arousal quadrants, plus a none tag for unlabeled MIDI."""

from __future__ import annotations

import re
from pathlib import Path

# Q1 happy/excited, Q2 tense/angry, Q3 sad/dark, Q4 calm/peaceful (Russell 4Q).
EMOTION_TAGS = ("none", "Q1", "Q2", "Q3", "Q4")
EMOTION_TOKEN_PREFIX = "EMOTION_"
_Q_NAME = re.compile(r"(?:^|[_-])Q([1-4])(?:[_-]|$)", re.IGNORECASE)


def emotion_token(tag: str) -> str:
    key = (tag or "none").strip()
    if key.lower() in ("", "none", "neutral"):
        return f"{EMOTION_TOKEN_PREFIX}none"
    m = re.fullmatch(r"Q?([1-4])", key, re.IGNORECASE)
    if m:
        return f"{EMOTION_TOKEN_PREFIX}Q{m.group(1)}"
    if key.upper().startswith("Q") and key[1:2] in "1234":
        return f"{EMOTION_TOKEN_PREFIX}Q{key[1]}"
    return f"{EMOTION_TOKEN_PREFIX}none"


def emotion_from_path(path: str | Path) -> str:
    """Infer Q1–Q4 from EMOPIA-style names (`Q1_youtubeid_2.mid`)."""
    p = Path(path)
    for part in (p.name, *p.parts[::-1]):
        m = _Q_NAME.search(part)
        if m:
            return f"Q{m.group(1)}"
    return "none"
