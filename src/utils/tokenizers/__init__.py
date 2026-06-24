from __future__ import annotations

import os

from utils.midi_timing import TIMESTEP_MS, TIMESTEP_SEC
from utils.tokenizers.base import BaseMidiTokenizer, TokenizerConfig
from utils.tokenizers.event import EventTokenizer
from utils.tokenizers.piano_roll import PianoRollTokenizer
from utils.tokenizers.raw import RawMidiTokenizer
from utils.tokenizers.remi import RemiTokenizer

TOKENIZER_NAMES = ("event", "raw", "remi", "piano_roll")

_REGISTRY: dict[str, type[BaseMidiTokenizer]] = {
    "event": EventTokenizer,
    "raw": RawMidiTokenizer,
    "remi": RemiTokenizer,
    "piano_roll": PianoRollTokenizer,
}


def get_tokenizer(name: str | None = None, config: TokenizerConfig | None = None) -> BaseMidiTokenizer:
    key = (name or os.environ.get("NOTELM_TOKENIZER", "event")).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer {key!r}. Choose from: {', '.join(TOKENIZER_NAMES)}")
    return _REGISTRY[key](config)


# Backward-compatible alias (default training / inference format).
MidiTokenizer = EventTokenizer
MidiTokenizerConfig = TokenizerConfig

__all__ = [
    "TIMESTEP_MS",
    "TIMESTEP_SEC",
    "TOKENIZER_NAMES",
    "BaseMidiTokenizer",
    "TokenizerConfig",
    "EventTokenizer",
    "RawMidiTokenizer",
    "RemiTokenizer",
    "PianoRollTokenizer",
    "MidiTokenizer",
    "MidiTokenizerConfig",
    "get_tokenizer",
]
