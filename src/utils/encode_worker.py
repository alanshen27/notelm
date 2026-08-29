"""Process-pool MIDI encode. No torch — spawn workers must not inherit CUDA."""

from __future__ import annotations

from pathlib import Path

_WORKER_TOK = None


def init_encode_worker(tokenizer_name: str) -> None:
    import sys

    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from utils.tokenizers import get_tokenizer

    global _WORKER_TOK
    _WORKER_TOK = get_tokenizer(tokenizer_name)


def encode_path(path: str) -> tuple[list, str]:
    from utils.instrument import instrument_from_pretty

    try:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(path)
        seq = _WORKER_TOK.encode_pretty_midi(midi)
        return seq, instrument_from_pretty(midi)
    except Exception:
        return [], "none"
