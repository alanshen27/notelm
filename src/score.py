"""MIDI → MusicXML for sheet-music rendering (requires music21)."""

from __future__ import annotations

from pathlib import Path

MAX_MEASURES = 32


def score_backend_available() -> bool:
    try:
        import music21  # noqa: F401
        return True
    except ImportError:
        return False


def _max_measure_number(score) -> int:
    from music21.stream.base import Measure

    numbers = [m.number for m in score.recurse().getElementsByClass(Measure)]
    return max(numbers) if numbers else 0


def midi_to_musicxml(
    midi_path: Path,
    xml_path: Path,
    *,
    max_measures: int = MAX_MEASURES,
) -> tuple[bool, str | None]:
    """
    Convert MIDI to MusicXML for OSMD.
    Returns (ok, error_message).
    """
    try:
        from music21 import converter
    except ImportError:
        return False, "music21 not installed — run: uv pip install music21"

    try:
        score = converter.parse(str(midi_path))
        last_measure = _max_measure_number(score)

        if last_measure > max_measures:
            score = score.measures(1, max_measures)

        xml_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("musicxml", fp=str(xml_path))
        return True, None
    except Exception as exc:
        return False, str(exc)
