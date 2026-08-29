"""REMI-style tokens: Bar, Position, Pitch, Velocity, Duration.

Positions are 16th notes in 4/4 using the file's tempo — not 20 ms ticks.
A single repeating Bar token marks each new measure (Huang & Yang 2020).
"""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from utils.emotion import EMOTION_TAGS, emotion_token
from utils.instrument import INSTRUMENT_TAGS, instrument_token
from utils.tokenizers.base import BaseMidiTokenizer

POSITIONS_PER_BAR = 16  # 16th-note grid in 4/4
MAX_DURATION_16THS = 32  # two bars
DEFAULT_BPM = 120.0


def _bpm_at(midi: pretty_midi.PrettyMIDI, t: float) -> float:
    times, tempos = midi.get_tempo_changes()
    if len(tempos) == 0:
        try:
            return float(midi.estimate_tempo())
        except Exception:
            return DEFAULT_BPM
    idx = int((times <= t).sum()) - 1
    return float(tempos[max(0, min(idx, len(tempos) - 1))])


def _sixteenth_sec(bpm: float) -> float:
    return 60.0 / max(bpm, 1.0) / 4.0


class RemiTokenizer(BaseMidiTokenizer):
    name = "remi"

    def _build_vocab(self) -> None:
        for t in ("PAD", "BOS", "EOS", "UNK"):
            self._add(t)
        self._add("Bar")
        for pos in range(POSITIONS_PER_BAR):
            self._add(f"Position_{pos}")
        for p in range(self.cfg.min_pitch, self.cfg.max_pitch + 1):
            self._add(f"Pitch_{p}")
        for v in range(self.cfg.velocity_bins):
            self._add(f"Velocity_{v}")
        for d in range(1, MAX_DURATION_16THS + 1):
            self._add(f"Duration_{d}")
        for tag in EMOTION_TAGS:
            self._add(emotion_token(tag))
        self._add("SPAN")
        self._add("SEP")
        for tag in INSTRUMENT_TAGS:
            self._add(instrument_token(tag))

    def encode_pretty_midi(self, midi: pretty_midi.PrettyMIDI) -> list[int]:
        notes: list[pretty_midi.Note] = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if self.cfg.min_pitch <= note.pitch <= self.cfg.max_pitch:
                    notes.append(note)
        notes.sort(key=lambda n: (n.start, n.pitch))

        tokens = ["BOS"]
        last_bar, last_pos = -1, -1

        for note in notes:
            bpm = _bpm_at(midi, note.start)
            step = _sixteenth_sec(bpm)
            t_on = max(0, round(note.start / step))
            t_off = max(t_on + 1, round(note.end / step))
            duration = min(t_off - t_on, MAX_DURATION_16THS)
            bar, pos = divmod(t_on, POSITIONS_PER_BAR)

            while last_bar < bar:
                tokens.append("Bar")
                last_bar += 1
                last_pos = -1
            if pos != last_pos:
                tokens.append(f"Position_{pos}")
                last_pos = pos

            tokens.append(f"Pitch_{note.pitch}")
            tokens.append(f"Velocity_{self.velocity_to_bin(note.velocity)}")
            tokens.append(f"Duration_{duration}")

        tokens.append("EOS")
        return self.ids_from_tokens(tokens)

    def tokens_to_midi(
        self, ids: list[int], output_path: str | Path, *, tempo: float | None = None
    ) -> Path:
        tokens = self.decode_tokens(ids)
        bpm = float(tempo) if tempo else DEFAULT_BPM
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        inst = pretty_midi.Instrument(program=0)
        step = _sixteenth_sec(bpm)

        bar, pos = 0, 0
        seen_bar = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("BOS", "PAD", "UNK", "SPAN", "SEP") or tok.startswith(
                ("EMOTION_", "INST_")
            ):
                i += 1
                continue
            if tok == "EOS":
                break
            if tok == "Bar":
                if seen_bar:
                    bar += 1
                seen_bar = True
                pos = 0
                i += 1
                continue
            if tok.startswith("Position_"):
                pos = int(tok.split("_", 1)[1]) % POSITIONS_PER_BAR
                i += 1
                continue
            if (
                tok.startswith("Pitch_")
                and i + 2 < len(tokens)
                and tokens[i + 1].startswith("Velocity_")
                and tokens[i + 2].startswith("Duration_")
            ):
                pitch = int(tok.split("_", 1)[1])
                vel = self.bin_to_velocity(int(tokens[i + 1].split("_", 1)[1]))
                dur = int(tokens[i + 2].split("_", 1)[1])
                start_t = bar * POSITIONS_PER_BAR + pos
                start = start_t * step
                end = (start_t + max(1, dur)) * step
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=vel,
                        pitch=pitch,
                        start=start,
                        end=max(start + step, end),
                    )
                )
                i += 3
                continue
            i += 1

        midi.instruments.append(inst)
        out = Path(output_path)
        midi.write(str(out))
        return out
