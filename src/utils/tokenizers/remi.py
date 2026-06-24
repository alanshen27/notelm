"""REMI-style tokens: Bar, Position, Pitch, Velocity, Duration (no NOTE_OFF)."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from utils.midi_timing import seconds_to_timesteps, timesteps_to_seconds
from utils.tokenizers.base import BaseMidiTokenizer

# 4/4 grid: 16 positions per bar at 20 ms → 16 * 20 = 320 ms per bar slice at default tempo mapping
POSITIONS_PER_BAR = 16
MAX_DURATION_STEPS = 200
MAX_BARS = 512


class RemiTokenizer(BaseMidiTokenizer):
    name = "remi"

    def _build_vocab(self) -> None:
        for t in ("PAD", "BOS", "EOS", "UNK"):
            self._add(t)
        for b in range(MAX_BARS):
            self._add(f"Bar_{b}")
        for pos in range(POSITIONS_PER_BAR):
            self._add(f"Position_{pos}")
        for p in range(self.cfg.min_pitch, self.cfg.max_pitch + 1):
            self._add(f"Pitch_{p}")
        for v in range(self.cfg.velocity_bins):
            self._add(f"Velocity_{v}")
        for d in range(1, MAX_DURATION_STEPS + 1):
            self._add(f"Duration_{d}")

    def _time_to_bar_pos(self, timestep: int) -> tuple[int, int]:
        steps_per_bar = POSITIONS_PER_BAR
        bar = timestep // steps_per_bar
        pos = timestep % steps_per_bar
        return min(bar, MAX_BARS - 1), pos

    def encode_midi(self, midi_path: str | Path) -> list[int]:
        notes = self._collect_notes(midi_path)
        tokens = ["BOS"]
        last_bar, last_pos = -1, -1

        for note in notes:
            t_on = seconds_to_timesteps(note.start)
            t_off = max(t_on + 1, seconds_to_timesteps(note.end))
            duration = min(t_off - t_on, MAX_DURATION_STEPS)
            bar, pos = self._time_to_bar_pos(t_on)

            if bar != last_bar:
                tokens.append(f"Bar_{bar}")
                last_bar, last_pos = bar, -1
            if pos != last_pos:
                tokens.append(f"Position_{pos}")
                last_pos = pos

            tokens.append(f"Pitch_{note.pitch}")
            tokens.append(f"Velocity_{self.velocity_to_bin(note.velocity)}")
            tokens.append(f"Duration_{duration}")

        tokens.append("EOS")
        return self.ids_from_tokens(tokens)

    def tokens_to_midi(self, ids: list[int], output_path: str | Path) -> Path:
        tokens = self.decode_tokens(ids)
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)

        bar, pos = 0, 0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("BOS", "PAD", "UNK"):
                i += 1
                continue
            if tok == "EOS":
                break
            if tok.startswith("Bar_"):
                bar = int(tok.split("_", 1)[1])
                i += 1
                continue
            if tok.startswith("Position_"):
                pos = int(tok.split("_", 1)[1])
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
                start = timesteps_to_seconds(start_t)
                end = timesteps_to_seconds(start_t + dur)
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=vel,
                        pitch=pitch,
                        start=start,
                        end=max(start + 0.05, end),
                    )
                )
                i += 3
                continue
            i += 1

        midi.instruments.append(inst)
        out = Path(output_path)
        midi.write(str(out))
        return out
