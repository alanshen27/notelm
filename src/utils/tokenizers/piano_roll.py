"""Piano-roll representation: one timestep column per TIMESTEP_MS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from utils.midi_timing import seconds_to_timesteps, timesteps_to_seconds
from utils.tokenizers.base import BaseMidiTokenizer


class PianoRollTokenizer(BaseMidiTokenizer):
    """
    Per-timestep tokens: TS_k then PITCH_p VEL_b for each active key at that step.
    Also exposes encode_piano_roll() → (num_steps, num_pitches) velocity matrix.
    """

    name = "piano_roll"

    def _build_vocab(self) -> None:
        for t in ("PAD", "BOS", "EOS", "UNK", "TS_STEP"):
            self._add(t)
        for v in range(self.cfg.velocity_bins):
            self._add(f"VEL_{v}")
        for p in range(self.cfg.min_pitch, self.cfg.max_pitch + 1):
            self._add(f"PITCH_{p}")

    @property
    def num_pitches(self) -> int:
        return self.cfg.max_pitch - self.cfg.min_pitch + 1

    def encode_piano_roll(self, midi_path: str | Path) -> np.ndarray:
        """Shape (T, num_pitches), values 0..127 (0 = off)."""
        notes = self._collect_notes(midi_path)
        if not notes:
            return np.zeros((1, self.num_pitches), dtype=np.uint8)

        end_t = max(seconds_to_timesteps(n.end) for n in notes) + 1
        roll = np.zeros((end_t, self.num_pitches), dtype=np.uint8)

        for note in notes:
            col = note.pitch - self.cfg.min_pitch
            t0 = seconds_to_timesteps(note.start)
            t1 = max(t0 + 1, seconds_to_timesteps(note.end))
            roll[t0:t1, col] = np.maximum(roll[t0:t1, col], note.velocity)

        return roll

    def _roll_to_tokens(self, roll: np.ndarray) -> list[str]:
        tokens = ["BOS"]
        for t in range(roll.shape[0]):
            active = False
            step_tokens: list[str] = []
            for col in range(roll.shape[1]):
                vel = int(roll[t, col])
                if vel > 0:
                    active = True
                    pitch = col + self.cfg.min_pitch
                    step_tokens.append(f"PITCH_{pitch}")
                    step_tokens.append(f"VEL_{self.velocity_to_bin(vel)}")
            if active:
                tokens.append("TS_STEP")
                tokens.extend(step_tokens)
        tokens.append("EOS")
        return tokens

    def encode_midi(self, midi_path: str | Path) -> list[int]:
        roll = self.encode_piano_roll(midi_path)
        return self.ids_from_tokens(self._roll_to_tokens(roll))

    def _tokens_to_roll(self, tokens: list[str]) -> np.ndarray:
        rows: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        for tok in tokens:
            if tok in ("BOS", "PAD", "UNK", "EOS"):
                continue
            if tok == "TS_STEP":
                if current:
                    rows.append(current)
                current = []
                continue
            if tok.startswith("PITCH_") and current is not None:
                pitch = int(tok.split("_", 1)[1])
                current.append((pitch, -1))
                continue
            if tok.startswith("VEL_") and current:
                vb = int(tok.split("_", 1)[1])
                pitch, _ = current[-1]
                current[-1] = (pitch, self.bin_to_velocity(vb))
        if current:
            rows.append(current)

        if not rows:
            return np.zeros((1, self.num_pitches), dtype=np.uint8)

        roll = np.zeros((len(rows), self.num_pitches), dtype=np.uint8)
        for t, pairs in enumerate(rows):
            for pitch, vel in pairs:
                if vel < 0:
                    continue
                roll[t, pitch - self.cfg.min_pitch] = vel
        return roll

    def tokens_to_midi(self, ids: list[int], output_path: str | Path) -> Path:
        roll = self._tokens_to_roll(self.decode_tokens(ids))
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)

        active: dict[int, tuple[int, int]] = {}
        for t in range(roll.shape[0]):
            time = timesteps_to_seconds(t)
            for col in range(roll.shape[1]):
                pitch = col + self.cfg.min_pitch
                vel = int(roll[t, col])
                prev = active.get(pitch)
                if vel > 0 and prev is None:
                    active[pitch] = (t, vel)
                elif vel == 0 and prev is not None:
                    t0, v = prev
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=v,
                            pitch=pitch,
                            start=timesteps_to_seconds(t0),
                            end=max(timesteps_to_seconds(t0) + 0.05, time),
                        )
                    )
                    del active[pitch]

        for pitch, (t0, v) in active.items():
            start = timesteps_to_seconds(t0)
            inst.notes.append(
                pretty_midi.Note(velocity=v, pitch=pitch, start=start, end=start + 0.5)
            )

        midi.instruments.append(inst)
        out = Path(output_path)
        midi.write(str(out))
        return out
