"""Event tokenizer: NOTE_ON / NOTE_OFF / VELOCITY / TIME_SHIFT (original notelm format)."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from utils.emotion import EMOTION_TAGS, emotion_token
from utils.instrument import INSTRUMENT_TAGS, instrument_token
from utils.midi_timing import MAX_TIME_SHIFT_STEPS, seconds_to_timesteps, timesteps_to_seconds
from utils.tokenizers.base import BaseMidiTokenizer, TokenizerConfig


class EventTokenizer(BaseMidiTokenizer):
    name = "event"

    def _build_vocab(self) -> None:
        for t in ("PAD", "BOS", "EOS", "UNK"):
            self._add(t)
        for p in range(self.cfg.min_pitch, self.cfg.max_pitch + 1):
            self._add(f"NOTE_ON_{p}")
            self._add(f"NOTE_OFF_{p}")
        for v in range(self.cfg.velocity_bins):
            self._add(f"VELOCITY_{v}")
        for s in range(1, MAX_TIME_SHIFT_STEPS + 1):
            self._add(f"TIME_SHIFT_{s}")
        if self.cfg.use_program:
            for program in range(128):
                self._add(f"PROGRAM_{program}")
        # Appended so older checkpoints keep the same note/time ids.
        for tag in EMOTION_TAGS:
            self._add(emotion_token(tag))
        for tag in INSTRUMENT_TAGS:
            self._add(instrument_token(tag))

    def _add_time_shift(self, tokens: list[str], steps: int) -> None:
        while steps > 0:
            shift = min(steps, MAX_TIME_SHIFT_STEPS)
            tokens.append(f"TIME_SHIFT_{shift}")
            steps -= shift

    def encode_pretty_midi(self, midi: pretty_midi.PrettyMIDI) -> list[int]:
        events: list[tuple[float, str]] = []

        for inst in midi.instruments:
            if inst.is_drum:
                continue
            if self.cfg.use_program:
                events.append((0.0, f"PROGRAM_{inst.program}"))
            for note in inst.notes:
                if not (self.cfg.min_pitch <= note.pitch <= self.cfg.max_pitch):
                    continue
                vb = self.velocity_to_bin(note.velocity)
                events.append((note.start, f"VELOCITY_{vb}"))
                events.append((note.start, f"NOTE_ON_{note.pitch}"))
                events.append((note.end, f"NOTE_OFF_{note.pitch}"))

        def sort_key(e: tuple[float, str]) -> tuple[float, int]:
            time, token = e
            priority = 0 if token.startswith("NOTE_OFF") else 1
            return (time, priority)

        events.sort(key=sort_key)

        tokens = ["BOS"]
        last_time = 0.0
        for time, token in events:
            delta = seconds_to_timesteps(time - last_time)
            if delta > 0:
                self._add_time_shift(tokens, delta)
                last_time = time
            tokens.append(token)
        tokens.append("EOS")
        return self.ids_from_tokens(tokens)

    def tokens_to_midi(
        self, ids: list[int], output_path: str | Path, *, tempo: float | None = None
    ) -> Path:
        tokens = self.decode_tokens(ids)
        current_time = 0.0
        pending_velocity = 64
        active: dict[int, tuple[float, int]] = {}

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        for tok in tokens:
            if tok in ("BOS", "PAD", "UNK") or tok.startswith(("EMOTION_", "INST_")):
                continue
            if tok == "EOS":
                break
            if tok.startswith("TIME_SHIFT_"):
                steps = int(tok.rsplit("_", 1)[-1])
                current_time += timesteps_to_seconds(steps)
            elif tok.startswith("VELOCITY_"):
                pending_velocity = self.bin_to_velocity(int(tok.rsplit("_", 1)[-1]))
            elif tok.startswith("NOTE_ON_"):
                pitch = int(tok.rsplit("_", 1)[-1])
                active[pitch] = (current_time, pending_velocity)
            elif tok.startswith("NOTE_OFF_"):
                pitch = int(tok.rsplit("_", 1)[-1])
                if pitch in active:
                    start, vel = active.pop(pitch)
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=vel,
                            pitch=pitch,
                            start=start,
                            end=max(start + 0.05, current_time),
                        )
                    )

        for pitch, (start, vel) in active.items():
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=pitch,
                    start=start,
                    end=start + 0.5,
                )
            )

        midi.instruments.append(instrument)
        out = Path(output_path)
        midi.write(str(out))
        return out
