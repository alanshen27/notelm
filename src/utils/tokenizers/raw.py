"""Raw MIDI-style events: explicit deltas + message type + pitch + velocity."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from utils.midi_timing import MAX_TIME_SHIFT_STEPS, seconds_to_timesteps, timesteps_to_seconds
from utils.tokenizers.base import BaseMidiTokenizer


class RawMidiTokenizer(BaseMidiTokenizer):
    """
    Chronological raw events (no separate velocity token before note-on).
    Format: DELTA_T, MSG_NOTE_ON, PITCH_n, VEL_b  |  DELTA_T, MSG_NOTE_OFF, PITCH_n
    """

    name = "raw"

    def _build_vocab(self) -> None:
        for t in ("PAD", "BOS", "EOS", "UNK"):
            self._add(t)
        self._add("MSG_NOTE_ON")
        self._add("MSG_NOTE_OFF")
        for s in range(0, MAX_TIME_SHIFT_STEPS + 1):
            self._add(f"DELTA_{s}")
        for p in range(self.cfg.min_pitch, self.cfg.max_pitch + 1):
            self._add(f"PITCH_{p}")
        for v in range(self.cfg.velocity_bins):
            self._add(f"VEL_{v}")

    def encode_midi(self, midi_path: str | Path) -> list[int]:
        notes = self._collect_notes(midi_path)
        events: list[tuple[int, list[str]]] = []
        for note in notes:
            t_on = seconds_to_timesteps(note.start)
            t_off = seconds_to_timesteps(note.end)
            vb = self.velocity_to_bin(note.velocity)
            events.append(
                (
                    t_on,
                    [
                        "MSG_NOTE_ON",
                        f"PITCH_{note.pitch}",
                        f"VEL_{vb}",
                    ],
                )
            )
            events.append((t_off, ["MSG_NOTE_OFF", f"PITCH_{note.pitch}"]))
        events.sort(key=lambda e: (e[0], 0 if e[1][0] == "MSG_NOTE_OFF" else 1))

        tokens = ["BOS"]
        last_t = 0
        for t, msg_tokens in events:
            delta = max(0, t - last_t)
            delta = min(delta, MAX_TIME_SHIFT_STEPS)
            tokens.append(f"DELTA_{delta}")
            tokens.extend(msg_tokens)
            last_t = t
        tokens.append("EOS")
        return self.ids_from_tokens(tokens)

    def tokens_to_midi(self, ids: list[int], output_path: str | Path) -> Path:
        tokens = self.decode_tokens(ids)
        current_t = 0
        pending: dict[int, tuple[float, int]] = {}
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        i = 0

        while i < len(tokens):
            tok = tokens[i]
            if tok in ("BOS", "PAD", "UNK"):
                i += 1
                continue
            if tok == "EOS":
                break
            if tok.startswith("DELTA_"):
                current_t += int(tok.split("_", 1)[1])
                i += 1
                continue
            if tok == "MSG_NOTE_ON" and i + 2 < len(tokens):
                pitch = int(tokens[i + 1].split("_", 1)[1])
                vel = self.bin_to_velocity(int(tokens[i + 2].split("_", 1)[1]))
                pending[pitch] = (timesteps_to_seconds(current_t), vel)
                i += 3
                continue
            if tok == "MSG_NOTE_OFF" and i + 1 < len(tokens):
                pitch = int(tokens[i + 1].split("_", 1)[1])
                if pitch in pending:
                    start, vel = pending.pop(pitch)
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=vel,
                            pitch=pitch,
                            start=start,
                            end=max(start + 0.05, timesteps_to_seconds(current_t)),
                        )
                    )
                i += 2
                continue
            i += 1

        for pitch, (start, vel) in pending.items():
            inst.notes.append(
                pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=start + 0.5)
            )

        midi.instruments.append(inst)
        out = Path(output_path)
        midi.write(str(out))
        return out
