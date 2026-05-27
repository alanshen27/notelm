from dataclasses import dataclass
from pathlib import Path
import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pretty_midi
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


@dataclass
class MidiTokenizerConfig:
    time_step_ms: int = 20          # quantization step
    max_time_shift_steps: int = 100 # max 2 seconds if 20ms
    velocity_bins: int = 16
    use_program: bool = False       # instrument tokens
    min_pitch: int = 21             # piano low A
    max_pitch: int = 108            # piano high C


class MidiTokenizer:
    def __init__(self, config=MidiTokenizerConfig()):
        self.cfg = config
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()

    def _add(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def _build_vocab(self):
        special = ["PAD", "BOS", "EOS", "UNK"]
        for t in special:
            self._add(t)

        for p in range(self.cfg.min_pitch, self.cfg.max_pitch + 1):
            self._add(f"NOTE_ON_{p}")
            self._add(f"NOTE_OFF_{p}")

        for v in range(self.cfg.velocity_bins):
            self._add(f"VELOCITY_{v}")

        for s in range(1, self.cfg.max_time_shift_steps + 1):
            self._add(f"TIME_SHIFT_{s}")

        if self.cfg.use_program:
            for program in range(128):
                self._add(f"PROGRAM_{program}")

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def velocity_to_bin(self, velocity):
        # MIDI velocity: 0–127
        bin_size = 128 / self.cfg.velocity_bins
        return min(int(velocity / bin_size), self.cfg.velocity_bins - 1)

    def seconds_to_steps(self, seconds):
        ms = seconds * 1000
        return round(ms / self.cfg.time_step_ms)

    def add_time_shift(self, tokens, steps):
        while steps > 0:
            shift = min(steps, self.cfg.max_time_shift_steps)
            tokens.append(f"TIME_SHIFT_{shift}")
            steps -= shift

    def encode_midi(self, midi_path):
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        events = []

        for inst in midi.instruments:
            if inst.is_drum:
                continue

            if self.cfg.use_program:
                events.append((0, f"PROGRAM_{inst.program}"))

            for note in inst.notes:
                if not (self.cfg.min_pitch <= note.pitch <= self.cfg.max_pitch):
                    continue

                velocity_bin = self.velocity_to_bin(note.velocity)

                events.append((note.start, f"VELOCITY_{velocity_bin}"))
                events.append((note.start, f"NOTE_ON_{note.pitch}"))
                events.append((note.end, f"NOTE_OFF_{note.pitch}"))

        # Sort by time; NOTE_OFF before NOTE_ON at same time avoids weird overlaps
        def event_sort_key(e):
            time, token = e
            priority = 0 if token.startswith("NOTE_OFF") else 1
            return (time, priority)

        events.sort(key=event_sort_key)

        tokens = ["BOS"]
        last_time = 0.0

        for time, token in events:
            delta_steps = self.seconds_to_steps(time - last_time)

            if delta_steps > 0:
                self.add_time_shift(tokens, delta_steps)
                last_time = time

            tokens.append(token)

        tokens.append("EOS")

        return [self.token_to_id.get(t, self.token_to_id["UNK"]) for t in tokens]

    def decode_tokens(self, ids):
        return [self.id_to_token.get(i, "UNK") for i in ids]

    def tokens_to_midi(self, ids, output_path):
        tokens = self.decode_tokens(ids)
        step_sec = self.cfg.time_step_ms / 1000.0

        current_time = 0.0
        pending_velocity = 64
        active = {}

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        for tok in tokens:
            if tok in ("BOS", "PAD", "UNK"):
                continue
            if tok == "EOS":
                break
            if tok.startswith("TIME_SHIFT_"):
                steps = int(tok.rsplit("_", 1)[-1])
                current_time += steps * step_sec
            elif tok.startswith("VELOCITY_"):
                bin_idx = int(tok.rsplit("_", 1)[-1])
                pending_velocity = min(
                    127,
                    int((bin_idx + 0.5) * 128 / self.cfg.velocity_bins),
                )
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
        midi.write(str(output_path))
        return output_path


def _file_to_seq(file_path, tokenizer):
    return tokenizer.encode_midi(file_path)

class MidiDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        seq_len=100,
        *,
        midi_folder=None,
        files=None,
        stride=None,
        num_workers=None,
        desc="Loading MIDI",
    ):
        if files is None:
            if midi_folder is None:
                raise ValueError("Provide midi_folder or files")
            files = sorted(Path(midi_folder).glob("*.midi"))
        else:
            files = [Path(p) for p in files]

        self.seq_len = seq_len
        self.stride = stride if stride is not None else max(1, seq_len // 2)

        if not files:
            self.tokens = torch.empty(0, dtype=torch.int64)
            return

        if num_workers is None:
            num_workers = min(32, os.cpu_count() or 4)

        worker = partial(_file_to_seq, tokenizer=tokenizer)

        seqs = []

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for seq in tqdm(
                pool.map(worker, files),
                total=len(files),
                desc=desc,
                unit="file",
            ):
                seqs.append(seq)

        all_tokens = []
        for seq in seqs:
            all_tokens.extend(seq)

        all_tokens = np.asarray(all_tokens, dtype=np.int64)
        self.tokens = torch.from_numpy(all_tokens)

    def __len__(self):
        n = len(self.tokens) - self.seq_len
        if n < 0:
            return 0
        return n // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


# dataset_train = MIDIDataset(
#     "train_maestro-v3.0.0",
#     seq_len=100
# )

# dataset_val = MIDIDataset(
#     "val_maestro-v3.0.0",
#     seq_len=100
# )