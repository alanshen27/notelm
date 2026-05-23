from dataclasses import dataclass
from pathlib import Path
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
    
class MidiDataset(Dataset):

    def __init__(self, midi_folder, tokenizer, seq_len=100):

        self.samples = []

        files = sorted(Path(midi_folder).glob("*.midi"))

        for file in tqdm(files, desc="Loading MIDI", unit="file"):
            seq = tokenizer.encode_midi(file)

            # sliding window
            pad_id = tokenizer.token_to_id["PAD"]

            if len(seq) < seq_len + 1:
                seq = seq + [pad_id] * (seq_len + 1 - len(seq))

            for i in range(len(seq)-seq_len):
                x = seq[i:i+seq_len]
                y = seq[i+1:i+seq_len+1]

                self.samples.append(
                    (
                        torch.tensor(x),
                        torch.tensor(y)
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# dataset_train = MIDIDataset(
#     "train_maestro-v3.0.0",
#     seq_len=100
# )

# dataset_val = MIDIDataset(
#     "val_maestro-v3.0.0",
#     seq_len=100
# )