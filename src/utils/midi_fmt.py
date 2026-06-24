"""MIDI dataset + re-exports (tokenizers live in utils.tokenizers)."""

from pathlib import Path
import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.midi_timing import TIMESTEP_MS, TIMESTEP_SEC, MAX_TIME_SHIFT_STEPS
from utils.tokenizers import (
    BaseMidiTokenizer,
    EventTokenizer,
    MidiTokenizer,
    MidiTokenizerConfig,
    TokenizerConfig,
    get_tokenizer,
)

__all__ = [
    "TIMESTEP_MS",
    "TIMESTEP_SEC",
    "MAX_TIME_SHIFT_STEPS",
    "BaseMidiTokenizer",
    "MidiTokenizer",
    "MidiTokenizerConfig",
    "TokenizerConfig",
    "get_tokenizer",
    "MidiDataset",
]


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
