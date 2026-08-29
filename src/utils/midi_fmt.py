"""MIDI dataset + re-exports (tokenizers live in utils.tokenizers)."""

from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.emotion import emotion_token
from utils.instrument import instrument_from_pretty, instrument_token
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

MIN_FILE_TOKENS = 32


def _file_to_seq(file_path, tokenizer):
    try:
        midi = pretty_midi.PrettyMIDI(str(file_path))
        seq = tokenizer.encode_pretty_midi(midi)
        return seq, instrument_from_pretty(midi)
    except Exception:
        return [], "none"


def _encode_files(files: list[Path], tokenizer, desc: str, num_workers: int):
    """Spawn processes so pretty_midi can use all CPUs (threads hit the GIL)."""
    n = len(files)
    workers = max(1, min(num_workers, n))
    paths = [str(p) for p in files]
    if workers == 1:
        return [_file_to_seq(p, tokenizer) for p in paths]

    from utils.encode_worker import encode_path, init_encode_worker

    chunksize = max(1, min(32, n // (workers * 4) or 1))
    print(f"  encode workers={workers} (spawn) chunksize={chunksize}", flush=True)
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_encode_worker,
        initargs=(tokenizer.name,),
    ) as pool:
        return list(
            tqdm(
                pool.map(encode_path, paths, chunksize=chunksize),
                total=n,
                desc=desc,
                unit="file",
            )
        )


class MidiDataset(Dataset):
    """Windows stay inside one MIDI file. Optional emotion token prefixes each window."""

    def __init__(
        self,
        tokenizer,
        seq_len=100,
        *,
        midi_folder=None,
        files=None,
        emotions=None,
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
        self.pad_id = tokenizer.token_to_id["PAD"]
        self.seqs: list[np.ndarray] = []
        self._index: list[tuple[int, int, int, int]] = []  # file_i, start, emotion_id, inst_id

        if not files:
            return

        if num_workers is None:
            env_w = os.environ.get("NOTELM_ENCODE_WORKERS", "").strip()
            if env_w:
                num_workers = max(1, int(env_w))
            else:
                num_workers = min(48, os.cpu_count() or 4)

        emo_tags = list(emotions) if emotions is not None else ["none"] * len(files)
        if len(emo_tags) != len(files):
            raise ValueError("emotions must match files")

        encoded = _encode_files(files, tokenizer, desc, num_workers)
        seqs: list[list[int]] = []
        inst_tags: list[str] = []
        skipped = 0
        for seq, inst_tag in encoded:
            if not seq:
                skipped += 1
            seqs.append(seq)
            inst_tags.append(inst_tag)
        if skipped:
            print(f"  skipped {skipped} unreadable/empty MIDI files")

        for i, seq in enumerate(seqs):
            if len(seq) < MIN_FILE_TOKENS:
                continue
            arr = np.asarray(seq, dtype=np.int64)
            emo_id = tokenizer.token_to_id.get(emotion_token(emo_tags[i]))
            if emo_id is None:
                emo_id = tokenizer.token_to_id.get(emotion_token("none"), 0)
            inst_id = tokenizer.token_to_id.get(instrument_token(inst_tags[i]))
            if inst_id is None:
                inst_id = tokenizer.token_to_id.get(instrument_token("none"), emo_id)
            file_i = len(self.seqs)
            self.seqs.append(arr)
            n = len(arr)
            need = seq_len + 1
            if n < need:
                self._index.append((file_i, 0, emo_id, inst_id))
            else:
                last = n - need
                for start in range(0, last + 1, self.stride):
                    self._index.append((file_i, start, emo_id, inst_id))

    def save_cache(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "seq_len": self.seq_len,
                "stride": self.stride,
                "pad_id": self.pad_id,
                "seqs": self.seqs,
                "index": self._index,
            },
            path,
        )
        gb = path.stat().st_size / 1e9
        print(f"  wrote token cache {path} ({gb:.2f} GB)", flush=True)

    @classmethod
    def load_cache(cls, path: Path, tokenizer) -> "MidiDataset":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        ds = cls.__new__(cls)
        ds.seq_len = int(blob["seq_len"])
        ds.stride = int(blob["stride"])
        ds.pad_id = int(blob["pad_id"])
        ds.seqs = blob["seqs"]
        ds._index = blob["index"]
        print(
            f"  loaded token cache {path} "
            f"({len(ds.seqs):,} files, {len(ds._index):,} windows)",
            flush=True,
        )
        return ds

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_i, start, emo_id, inst_id = self._index[idx]
        seq = self.seqs[file_i]
        need = self.seq_len + 1
        if start + need <= len(seq):
            window = seq[start : start + need]
        else:
            window = np.empty(need, dtype=np.int64)
            n = len(seq)
            window[:n] = seq
            window[n:] = self.pad_id
        music = window[: self.seq_len]
        x = np.empty(self.seq_len, dtype=np.int64)
        x[0] = emo_id
        x[1] = inst_id
        x[2:] = music[:-2]
        y = music
        return torch.from_numpy(x), torch.from_numpy(y)
