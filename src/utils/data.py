"""Dataset registry, file lists, and windowed MIDI loading."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.emotion import emotion_from_path
from utils.midi_fmt import MidiDataset
from utils.midi_timing import TIMESTEP_MS
from utils.tokenizers import get_tokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DATA_DIR = DATA_ROOT / "maestro-v3.0.0/2004"

TRAIN_SPLIT = 0.9
SPLIT_SEED = 42
TOKEN_CACHE_VERSION = "v1"

# Atomic corpora on disk. Aliases union whatever is present.
ATOMIC_DATASETS = (
    "pop909",
    "pop1k7",
    "emopia",
    "adl",
    "gigamidi",
    "giantmidi",
    "atepp",
    "pdmx",
    "asap",
    "maestro",
    "maestro_full",
)
ALIASES: dict[str, tuple[str, ...]] = {
    "pop": ("pop909", "pop1k7", "emopia"),
    "pretrain": ("maestro_full", "adl", "asap", "gigamidi", "giantmidi", "atepp"),
    # All-instrument pretrain: PDMX (multi) + piano corpora. INST_* prefixes on.
    "instruments": (
        "pdmx",
        "giantmidi",
        "atepp",
        "maestro_full",
        "adl",
        "asap",
        "pop909",
        "pop1k7",
        "emopia",
        "gigamidi",
    ),
    # Piano finetune (clavier). Still conditions on INST_piano.
    "piano": (
        "pop909",
        "pop1k7",
        "emopia",
        "giantmidi",
        "atepp",
        "adl",
        "asap",
        "maestro_full",
    ),
    "canon": (
        "pdmx",
        "giantmidi",
        "atepp",
        "maestro_full",
        "adl",
        "asap",
        "gigamidi",
        "pop909",
        "pop1k7",
        "emopia",
    ),
    "all": (
        "pop909",
        "pop1k7",
        "emopia",
        "adl",
        "gigamidi",
        "giantmidi",
        "atepp",
        "pdmx",
        "asap",
        "maestro_full",
    ),
}
DATASET_NAMES = ATOMIC_DATASETS + tuple(ALIASES)
DEFAULT_DATASET = "pop909"

TOKENIZER_SEQ_LEN = {
    "event": 4096,
    "remi": 2048,
}
TRAIN_TOKENIZERS = ("event", "remi")


def normalize_dataset(name: str | None = None) -> str:
    key = (name or os.environ.get("NOTELM_DATASET", DEFAULT_DATASET)).strip().lower()
    if key not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {key!r}. Choose from: {', '.join(DATASET_NAMES)}")
    return key


def dataset_dir(name: str | None = None) -> Path:
    key = normalize_dataset(name)
    if key in ALIASES:
        return DATA_ROOT
    return {
        "pop909": DATA_ROOT / "POP909",
        "pop1k7": DATA_ROOT / "Pop1K7",
        "emopia": DATA_ROOT / "EMOPIA",
        "adl": DATA_ROOT / "adl-piano-midi",
        "gigamidi": DATA_ROOT / "GigaMIDI",
        "giantmidi": DATA_ROOT / "GiantMIDI-Piano",
        "atepp": DATA_ROOT / "ATEPP",
        "pdmx": DATA_ROOT / "PDMX",
        "asap": DATA_ROOT / "asap-dataset",
        "maestro": DATA_ROOT / "maestro-v3.0.0/2004",
        "maestro_full": DATA_ROOT / "maestro-v3.0.0",
    }[key]


def _is_training_midi(path: Path) -> bool:
    """Drop macOS resource forks and unzip junk that glob as *.mid."""
    if not path.is_file():
        return False
    if path.name.startswith("._"):
        return False
    return "__MACOSX" not in path.parts


def _midi_under(root: Path, pattern: str = "**/*") -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for ext in (".mid", ".midi"):
        files.extend(p for p in root.glob(f"{pattern}{ext}") if _is_training_midi(p))
    return files


def dataset_files(name: str | None = None) -> list[Path]:
    """Training MIDI paths for a corpus (or alias). Missing folders yield []."""
    key = normalize_dataset(name)
    if key in ALIASES:
        seen: set[str] = set()
        out: list[Path] = []
        for part in ALIASES[key]:
            for p in dataset_files(part):
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
        return out

    root = dataset_dir(key)
    if key == "pop909":
        return sorted(
            p
            for p in root.glob("*/*.mid")
            if _is_training_midi(p) and p.stem == p.parent.name
        )
    if key == "maestro":
        return sorted(p for p in root.glob("*.midi") if _is_training_midi(p))
    if key == "maestro_full":
        return sorted(
            p
            for p in root.glob("*/*.midi")
            if _is_training_midi(p) and p.parent.name != "test"
        )
    if key == "emopia":
        return sorted(_midi_under(root / "midis") or _midi_under(root))
    if key == "pop1k7":
        preferred = _midi_under(root / "midi_analyzed") or _midi_under(root / "midi")
        return sorted(preferred or _midi_under(root))
    if key == "pdmx":
        files = sorted(_midi_under(root))
        cap = int(os.environ.get("NOTELM_PDMX_MAX", "40000"))
        if cap > 0 and len(files) > cap:
            rng = random.Random(SPLIT_SEED)
            files = sorted(rng.sample(files, cap))
        return files
    return sorted(_midi_under(root))


def seq_len_for(tokenizer_name: str) -> int:
    return TOKENIZER_SEQ_LEN.get(tokenizer_name, 4096)


def stride_for(seq_len: int) -> int:
    return max(1, seq_len // 2)


def token_cache_dir(tokenizer_name: str, ds_name: str) -> Path:
    return DATA_ROOT / "token_cache" / TOKEN_CACHE_VERSION / tokenizer_name / ds_name


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _dist_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_datasets(
    tokenizer_name: str | None = None,
    *,
    dataset: str | None = None,
    seq_len: int | None = None,
    limit_files: int | None = None,
):
    name = (tokenizer_name or os.environ.get("NOTELM_TOKENIZER", "remi")).strip().lower()
    ds_name = normalize_dataset(dataset)
    tokenizer = get_tokenizer(name)
    seq_len = seq_len or seq_len_for(name)
    stride = stride_for(seq_len)
    cache_dir = token_cache_dir(name, ds_name)
    train_pt = cache_dir / "train.pt"
    val_pt = cache_dir / "val.pt"
    rebuild = os.environ.get("NOTELM_REBUILD_CACHE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    def _from_cache():
        print(f"Token cache hit {cache_dir}", flush=True)
        train_ds = MidiDataset.load_cache(train_pt, tokenizer)
        val_ds = MidiDataset.load_cache(val_pt, tokenizer)
        print(
            f"Dataset ready: {len(train_ds):,} train windows, "
            f"{len(val_ds):,} val windows (cached)"
        )
        return train_ds, val_ds, tokenizer

    cache_ok = (not rebuild) and train_pt.is_file() and val_pt.is_file()
    if cache_ok:
        return _from_cache()

    if _dist_rank() != 0:
        _dist_barrier()
        if train_pt.is_file() and val_pt.is_file():
            return _from_cache()
        raise FileNotFoundError(f"Rank {_dist_rank()} waiting for token cache at {cache_dir}")

    files = dataset_files(ds_name)
    if not files:
        raise FileNotFoundError(
            f"No MIDI files found for dataset {ds_name!r}. "
            f"Fetch with: ./scripts/setup.sh --fetch-{ds_name}"
            if ds_name in ATOMIC_DATASETS
            else f"No MIDI files found for {ds_name!r}. Fetch corpora first "
            f"(./scripts/setup.sh --fetch-extra)."
        )

    rng = random.Random(SPLIT_SEED)
    shuffled = list(files)
    rng.shuffle(shuffled)
    if limit_files is not None:
        shuffled = shuffled[: max(2, limit_files)]

    n_train = max(1, int(TRAIN_SPLIT * len(shuffled)))
    if n_train == len(shuffled):
        n_train = len(shuffled) - 1
    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:]

    print(
        f"Dataset {ds_name}: split {len(shuffled):,} files by path "
        f"({len(train_files):,} train / {len(val_files):,} val), "
        f"tokenizer={tokenizer.name}, timestep_ms={TIMESTEP_MS}, "
        f"seq_len={seq_len}, stride={stride}"
    )

    train_emotions = [emotion_from_path(p) for p in train_files]
    val_emotions = [emotion_from_path(p) for p in val_files]
    labeled = sum(1 for e in train_emotions + val_emotions if e != "none")
    if labeled:
        print(f"  emotion labels on {labeled:,} / {len(shuffled):,} files (EMOPIA-style Q1–Q4)")

    train_dataset = MidiDataset(
        tokenizer,
        seq_len=seq_len,
        files=train_files,
        emotions=train_emotions,
        stride=stride,
        desc=f"Loading train MIDI [{ds_name}/{name}]",
    )
    val_dataset = MidiDataset(
        tokenizer,
        seq_len=seq_len,
        files=val_files,
        emotions=val_emotions,
        stride=stride,
        desc=f"Loading val MIDI [{ds_name}/{name}]",
    )

    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f"No training windows (need token stream >= {seq_len} or padded clips). "
            f"Check MIDI content for {ds_name}."
        )

    train_dataset.save_cache(train_pt)
    val_dataset.save_cache(val_pt)
    print(
        f"Dataset ready: {len(train_dataset):,} train windows, "
        f"{len(val_dataset):,} val windows"
    )
    _dist_barrier()
    return train_dataset, val_dataset, tokenizer


SPAN_K = (1, 2, 4, 8)
SPAN_P = (0.35, 0.25, 0.25, 0.15)


class SpanInfillDataset(Dataset):
    """Mask a REMI bar span; encoder sees left+SPAN+right, decoder the gap."""

    def __init__(
        self,
        source: MidiDataset,
        tokenizer,
        *,
        enc_len: int = 2048,
        dec_len: int = 768,
    ):
        self.source = source
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.pad_id = tokenizer.token_to_id["PAD"]
        self.bos_id = tokenizer.token_to_id["BOS"]
        self.eos_id = tokenizer.token_to_id["EOS"]
        span = tokenizer.token_to_id.get("SPAN")
        if span is None:
            raise ValueError("tokenizer must define SPAN for canon infill")
        self.span_id = span
        self.bar_id = tokenizer.token_to_id.get("Bar")
        if self.bar_id is None:
            raise ValueError("canon span infill requires a remi tokenizer with Bar")

    def __len__(self):
        return len(self.source)

    def _window(self, idx: int) -> tuple[np.ndarray, int, int]:
        file_i, start, emo_id, inst_id = self.source._index[idx]
        seq = self.source.seqs[file_i]
        need = self.source.seq_len
        if start + need <= len(seq):
            window = seq[start : start + need]
        else:
            window = np.empty(need, dtype=np.int64)
            n = len(seq)
            window[:n] = seq
            window[n:] = self.pad_id
        return window, int(emo_id), int(inst_id)

    def _cut(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bars = np.flatnonzero(window == self.bar_id)
        n_bars = int(bars.size)
        if n_bars >= 2:
            k = int(np.random.choice(SPAN_K, p=SPAN_P))
            k = max(1, min(k, n_bars - 1))
            i = int(np.random.randint(0, n_bars - k + 1))
            a = int(bars[i])
            b = int(bars[i + k]) if (i + k) < n_bars else len(window)
            left, span, right = window[:a], window[a:b], window[b:]
        else:
            n = int((window != self.pad_id).sum())
            span_len = min(n, max(32, int(np.random.randint(48, 161))))
            a = int(np.random.randint(0, max(1, n - span_len + 1)))
            b = a + span_len
            left, span, right = window[:a], window[a:b], window[b:n]
        left = left[left != self.pad_id]
        right = right[right != self.pad_id]
        span = span[span != self.pad_id]
        if span.size == 0:
            span = window[: min(32, len(window))]
        return left, span, right

    def __getitem__(self, idx):
        window, emo_id, inst_id = self._window(idx)
        left, span, right = self._cut(window)
        budget = self.enc_len - 3
        if left.size == 0:
            right = right[:budget]
        elif right.size == 0:
            left = left[-budget:]
        else:
            half = budget // 2
            left = left[-half:]
            right = right[: budget - len(left)]
        enc = np.full(self.enc_len, self.pad_id, dtype=np.int64)
        pieces = np.concatenate(
            ([emo_id, inst_id], left, [self.span_id], right)
        ).astype(np.int64, copy=False)
        enc[: min(len(pieces), self.enc_len)] = pieces[: self.enc_len]

        target = np.empty(len(span) + 1, dtype=np.int64)
        target[:-1] = span
        target[-1] = self.eos_id
        if len(target) > self.dec_len:
            target = target[: self.dec_len]
            target[-1] = self.eos_id
        dec = np.full(self.dec_len, self.pad_id, dtype=np.int64)
        labels = np.full(self.dec_len, self.pad_id, dtype=np.int64)
        dec[0] = self.bos_id
        n = min(len(target), self.dec_len - 1)
        dec[1 : 1 + n] = target[:n]
        labels[:n] = target[:n]
        if n < self.dec_len:
            labels[n] = self.pad_id
        return (
            torch.from_numpy(enc),
            torch.from_numpy(dec),
            torch.from_numpy(labels),
        )


def load_span_datasets(
    tokenizer_name: str | None = None,
    *,
    dataset: str | None = None,
    seq_len: int | None = None,
    limit_files: int | None = None,
    enc_len: int = 2048,
    dec_len: int = 768,
):
    train_lm, val_lm, tokenizer = load_datasets(
        tokenizer_name,
        dataset=dataset,
        seq_len=seq_len,
        limit_files=limit_files,
    )
    train = SpanInfillDataset(train_lm, tokenizer, enc_len=enc_len, dec_len=dec_len)
    val = SpanInfillDataset(val_lm, tokenizer, enc_len=enc_len, dec_len=dec_len)
    print(
        f"Canon spans: {len(train):,} train / {len(val):,} val "
        f"(enc_len={enc_len}, dec_len={dec_len}, k∈{SPAN_K})"
    )
    return train, val, tokenizer
