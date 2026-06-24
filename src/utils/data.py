import os
import random
from pathlib import Path

from utils.midi_fmt import MidiDataset
from utils.midi_timing import TIMESTEP_MS
from utils.tokenizers import TOKENIZER_NAMES, get_tokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/maestro-v3.0.0/2004"
TRAIN_SPLIT = 0.9
SPLIT_SEED = 42

# Per-tokenizer window sizes (piano_roll is much denser per second of audio).
TOKENIZER_SEQ_LEN = {
    "event": 4096,
    "raw": 4096,
    "remi": 4096,
    "piano_roll": 1024,
}

# The three alternate representations (+ event as default).
TRAIN_TOKENIZERS = ("event", "raw", "remi", "piano_roll")
ALT_TOKENIZERS = ("raw", "remi", "piano_roll")


def seq_len_for(tokenizer_name: str) -> int:
    return TOKENIZER_SEQ_LEN.get(tokenizer_name, 4096)


def stride_for(tokenizer_name: str) -> int:
    return max(1, seq_len_for(tokenizer_name) // 2)


def load_datasets(tokenizer_name: str | None = None):
    name = (tokenizer_name or os.environ.get("NOTELM_TOKENIZER", "event")).strip().lower()
    tokenizer = get_tokenizer(name)
    seq_len = seq_len_for(name)
    stride = stride_for(name)

    files = sorted(DATA_DIR.glob("*.midi"))
    if not files:
        raise FileNotFoundError(
            f"No MIDI files found in {DATA_DIR}. "
            "Check that MAESTRO MIDI files exist at the project data path."
        )

    rng = random.Random(SPLIT_SEED)
    shuffled = list(files)
    rng.shuffle(shuffled)

    n_train = int(TRAIN_SPLIT * len(shuffled))
    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:]

    print(
        f"Split {len(files):,} files by path "
        f"({len(train_files):,} train / {len(val_files):,} val), "
        f"tokenizer={tokenizer.name}, timestep_ms={TIMESTEP_MS}, "
        f"seq_len={seq_len}, stride={stride}"
    )

    train_dataset = MidiDataset(
        tokenizer,
        seq_len=seq_len,
        files=train_files,
        stride=stride,
        desc=f"Loading train MIDI [{name}]",
    )
    val_dataset = MidiDataset(
        tokenizer,
        seq_len=seq_len,
        files=val_files,
        stride=stride,
        desc=f"Loading val MIDI [{name}]",
    )

    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f"No training windows (need token stream >= {seq_len}). "
            f"Check MIDI content in {DATA_DIR}."
        )

    print(
        f"Dataset ready: {len(train_dataset):,} train windows, "
        f"{len(val_dataset):,} val windows"
    )
    return train_dataset, val_dataset, tokenizer
