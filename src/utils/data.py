import random
from pathlib import Path

from utils.midi_fmt import MidiDataset, MidiTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/maestro-v3.0.0/2004"
SEQ_LEN = 4096
STRIDE = SEQ_LEN // 2
TRAIN_SPLIT = 0.9
SPLIT_SEED = 42

tokenizer = MidiTokenizer()


def load_datasets():
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
        f"seq_len={SEQ_LEN}, stride={STRIDE}"
    )

    train_dataset = MidiDataset(
        tokenizer,
        seq_len=SEQ_LEN,
        files=train_files,
        stride=STRIDE,
        desc="Loading train MIDI",
    )
    val_dataset = MidiDataset(
        tokenizer,
        seq_len=SEQ_LEN,
        files=val_files,
        stride=STRIDE,
        desc="Loading val MIDI",
    )

    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f"No training windows (need token stream >= {SEQ_LEN}). "
            f"Check MIDI content in {DATA_DIR}."
        )

    print(
        f"Dataset ready: {len(train_dataset):,} train windows, "
        f"{len(val_dataset):,} val windows"
    )
    return train_dataset, val_dataset
