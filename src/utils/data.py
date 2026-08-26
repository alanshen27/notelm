import os
import random
from pathlib import Path

from utils.midi_fmt import MidiDataset
from utils.midi_timing import TIMESTEP_MS
from utils.tokenizers import TOKENIZER_NAMES, get_tokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"

# Legacy alias (older scripts import DATA_DIR directly).
DATA_DIR = DATA_ROOT / "maestro-v3.0.0/2004"

TRAIN_SPLIT = 0.9
SPLIT_SEED = 42

DATASET_NAMES = ("pop909", "maestro", "maestro_full")
DEFAULT_DATASET = "pop909"

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


def normalize_dataset(name: str | None = None) -> str:
    key = (name or os.environ.get("NOTELM_DATASET", DEFAULT_DATASET)).strip().lower()
    if key not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {key!r}. Choose from: {', '.join(DATASET_NAMES)}")
    return key


def dataset_dir(name: str | None = None) -> Path:
    key = normalize_dataset(name)
    if key == "maestro":
        return DATA_ROOT / "maestro-v3.0.0/2004"
    if key == "maestro_full":
        return DATA_ROOT / "maestro-v3.0.0"
    return DATA_ROOT / "POP909"


def dataset_files(name: str | None = None) -> list[Path]:
    """All training MIDI files for a dataset.

    POP909 layout is POP909/{id}/{id}.mid plus a versions/ folder of alternate
    takes of the same song; only the main arrangement (stem == folder name) is
    used so near-duplicates never straddle the train/val split.
    """
    key = normalize_dataset(name)
    root = dataset_dir(key)
    if key == "maestro":
        return sorted(root.glob("*.midi"))
    if key == "maestro_full":
        return sorted(root.glob("*/*.midi"))
    return sorted(p for p in root.glob("*/*.mid") if p.stem == p.parent.name)


def seq_len_for(tokenizer_name: str) -> int:
    return TOKENIZER_SEQ_LEN.get(tokenizer_name, 4096)


def stride_for(seq_len: int) -> int:
    return max(1, seq_len // 2)


def load_datasets(
    tokenizer_name: str | None = None,
    *,
    dataset: str | None = None,
    seq_len: int | None = None,
    limit_files: int | None = None,
):
    name = (tokenizer_name or os.environ.get("NOTELM_TOKENIZER", "event")).strip().lower()
    ds_name = normalize_dataset(dataset)
    tokenizer = get_tokenizer(name)
    seq_len = seq_len or seq_len_for(name)
    stride = stride_for(seq_len)

    files = dataset_files(ds_name)
    if not files:
        fetch_flag = "--fetch-pop909" if ds_name == "pop909" else "--fetch-maestro"
        raise FileNotFoundError(
            f"No MIDI files found for dataset {ds_name!r} in {dataset_dir(ds_name)}. "
            f"Fetch it first: ./scripts/setup.sh {fetch_flag}"
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

    train_dataset = MidiDataset(
        tokenizer,
        seq_len=seq_len,
        files=train_files,
        stride=stride,
        desc=f"Loading train MIDI [{ds_name}/{name}]",
    )
    val_dataset = MidiDataset(
        tokenizer,
        seq_len=seq_len,
        files=val_files,
        stride=stride,
        desc=f"Loading val MIDI [{ds_name}/{name}]",
    )

    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f"No training windows (need token stream >= {seq_len}). "
            f"Check MIDI content in {dataset_dir(ds_name)}."
        )

    print(
        f"Dataset ready: {len(train_dataset):,} train windows, "
        f"{len(val_dataset):,} val windows"
    )
    return train_dataset, val_dataset, tokenizer
