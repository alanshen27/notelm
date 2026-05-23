from pathlib import Path

from torch.utils.data import random_split
import torch

from utils.midi_fmt import MidiDataset, MidiTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/maestro-v3.0.0/2004"
SEQ_LEN = 4096
TRAIN_SPLIT = 0.9
SPLIT_SEED = 42

tokenizer = MidiTokenizer()
dataset = MidiDataset(DATA_DIR, tokenizer, seq_len=SEQ_LEN)

if len(dataset) == 0:
    raise FileNotFoundError(
        f"No training samples found in {DATA_DIR}. "
        "Check that MAESTRO MIDI files exist at the project data path."
    )

train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SPLIT_SEED)
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)
