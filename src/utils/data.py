from utils.midi_fmt import MidiDataset, MidiTokenizer
from torch.utils.data import random_split, DataLoader

tokenizer = MidiTokenizer()

dataset = MidiDataset("data/maestro-v3.0.0/2004", tokenizer, seq_len=4096)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)