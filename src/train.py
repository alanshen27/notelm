from utils.data import tokenizer, train_dataset, val_dataset
from models.lstm import LSTM
import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = LSTM(
    train_dataset,
    val_dataset,
    tokenizer.vocab_size,
    device,
    2
).to(device)

print("Using device:", device)

model.fit(epochs=10) 

torch.save(
    model.state_dict(),
    "weights.pt"
)