from utils.data import tokenizer, train_dataset, val_dataset
from utils.notify import notify_training_complete
from models.lstm import LSTM
import torch
import time

EPOCHS = 320
WEIGHTS_PATH = "weights.pt"
BATCH_SIZE = 2
PAD_ID = tokenizer.token_to_id["PAD"]

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

start = time.time()
try:
    model = LSTM(
        train_dataset,
        val_dataset,
        tokenizer.vocab_size,
        device,
        PAD_ID,
        BATCH_SIZE,
    ).to(device)

    model.fit(epochs=EPOCHS)

    torch.save(model.state_dict(), WEIGHTS_PATH)

    notify_training_complete(
        success=True,
        epochs=EPOCHS,
        device=device,
        elapsed_s=time.time() - start,
        weights_path=WEIGHTS_PATH,
    )
except Exception as exc:
    notify_training_complete(
        success=False,
        epochs=EPOCHS,
        device=device,
        elapsed_s=time.time() - start,
        error=str(exc),
    )
    raise
