import os
import time

import torch

from models.lstm import LSTM
from utils.data import load_datasets, tokenizer
from utils.notify import notify_training_complete

EPOCHS = 320
WEIGHTS_PATH = "weights.pt"
PAD_ID = tokenizer.token_to_id["PAD"]


def _training_config(device: str) -> dict:
    """Tune batch size from available device memory."""
    cpus = os.cpu_count() or 1
    if device == "cuda":
        gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gb >= 35:
            return {"batch_size": 64, "accum_steps": 1, "num_workers": min(8, cpus)}
        if gb >= 20:
            return {"batch_size": 32, "accum_steps": 1, "num_workers": min(6, cpus)}
        if gb >= 12:
            return {"batch_size": 16, "accum_steps": 1, "num_workers": min(4, cpus)}
        return {"batch_size": 8, "accum_steps": 2, "num_workers": min(4, cpus)}
    if device == "mps":
        return {"batch_size": 4, "accum_steps": 4, "num_workers": 0}
    return {"batch_size": 2, "accum_steps": 8, "num_workers": min(2, cpus)}


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
train_cfg = _training_config(device)

if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    print(f"Using device: {device} ({props.name}, {vram_gb:.0f} GB)")
else:
    print("Using device:", device)

print(
    f"Training config: batch_size={train_cfg['batch_size']}, "
    f"accum_steps={train_cfg['accum_steps']} "
    f"(effective {train_cfg['batch_size'] * train_cfg['accum_steps']}), "
    f"num_workers={train_cfg['num_workers']}"
)

train_dataset, val_dataset = load_datasets()

start = time.time()
try:
    model = LSTM(
        train_dataset,
        val_dataset,
        tokenizer.vocab_size,
        device,
        PAD_ID,
        batch_size=train_cfg["batch_size"],
        accum_steps=train_cfg["accum_steps"],
        num_workers=train_cfg["num_workers"],
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
