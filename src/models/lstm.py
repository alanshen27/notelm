from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import datetime
from tqdm import tqdm


class MidiLSTM(nn.Module):
    """Core LSTM for training and inference."""

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 512, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, hidden=None):
        embedding = self.embedding(x)
        out, hidden = self.lstm(embedding, hidden)
        logits = self.fc(out)
        return logits, hidden


class LSTM(MidiLSTM):
    def __init__(
        self,
        train,
        val,
        vocab_size,
        device,
        pad_id,
        batch_size=2,
        accum_steps=1,
        num_workers=0,
    ):
        super().__init__(vocab_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = device
        self.accum_steps = accum_steps
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

        pin_memory = device == "cuda"
        loader_kw = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kw["persistent_workers"] = True
            loader_kw["prefetch_factor"] = 2

        self.train_data = DataLoader(
            train, shuffle=True, drop_last=True, **loader_kw
        )
        self.val_data = DataLoader(val, shuffle=False, **loader_kw)

    def forward(self, x, hidden=None):
        logits, _ = super().forward(x, hidden)
        return logits

    def train_unit(self):
        self.train()
        running_loss = 0.0
        self.optimizer.zero_grad()

        batch_bar = tqdm(
            self.train_data,
            desc="  train",
            leave=False,
            unit="batch",
        )
        for step, (inputs, labels) in enumerate(batch_bar):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits, _ = super().forward(inputs)

            loss = self.loss_fn(
                logits.reshape(-1, self.vocab_size), labels.reshape(-1)
            )
            scaled_loss = loss / self.accum_steps
            scaled_loss.backward()

            running_loss += loss.item()

            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(
                self.train_data
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_data)

    def validate(self):
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                self.val_data,
                desc="  val  ",
                leave=False,
                unit="batch",
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits, _ = super().forward(inputs)

                loss = self.loss_fn(
                    logits.reshape(-1, self.vocab_size),
                    labels.reshape(-1),
                )

                total_loss += loss.item()

        return total_loss / len(self.val_data)

    def _save_checkpoint(self, epoch):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(f"checkpoints/epoch-{epoch + 1}/{ts}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def fit(self, epochs):
        ckpt_root = Path("checkpoints").resolve()
        batches_per_epoch = len(self.train_data)
        eff_batch = self.train_data.batch_size * self.accum_steps
        print(
            f"Begin training: {epochs} epochs, "
            f"{batches_per_epoch:,} train batches/epoch "
            f"(micro-batch {self.train_data.batch_size}, "
            f"effective {eff_batch}), "
            f"checkpoints -> {ckpt_root}/"
        )
        epoch_bar = tqdm(range(epochs), desc="epochs", unit="epoch")
        for epoch in epoch_bar:
            train_loss = self.train_unit()
            val_loss = self.validate()

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
            )

            ckpt = self._save_checkpoint(epoch)
            tqdm.write(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"saved {ckpt}"
            )
