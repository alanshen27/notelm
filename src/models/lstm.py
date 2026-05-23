from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import datetime
from tqdm import tqdm

class LSTM (nn.Module):
    def __init__(self, train, val, vocab_size, device, pad_id, batch_size=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 512, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.train_data = DataLoader(train, batch_size, shuffle=True)
        self.val_data = DataLoader(val, batch_size, shuffle=False)
        self.vocab_size = vocab_size
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, x, hidden = None):
        embedding = self.embedding(x)

        out, hidden = self.lstm(embedding, hidden)

        logits = self.fc(out)

        return logits

    def train_unit(self):
        self.train()
        running_loss = 0.

        batch_bar = tqdm(
            self.train_data,
            desc="  train",
            leave=False,
            unit="batch",
        )
        for inputs, labels in batch_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()

            logits = self(inputs)

            loss = self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
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
                logits = self(inputs.to(self.device))

                loss = self.loss_fn(
                    logits.reshape(-1, self.vocab_size),
                    labels.to(self.device).reshape(-1)
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
        print("Begin training")
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

