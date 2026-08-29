"""Decoder-only Transformer (GPT-style) for MIDI token modeling."""

from __future__ import annotations

import datetime
import re
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.music_loss import PC_AUX, class_weights, pitch_class_loss, pitch_lookups

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 8
FFN_DIM = 2048
DROPOUT = 0.1
MAX_LEN = 4096
ROPE_BASE = 10_000.0

LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE: rotate Q/K by token distance instead of adding absolute pos vectors."""

    def __init__(self, head_dim: int, base: float = ROPE_BASE):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def rotate(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin(q.size(2), q.device, q.dtype)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        causal: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope = RotaryEmbedding(d_model // n_heads) if use_rope else None

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
        k = k.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
        v = v.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope.rotate(q, k)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


CausalSelfAttention = SelfAttention


class Block(nn.Module):
    """Pre-norm transformer block: LN -> attention, LN -> MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        use_rope: bool = True,
        *,
        causal: bool = True,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(
            d_model, n_heads, dropout, causal=causal, use_rope=use_rope
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MidiTransformer(nn.Module):
    """Core decoder-only model for training and inference."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
        max_len: int = MAX_LEN,
        use_rope: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_rope = use_rope
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = None if use_rope else nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, ffn_dim, dropout, use_rope=use_rope)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len {self.max_len}")
        h = self.embedding(x)
        if self.pos_embedding is not None:
            pos = torch.arange(t, device=x.device)
            h = h + self.pos_embedding(pos)[None, :, :]
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        return self.fc(self.norm(h))

    @classmethod
    def from_state_dict(cls, state: dict) -> "MidiTransformer":
        """Rebuild architecture from tensor shapes (checkpoints are plain state dicts)."""
        vocab_size, d_model = state["embedding.weight"].shape
        use_rope = "pos_embedding.weight" not in state
        max_len = MAX_LEN if use_rope else state["pos_embedding.weight"].shape[0]
        ffn_dim = state["blocks.0.mlp.0.weight"].shape[0]
        layer_ids = {
            int(m.group(1))
            for key in state
            if (m := re.match(r"blocks\.(\d+)\.", key))
        }
        model = cls(
            vocab_size,
            d_model=d_model,
            n_heads=N_HEADS,
            n_layers=max(layer_ids) + 1,
            ffn_dim=ffn_dim,
            dropout=0.0,
            max_len=max_len,
            use_rope=use_rope,
        )
        # Training checkpoints may include loss buffers (pc_of_id, pc_member).
        model.load_state_dict(state, strict=False)
        return model


class Transformer(MidiTransformer):
    """Training wrapper: fit loop, AMP, checkpoints."""

    def __init__(
        self,
        train,
        val,
        vocab_size,
        device,
        pad_id,
        tokenizer=None,
        batch_size=2,
        accum_steps=1,
        num_workers=0,
        checkpoint_dir="checkpoints",
        max_len: int = MAX_LEN,
        lr: float | None = None,
    ):
        super().__init__(vocab_size, max_len=max_len)
        self.optimizer = optim.AdamW(
            self.parameters(), lr=lr or LR, weight_decay=WEIGHT_DECAY
        )
        self.device = device
        self.accum_steps = accum_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.raw_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        if tokenizer is not None:
            w = class_weights(tokenizer, vocab_size, torch.device(device))
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, weight=w)
            pc_of, pc_mem = pitch_lookups(tokenizer, vocab_size, torch.device(device))
            self.register_buffer("pc_of_id", pc_of)
            self.register_buffer("pc_member", pc_mem)
            self.use_music_loss = True
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
            self.use_music_loss = False
        self.use_amp = device == "cuda" and torch.cuda.is_bf16_supported()

        pin_memory = device == "cuda"
        loader_kw = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kw["persistent_workers"] = True
            loader_kw["prefetch_factor"] = 2

        self.train_data = DataLoader(train, shuffle=True, drop_last=True, **loader_kw)
        self.val_data = DataLoader(val, shuffle=False, **loader_kw)

    def _forward_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        def _compute(logits: torch.Tensor) -> torch.Tensor:
            ce = self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            if not self.use_music_loss:
                return ce
            pc = pitch_class_loss(logits, labels, self.pc_of_id, self.pc_member)
            return ce + PC_AUX * pc

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _compute(super().forward(inputs))
        return _compute(super().forward(inputs))

    def train_unit(self):
        self.train()
        running_loss = 0.0
        self.optimizer.zero_grad()

        batch_bar = tqdm(self.train_data, desc="  train", leave=False, unit="batch")
        for step, (inputs, labels) in enumerate(batch_bar):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            loss = self._forward_loss(inputs, labels)
            (loss / self.accum_steps).backward()
            running_loss += loss.item()

            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_data):
                nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            if step == 0 or (step + 1) % 15 == 0:
                tqdm.write(
                    f"  batch {step + 1}/{len(self.train_data)} loss={loss.item():.4f}"
                )

        return running_loss / len(self.train_data)

    def validate(self):
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(
                self.val_data, desc="  val  ", leave=False, unit="batch"
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                total_loss += self._forward_loss(inputs, labels).item()

        return total_loss / len(self.val_data)

    def _save_checkpoint(self, epoch):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.checkpoint_dir / f"epoch-{epoch + 1}" / f"{ts}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def fit(self, epochs, start_epoch=0):
        if start_epoch < 0 or start_epoch >= epochs:
            raise ValueError(f"start_epoch must be in [0, {epochs}), got {start_epoch}")

        n_params = sum(p.numel() for p in self.parameters())
        ckpt_root = self.checkpoint_dir.resolve()
        batches_per_epoch = len(self.train_data)
        eff_batch = self.train_data.batch_size * self.accum_steps
        remaining = epochs - start_epoch
        print(
            f"Begin training: transformer ({n_params / 1e6:.1f}M params), "
            f"epochs {start_epoch + 1}–{epochs} ({remaining} remaining), "
            f"{batches_per_epoch:,} train batches/epoch "
            f"(micro-batch {self.train_data.batch_size}, effective {eff_batch}), "
            f"amp={'bf16' if self.use_amp else 'off'}, "
            f"pos={'RoPE' if self.use_rope else 'absolute'}, "
            f"layers={len(self.blocks)}, "
            f"loss={'weighted CE (pitch/meter) + pitch-class' if self.use_music_loss else 'CE'}, "
            f"checkpoints -> {ckpt_root}/"
        )
        epoch_bar = tqdm(range(start_epoch, epochs), desc="epochs", unit="epoch")
        best_val = float("inf")
        best_epoch = start_epoch
        best_path = None
        for epoch in epoch_bar:
            train_loss = self.train_unit()
            val_loss = self.validate()

            epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

            ckpt = self._save_checkpoint(epoch)
            tqdm.write(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"saved {ckpt}"
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                best_path = ckpt
                shutil.copy2(ckpt, self.checkpoint_dir / "best.pt")

        if best_path is not None:
            state = torch.load(best_path, map_location=self.device, weights_only=True)
            self.load_state_dict(state)
            tqdm.write(
                f"Best val {best_val:.4f} at epoch {best_epoch} — "
                f"restored {best_path} for weights.pt"
            )
