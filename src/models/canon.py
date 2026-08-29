"""Encoder-decoder span infill (canon): bidirectional encoder, causal decoder."""

from __future__ import annotations

import datetime
import re
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from models.transformer import (
    DROPOUT,
    FFN_DIM,
    GRAD_CLIP,
    LR,
    N_HEADS,
    WEIGHT_DECAY,
    Block,
    SelfAttention,
)
from utils.music_loss import PC_AUX, class_weights, pitch_class_loss, pitch_lookups

D_MODEL = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_MAX_LEN = 2048
DEC_MAX_LEN = 768


def _rank0() -> bool:
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


class CrossAttention(nn.Module):
    """Decoder queries attend to encoder memory. No RoPE (different lengths)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, d = x.shape
        s = memory.size(1)
        q = self.q(x).view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, dim=-1)
        k = k.view(b, s, self.n_heads, d // self.n_heads).transpose(1, 2)
        v = v.view(b, s, self.n_heads, d // self.n_heads).transpose(1, 2)
        attn_mask = None
        if memory_pad is not None:
            attn_mask = memory_pad[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class DecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, dropout: float, use_rope: bool = True
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(
            d_model, n_heads, dropout, causal=True, use_rope=use_rope
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_pad: torch.Tensor | None = None,
        tgt_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), key_padding_mask=tgt_pad)
        x = x + self.cross_attn(self.ln2(x), memory, memory_pad=memory_pad)
        x = x + self.mlp(self.ln3(x))
        return x


class MidiCanon(nn.Module):
    """6+6 × 512 encoder-decoder. Encoder is bidirectional; decoder is causal."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        enc_layers: int = ENC_LAYERS,
        dec_layers: int = DEC_LAYERS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
        enc_max_len: int = ENC_MAX_LEN,
        dec_max_len: int = DEC_MAX_LEN,
        use_rope: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.enc_max_len = enc_max_len
        self.dec_max_len = dec_max_len
        self.max_len = enc_max_len
        self.use_rope = use_rope
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.enc_blocks = nn.ModuleList(
            Block(d_model, n_heads, ffn_dim, dropout, use_rope=use_rope, causal=False)
            for _ in range(enc_layers)
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_blocks = nn.ModuleList(
            DecoderBlock(d_model, n_heads, ffn_dim, dropout, use_rope=use_rope)
            for _ in range(dec_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self, enc_ids: torch.Tensor, enc_pad: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.drop(self.embedding(enc_ids))
        for block in self.enc_blocks:
            h = block(h, key_padding_mask=enc_pad)
        return self.enc_norm(h)

    def forward(
        self,
        enc_ids: torch.Tensor,
        dec_ids: torch.Tensor,
        enc_pad: torch.Tensor | None = None,
        dec_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if enc_ids.size(1) > self.enc_max_len:
            raise ValueError(f"encoder length {enc_ids.size(1)} > {self.enc_max_len}")
        if dec_ids.size(1) > self.dec_max_len:
            raise ValueError(f"decoder length {dec_ids.size(1)} > {self.dec_max_len}")
        memory = self.encode(enc_ids, enc_pad)
        h = self.drop(self.embedding(dec_ids))
        for block in self.dec_blocks:
            h = block(h, memory, memory_pad=enc_pad, tgt_pad=dec_pad)
        return self.fc(self.norm(h))

    @classmethod
    def from_state_dict(cls, state: dict) -> "MidiCanon":
        vocab_size, d_model = state["embedding.weight"].shape
        ffn_dim = state["enc_blocks.0.mlp.0.weight"].shape[0]
        enc_n = max(
            int(m.group(1))
            for key in state
            if (m := re.match(r"enc_blocks\.(\d+)\.", key))
        ) + 1
        dec_n = max(
            int(m.group(1))
            for key in state
            if (m := re.match(r"dec_blocks\.(\d+)\.", key))
        ) + 1
        model = cls(
            vocab_size,
            d_model=d_model,
            n_heads=N_HEADS,
            enc_layers=enc_n,
            dec_layers=dec_n,
            ffn_dim=ffn_dim,
            dropout=0.0,
        )
        model.load_state_dict(state, strict=False)
        return model


class Canon(MidiCanon):
    """Training wrapper: span CE, AMP, checkpoints under checkpoints/canon/remi/."""

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
        lr: float | None = None,
        enc_max_len: int = ENC_MAX_LEN,
        dec_max_len: int = DEC_MAX_LEN,
    ):
        super().__init__(
            vocab_size, enc_max_len=enc_max_len, dec_max_len=dec_max_len
        )
        self.pad_id = pad_id
        self.optimizer = optim.AdamW(
            self.parameters(), lr=lr or LR, weight_decay=WEIGHT_DECAY
        )
        self.device = device
        self.accum_steps = accum_steps
        self.checkpoint_dir = Path(checkpoint_dir)
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
        self.ddp = None
        self.train_sampler = None

        pin_memory = device == "cuda"
        loader_kw = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kw["persistent_workers"] = True
            loader_kw["prefetch_factor"] = 2
        if torch.distributed.is_initialized():
            self.train_sampler = DistributedSampler(train, shuffle=True, drop_last=True)
            self.train_data = DataLoader(
                train, sampler=self.train_sampler, shuffle=False, drop_last=True, **loader_kw
            )
            val_sampler = DistributedSampler(val, shuffle=False, drop_last=False)
            self.val_data = DataLoader(val, sampler=val_sampler, shuffle=False, **loader_kw)
        else:
            self.train_data = DataLoader(train, shuffle=True, drop_last=True, **loader_kw)
            self.val_data = DataLoader(val, shuffle=False, **loader_kw)

    def _pad_masks(self, enc_ids: torch.Tensor, dec_ids: torch.Tensor):
        enc_pad = enc_ids.eq(self.pad_id)
        dec_pad = dec_ids.eq(self.pad_id)
        return enc_pad, dec_pad

    def _forward_loss(
        self, enc_ids: torch.Tensor, dec_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        enc_pad, dec_pad = self._pad_masks(enc_ids, dec_ids)

        def _compute(logits: torch.Tensor) -> torch.Tensor:
            ce = self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            if not self.use_music_loss:
                return ce
            pc = pitch_class_loss(logits, labels, self.pc_of_id, self.pc_member)
            return ce + PC_AUX * pc

        def _logits() -> torch.Tensor:
            if self.ddp is not None:
                return self.ddp(enc_ids, dec_ids, enc_pad, dec_pad)
            return super(Canon, self).forward(enc_ids, dec_ids, enc_pad, dec_pad)

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _compute(_logits())
        return _compute(_logits())

    def train_unit(self):
        self.train()
        running_loss = 0.0
        self.optimizer.zero_grad()
        batch_bar = tqdm(
            self.train_data,
            desc="  train",
            leave=False,
            unit="batch",
            disable=not _rank0(),
        )
        for step, (enc_ids, dec_ids, labels) in enumerate(batch_bar):
            enc_ids = enc_ids.to(self.device, non_blocking=True)
            dec_ids = dec_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            loss = self._forward_loss(enc_ids, dec_ids, labels)
            (loss / self.accum_steps).backward()
            running_loss += loss.item()
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_data):
                nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            if _rank0() and (step == 0 or (step + 1) % 15 == 0):
                tqdm.write(
                    f"  batch {step + 1}/{len(self.train_data)} loss={loss.item():.4f}"
                )
        return running_loss / len(self.train_data)

    def validate(self):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for enc_ids, dec_ids, labels in tqdm(
                self.val_data, desc="  val  ", leave=False, unit="batch", disable=not _rank0()
            ):
                enc_ids = enc_ids.to(self.device, non_blocking=True)
                dec_ids = dec_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                total_loss += self._forward_loss(enc_ids, dec_ids, labels).item()
        n = max(1, len(self.val_data))
        mean = total_loss / n
        if torch.distributed.is_initialized():
            t = torch.tensor([total_loss, float(n)], device=self.device)
            torch.distributed.all_reduce(t)
            mean = (t[0] / t[1]).item()
        return mean

    def _save_checkpoint(self, epoch):
        if not _rank0():
            return None
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
        if _rank0():
            print(
                f"Begin training: canon ({n_params / 1e6:.1f}M params), "
                f"epochs {start_epoch + 1}–{epochs} ({remaining} remaining), "
                f"{batches_per_epoch:,} train batches/epoch "
                f"(micro-batch {self.train_data.batch_size}, effective {eff_batch}), "
                f"amp={'bf16' if self.use_amp else 'off'}, "
                f"ddp={torch.distributed.is_initialized()}, "
                f"enc={len(self.enc_blocks)} dec={len(self.dec_blocks)} × {D_MODEL}, "
                f"loss={'weighted CE (pitch/meter) + pitch-class' if self.use_music_loss else 'CE'}, "
                f"checkpoints -> {ckpt_root}/"
            )
        epoch_bar = tqdm(
            range(start_epoch, epochs),
            desc="epochs",
            unit="epoch",
            disable=not _rank0(),
        )
        best_val = float("inf")
        best_epoch = start_epoch
        best_path = None
        for epoch in epoch_bar:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            train_loss = self.train_unit()
            val_loss = self.validate()
            epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
            ckpt = self._save_checkpoint(epoch)
            if _rank0():
                tqdm.write(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"train loss: {train_loss:.4f} | "
                    f"val loss: {val_loss:.4f} | "
                    f"saved {ckpt}"
                )
            if _rank0() and val_loss < best_val:
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
