from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from models.transformer import MidiTransformer
from utils.checkpoints import (
    default_model,
    infer_model_from_path,
    infer_tokenizer_from_path,
    legacy_tokenizer_dir,
    legacy_weights_path,
    weights_path,
)
from utils.emotion import emotion_token
from utils.instrument import instrument_token
from utils.tokenizers import BaseMidiTokenizer, get_tokenizer

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _search_roots() -> list[Path]:
    roots: list[Path] = [SRC, PROJECT, Path.cwd()]
    cwd_src = Path.cwd() / "src"
    if cwd_src.is_dir():
        roots.append(cwd_src)

    extra = os.environ.get("NOTELM_CHECKPOINT_DIRS", "")
    for part in extra.split(":"):
        if part.strip():
            roots.append(Path(part.strip()).expanduser())

    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(Path(resolved))
    return unique


def _is_training_dump(path: Path) -> bool:
    """Epoch folders and timestamped .pt files are training artifacts, not shipped weights."""
    if path.parent.name.lower().startswith("epoch-"):
        return True
    return bool(re.fullmatch(r"\d{8}-\d{6}\.pt", path.name, re.IGNORECASE))


def list_checkpoints() -> list[str]:
    """Find shipped .pt checkpoints (named snapshots + weights.pt)."""
    found: dict[str, Path] = {}
    patterns = (
        "weights.pt",
        "weights-*.pt",
        "checkpoints/*/weights.pt",
        "checkpoints/**/*.pt",
    )

    for root in _search_roots():
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and path.suffix == ".pt" and not _is_training_dump(path):
                    found[str(path.resolve())] = path.resolve()

    ordered = sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in ordered]


def default_checkpoint_for_tokenizer(
    tokenizer_name: str | None = None,
    model_name: str | None = None,
) -> str | None:
    tok = (tokenizer_name or os.environ.get("NOTELM_TOKENIZER", "remi")).strip().lower()
    model = (model_name or default_model()).strip().lower()
    for root in _search_roots():
        for candidate in (
            root / weights_path(model, tok),
            root / legacy_tokenizer_dir(tok) / "weights.pt",
            root / legacy_weights_path(tok),
            root / "weights.pt",
        ):
            if candidate.is_file():
                return str(candidate.resolve())
    ckpts = list_checkpoints()
    for p in ckpts:
        path = Path(p)
        if infer_tokenizer_from_path(path) == tok and infer_model_from_path(path) == model:
            return p
    return ckpts[0] if ckpts else None


def resolve_checkpoint(checkpoint: str) -> Path:
    raw = checkpoint.strip()
    if not raw or raw.startswith("("):
        raise FileNotFoundError(
            "No checkpoint selected. Train first, or paste a path to a .pt file."
        )

    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        for root in _search_roots():
            candidates.append(root / raw)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved

    searched = ", ".join(str(r) for r in _search_roots())
    raise FileNotFoundError(
        f"Checkpoint not found: {raw}\n"
        f"Searched under: {searched}"
    )


def load_model(
    checkpoint: str,
    device: torch.device | None = None,
    tokenizer: BaseMidiTokenizer | None = None,
) -> tuple[MidiTransformer, BaseMidiTokenizer]:
    device = device or get_device()
    ckpt_path = resolve_checkpoint(checkpoint)
    if tokenizer is None:
        tok_name = infer_tokenizer_from_path(ckpt_path)
        tokenizer = get_tokenizer(tok_name)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("enc_blocks.") for k in state):
        from models.canon import MidiCanon

        model = MidiCanon.from_state_dict(state)
    else:
        model = MidiTransformer.from_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
) -> int:
    logits = logits / max(temperature, 1e-6)
    if not torch.isfinite(logits).any():
        return 0

    if top_k > 0:
        k = min(top_k, int(logits.numel()))
        values, _ = torch.topk(logits, k)
        logits = logits.clone()
        logits[logits < values[-1]] = -float("inf")

    if 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        drop = cum - probs > top_p
        drop[0] = False
        sorted_logits = sorted_logits.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits[sorted_idx] = sorted_logits

    finite = torch.isfinite(logits)
    if not finite.any():
        return int(torch.argmax(logits).item())
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _event_kind(tokenizer: BaseMidiTokenizer, token_id: int) -> tuple[str, int | None]:
    name = tokenizer.id_to_token.get(int(token_id), "")
    if name.startswith("NOTE_ON_"):
        return "on", int(name.rsplit("_", 1)[-1])
    if name.startswith("NOTE_OFF_"):
        return "off", int(name.rsplit("_", 1)[-1])
    if name.startswith("TIME_SHIFT_"):
        return "shift", int(name.rsplit("_", 1)[-1])
    return name, None


def _active_pitches(tokenizer: BaseMidiTokenizer, tokens: list[int]) -> set[int]:
    active: set[int] = set()
    for tid in tokens:
        kind, pitch = _event_kind(tokenizer, tid)
        if kind == "on" and pitch is not None:
            active.add(pitch)
        elif kind == "off" and pitch is not None:
            active.discard(pitch)
    return active


def _mask_event_logits(
    logits: torch.Tensor,
    tokenizer: BaseMidiTokenizer,
    *,
    active: set[int],
    allow_eos: bool,
    max_time_shift_steps: int | None,
    pitch_min: int | None,
    pitch_max: int | None,
) -> torch.Tensor:
    """Block tokens that make spaghetti MIDI (PAD, 2s rests, orphan NOTE_OFFs)."""
    vocab = logits.numel()
    for name in ("PAD", "BOS", "UNK"):
        idx = tokenizer.token_to_id.get(name)
        if idx is not None and idx < vocab:
            logits[idx] = -float("inf")
    for tok, idx in tokenizer.token_to_id.items():
        if idx >= vocab:
            continue
        if tok.startswith("EMOTION_") or tok.startswith("INST_"):
            logits[idx] = -float("inf")
        if not allow_eos and tok == "EOS":
            logits[idx] = -float("inf")
        if tok.startswith("NOTE_OFF_"):
            pitch = int(tok.rsplit("_", 1)[-1])
            if pitch not in active:
                logits[idx] = -float("inf")
        if tok.startswith("NOTE_ON_"):
            pitch = int(tok.rsplit("_", 1)[-1])
            if pitch_min is not None and pitch < pitch_min:
                logits[idx] = -float("inf")
            if pitch_max is not None and pitch > pitch_max:
                logits[idx] = -float("inf")
        if max_time_shift_steps is not None and tok.startswith("TIME_SHIFT_"):
            steps = int(tok.rsplit("_", 1)[-1])
            if steps > max_time_shift_steps:
                logits[idx] = -float("inf")
    if not torch.isfinite(logits).any():
        eos = tokenizer.token_to_id.get("EOS")
        if eos is not None and eos < vocab:
            logits[eos] = 0.0
    return logits


def _cond_id(
    tokenizer: BaseMidiTokenizer, token_name: str, vocab_size: int
) -> int | None:
    tid = tokenizer.token_to_id.get(token_name)
    if tid is None or tid >= vocab_size:
        return None
    return tid


def _emotion_id(
    tokenizer: BaseMidiTokenizer, emotion: str | None, vocab_size: int
) -> int | None:
    return _cond_id(tokenizer, emotion_token(emotion or "none"), vocab_size)


def _instrument_id(
    tokenizer: BaseMidiTokenizer, instrument: str | None, vocab_size: int
) -> int | None:
    return _cond_id(tokenizer, instrument_token(instrument or "piano"), vocab_size)


def _prepend_cond(tokens: list[int], extra: list[int | None], bos: int) -> list[int]:
    ids = [i for i in extra if i is not None]
    if not ids:
        return tokens
    drop = set(ids)
    if tokens and tokens[0] == bos:
        return [bos] + ids + [t for t in tokens[1:] if t not in drop]
    return ids + [t for t in tokens if t not in drop]


def _seed_tokens(
    tokenizer: BaseMidiTokenizer,
    seed_midi: str | None,
    context_len: int,
    *,
    emotion_id: int | None = None,
    instrument_id: int | None = None,
) -> list[int]:
    bos = tokenizer.token_to_id["BOS"]
    eos = tokenizer.token_to_id["EOS"]
    pad = tokenizer.token_to_id["PAD"]
    if not seed_midi:
        tokens = [bos]
    else:
        tokens = tokenizer.encode_midi(seed_midi)
        tokens = [t for t in tokens if t not in (eos, pad)]
        tokens = tokens[-context_len:]
        if not tokens or tokens[0] != bos:
            tokens = [bos] + tokens
    return _prepend_cond(tokens, [emotion_id, instrument_id], bos)


@torch.no_grad()
def _generate_transformer(
    model: MidiTransformer,
    tokens: list[int],
    *,
    eos: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> list[int]:
    for _ in range(max_new_tokens):
        ctx = tokens[-model.max_len :]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        next_id = _sample(logits, temperature, top_k)

        if next_id == eos:
            break
        tokens.append(next_id)

    return tokens


@torch.no_grad()
def generate_tokens(
    model: MidiTransformer,
    tokenizer: BaseMidiTokenizer,
    *,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    seed_midi: str | None = None,
    context_len: int = 256,
    emotion: str | None = None,
    instrument: str | None = "piano",
    device: torch.device | None = None,
) -> list[int]:
    device = device or next(model.parameters()).device
    if hasattr(model, "enc_blocks"):
        raise TypeError("canon is a fill model — use POST /api/fill, not continue/generate")
    eos = tokenizer.token_to_id["EOS"]
    vocab_size = model.embedding.num_embeddings
    emotion_id = _emotion_id(tokenizer, emotion, vocab_size)
    instrument_id = _instrument_id(tokenizer, instrument, vocab_size)
    tokens = _seed_tokens(
        tokenizer,
        seed_midi,
        context_len,
        emotion_id=emotion_id,
        instrument_id=instrument_id,
    )

    return _generate_transformer(
        model,
        tokens,
        eos=eos,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )


def _strip_special(tokenizer: BaseMidiTokenizer, ids: list[int]) -> list[int]:
    drop = {
        tokenizer.token_to_id.get("PAD"),
        tokenizer.token_to_id.get("EOS"),
        tokenizer.token_to_id.get("UNK"),
        tokenizer.token_to_id.get("SPAN"),
        tokenizer.token_to_id.get("SEP"),
    }
    drop.discard(None)
    bos = tokenizer.token_to_id.get("BOS")
    out = [t for t in ids if t not in drop]
    if bos is not None and out and out[0] == bos:
        return out
    if bos is not None:
        return [bos] + out
    return out


@torch.no_grad()
def fill_span_tokens(
    model,
    tokenizer: BaseMidiTokenizer,
    *,
    left_midi: str | None,
    right_midi: str | None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    emotion: str | None = None,
    instrument: str | None = "piano",
    device: torch.device | None = None,
) -> list[int]:
    """Encode left+SPAN+right; sample the missing span on the decoder."""
    device = device or next(model.parameters()).device
    pad = tokenizer.token_to_id["PAD"]
    bos = tokenizer.token_to_id["BOS"]
    eos = tokenizer.token_to_id["EOS"]
    span_id = tokenizer.token_to_id["SPAN"]
    vocab_size = model.embedding.num_embeddings
    emotion_id = _emotion_id(tokenizer, emotion, vocab_size)
    instrument_id = _instrument_id(tokenizer, instrument, vocab_size)
    enc_max = int(getattr(model, "enc_max_len", 2048))
    dec_max = int(getattr(model, "dec_max_len", 768))

    left = _strip_special(tokenizer, tokenizer.encode_midi(left_midi)) if left_midi else [bos]
    right = []
    if right_midi:
        right = [
            t
            for t in tokenizer.encode_midi(right_midi)
            if t not in (pad, eos, bos, span_id)
        ]
    prefix = [i for i in (emotion_id, instrument_id) if i is not None]
    budget = enc_max - 1 - len(prefix)
    if not left:
        left = [bos]
    if len(left) + len(right) > budget:
        keep_left = min(len(left), max(1, budget // 2))
        left = left[-keep_left:]
        right = right[: budget - len(left)]
    enc = prefix + left + [span_id] + right
    enc = enc[:enc_max]
    while len(enc) < enc_max:
        enc.append(pad)
    enc_t = torch.tensor([enc], dtype=torch.long, device=device)
    enc_pad = enc_t.eq(pad)

    block = {
        tokenizer.token_to_id.get(n)
        for n in ("PAD", "BOS", "UNK", "SPAN", "SEP")
        if tokenizer.token_to_id.get(n) is not None
    }
    for name, idx in tokenizer.token_to_id.items():
        if name.startswith("EMOTION_") or name.startswith("INST_"):
            block.add(idx)

    tokens = [bos]
    for _ in range(max_new_tokens):
        if len(tokens) >= dec_max:
            break
        dec_t = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(enc_t, dec_t, enc_pad=enc_pad)[0, -1].clone()
        for idx in block:
            if idx is not None and idx < logits.numel():
                logits[idx] = -float("inf")
        next_id = _sample(logits, temperature, top_k)
        if next_id == eos:
            break
        tokens.append(next_id)
    return tokens


def generate_midi(
    checkpoint: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    seed_midi: str | None = None,
    context_len: int = 256,
    emotion: str | None = None,
    instrument: str | None = "piano",
) -> tuple[str, str, list[int]]:
    device = get_device()
    ckpt_path = resolve_checkpoint(checkpoint)
    model, tokenizer = load_model(str(ckpt_path), device)

    tokens = generate_tokens(
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed_midi=seed_midi,
        context_len=context_len,
        emotion=emotion,
        instrument=instrument,
        device=device,
    )

    out_dir = Path(tempfile.mkdtemp(prefix="notelm_"))
    midi_path = out_dir / "generated.midi"
    tokenizer.tokens_to_midi(tokens, midi_path)

    preview = " ".join(tokenizer.decode_tokens(tokens[:120]))
    if len(tokens) > 120:
        preview += f" ... (+{len(tokens) - 120} tokens)"

    return str(midi_path), preview, tokens
