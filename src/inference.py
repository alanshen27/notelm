from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from models.lstm import MidiLSTM
from utils.midi_fmt import MidiTokenizer

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


def list_checkpoints() -> list[str]:
    """Find all .pt checkpoints under common project locations."""
    found: dict[str, Path] = {}
    patterns = ("weights.pt", "checkpoints/**/*.pt")

    for root in _search_roots():
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and path.suffix == ".pt":
                    found[str(path.resolve())] = path.resolve()

    ordered = sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in ordered]


def resolve_checkpoint(checkpoint: str) -> Path:
    """Resolve user input to an existing .pt file."""
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
        f"Searched under: {searched}\n"
        f"Hint: run `find /notelm -name '*.pt'` on your server."
    )


def load_model(checkpoint: str, device: torch.device | None = None) -> tuple[MidiLSTM, MidiTokenizer]:
    device = device or get_device()
    tokenizer = MidiTokenizer()
    model = MidiLSTM(tokenizer.vocab_size)

    ckpt_path = resolve_checkpoint(checkpoint)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)

    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[-1]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_tokens(
    model: MidiLSTM,
    tokenizer: MidiTokenizer,
    *,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    seed_midi: str | None = None,
    context_len: int = 256,
    device: torch.device | None = None,
) -> list[int]:
    device = device or next(model.parameters()).device
    bos = tokenizer.token_to_id["BOS"]
    eos = tokenizer.token_to_id["EOS"]
    pad = tokenizer.token_to_id["PAD"]

    if seed_midi:
        tokens = tokenizer.encode_midi(seed_midi)
        tokens = [t for t in tokens if t not in (eos, pad)]
        tokens = tokens[-context_len:]
        if not tokens or tokens[0] != bos:
            tokens = [bos] + tokens
    else:
        tokens = [bos]

    hidden = None
    out = None

    for token_id in tokens:
        x = torch.tensor([[token_id]], dtype=torch.long, device=device)
        emb = model.embedding(x)
        out, hidden = model.lstm(emb, hidden)

    for _ in range(max_new_tokens):
        logits = model.fc(out[:, -1, :]).squeeze(0)
        next_id = _sample(logits, temperature, top_k)

        if next_id == eos:
            break

        tokens.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        emb = model.embedding(x)
        out, hidden = model.lstm(emb, hidden)

    return tokens


def generate_midi(
    checkpoint: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    seed_midi: str | None = None,
    context_len: int = 256,
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
        device=device,
    )

    out_dir = Path(tempfile.mkdtemp(prefix="notelm_"))
    midi_path = out_dir / "generated.midi"
    tokenizer.tokens_to_midi(tokens, midi_path)

    preview = " ".join(tokenizer.decode_tokens(tokens[:120]))
    if len(tokens) > 120:
        preview += f" ... (+{len(tokens) - 120} tokens)"

    return str(midi_path), preview, tokens
