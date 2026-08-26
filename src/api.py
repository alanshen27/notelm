"""Research lab API — inference + static React UI."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pretty_midi
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import (
    _search_roots,
    default_checkpoint_for_tokenizer,
    generate_tokens,
    get_device,
    list_checkpoints,
    load_model,
    resolve_checkpoint,
)
from score import MAX_MEASURES, midi_to_musicxml, score_backend_available
from utils.checkpoints import MODEL_NAMES, infer_model_from_path, infer_tokenizer_from_path
from utils.tokenizers import TOKENIZER_NAMES, BaseMidiTokenizer

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent
UI_DIST = PROJECT / "ui" / "dist"
OUTPUTS = PROJECT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="notelm lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: dict[str, tuple] = {}


def _get_model(checkpoint: str):
    ckpt = str(resolve_checkpoint(checkpoint))
    if ckpt not in _model_cache:
        _model_cache[ckpt] = load_model(ckpt)
    return _model_cache[ckpt], ckpt


def _token_stats(tokenizer: BaseMidiTokenizer, tokens: list[int]) -> dict:
    names = tokenizer.decode_tokens(tokens)
    families = Counter(
        n.split("_")[0] if "_" in n else n for n in names
    )
    return {
        "length": len(tokens),
        "unique": len(set(tokens)),
        "families": dict(families.most_common()),
    }


@app.get("/api/health")
def health():
    return {
        "device": str(get_device()),
        "search_roots": [str(r) for r in _search_roots()],
        "ui_built": UI_DIST.exists(),
        "score_backend": score_backend_available(),
        "models": list(MODEL_NAMES),
        "tokenizers": list(TOKENIZER_NAMES),
    }


@app.get("/api/checkpoints")
def checkpoints():
    paths = list_checkpoints()
    items = []
    for p in paths:
        path = Path(p)
        tok = infer_tokenizer_from_path(path)
        model = infer_model_from_path(path)
        items.append({
            "path": p,
            "name": path.name,
            "model": model,
            "tokenizer": tok,
            "parent": path.parent.name,
            "modified": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        })
    return {"checkpoints": items, "search_roots": [str(r) for r in _search_roots()]}


@app.post("/api/generate")
async def generate(
    checkpoint: str = Form(...),
    max_new_tokens: int = Form(512),
    temperature: float = Form(1.0),
    top_k: int = Form(40),
    context_len: int = Form(256),
    seed_midi: UploadFile | None = File(None),
):
    try:
        (nn, tokenizer), ckpt_path = _get_model(checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    ckpt_path_obj = Path(ckpt_path)
    model_name = infer_model_from_path(ckpt_path_obj)
    tokenizer_name = tokenizer.name

    run_id = str(uuid.uuid4())[:8]
    run_dir = OUTPUTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_path = None
    if seed_midi and seed_midi.filename:
        seed_path = run_dir / f"seed_{seed_midi.filename}"
        seed_path.write_bytes(await seed_midi.read())

    device = get_device()
    tokens = generate_tokens(
        nn,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else 0,
        seed_midi=str(seed_path) if seed_path else None,
        context_len=context_len,
        device=device,
    )

    midi_path = run_dir / "generated.midi"
    tokenizer.tokens_to_midi(tokens, midi_path)

    decoded = tokenizer.decode_tokens(tokens)
    preview_n = 80
    preview = " ".join(decoded[:preview_n])
    if len(decoded) > preview_n:
        preview += f" … (+{len(decoded) - preview_n})"

    params = {
        "checkpoint": ckpt_path,
        "model": model_name,
        "tokenizer": tokenizer_name,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "context_len": context_len,
        "seed": seed_path.name if seed_path else None,
    }

    meta = {
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "model": model_name,
        "tokenizer": tokenizer_name,
        "params": params,
        "stats": _token_stats(tokenizer, tokens),
        "tokens_preview": preview,
        "midi_url": f"/api/runs/{run_id}/generated.midi",
        "score_url": f"/api/runs/{run_id}/score.musicxml",
        "score_note": (
            f"First {MAX_MEASURES} measures · MusicXML via music21"
            if score_backend_available()
            else "Install music21 for notation (uv pip install music21)"
        ),
    }

    with open(run_dir / "run.json", "w") as f:
        json.dump({**meta, "tokens": tokens}, f, indent=2)

    return meta


class NoteIn(BaseModel):
    pitch: int = Field(ge=0, le=127)
    start: float = Field(ge=0)
    duration: float = Field(gt=0)
    velocity: int = Field(default=100, ge=1, le=127)


class ContinueRequest(BaseModel):
    notes: list[NoteIn] = Field(min_length=1)
    checkpoint: str | None = None
    max_new_tokens: int = Field(default=512, ge=16, le=4096)
    temperature: float = Field(default=1.0, gt=0, le=3)
    top_k: int = Field(default=40, ge=0, le=300)
    context_len: int = Field(default=1024, ge=32, le=4096)


def _notes_to_midi(notes: list[NoteIn], path: Path) -> float:
    """Write user notes as a MIDI file; returns the seed end time in seconds."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for n in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.start + n.duration
            )
        )
    midi.instruments.append(inst)
    midi.write(str(path))
    return max(n.start + n.duration for n in notes)


def _default_cowriter_checkpoint() -> str | None:
    for model in ("transformer", "lstm"):
        ckpt = default_checkpoint_for_tokenizer("event", model)
        if ckpt:
            return ckpt
    return None


@app.post("/api/continue")
def continue_notes(req: ContinueRequest):
    """Co-writer: continue a user-entered melody/chord fragment."""
    ckpt_spec = req.checkpoint or _default_cowriter_checkpoint()
    if not ckpt_spec:
        raise HTTPException(status_code=404, detail="No trained checkpoint found.")
    try:
        (nn, tokenizer), ckpt_path = _get_model(ckpt_spec)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_id = str(uuid.uuid4())[:8]
    run_dir = OUTPUTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_path = run_dir / "seed.midi"
    seed_end = _notes_to_midi(req.notes, seed_path)

    tokens = generate_tokens(
        nn,
        tokenizer,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        seed_midi=str(seed_path),
        context_len=req.context_len,
        device=get_device(),
    )

    midi_path = run_dir / "generated.midi"
    tokenizer.tokens_to_midi(tokens, midi_path)

    # Split the decoded timeline into seed vs. continuation (20 ms grid → 10 ms slack).
    full = pretty_midi.PrettyMIDI(str(midi_path))
    generated = [
        {
            "pitch": int(note.pitch),
            "start": float(round(note.start, 4)),
            "duration": float(round(note.end - note.start, 4)),
            "velocity": int(note.velocity),
        }
        for inst in full.instruments
        for note in inst.notes
        if note.start >= seed_end - 0.010
    ]
    generated.sort(key=lambda n: (n["start"], n["pitch"]))

    meta = {
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "model": infer_model_from_path(Path(ckpt_path)),
        "tokenizer": tokenizer.name,
        "checkpoint": ckpt_path,
        "seed_end": round(seed_end, 4),
        "notes": generated,
        "midi_url": f"/api/runs/{run_id}/generated.midi",
        "stats": _token_stats(tokenizer, tokens),
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump({**meta, "tokens": tokens}, f, indent=2)
    return meta


@app.get("/api/runs/{run_id}/generated.midi")
def get_midi(run_id: str):
    path = OUTPUTS / run_id / "generated.midi"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Run not found")
    return FileResponse(path, media_type="audio/midi", filename="generated.midi")


@app.get("/api/runs/{run_id}/score.musicxml")
def get_score(run_id: str):
    midi_path = OUTPUTS / run_id / "generated.midi"
    xml_path = OUTPUTS / run_id / "score.musicxml"
    if not midi_path.is_file():
        raise HTTPException(status_code=404, detail="Run not found")

    if not xml_path.is_file():
        ok, err = midi_to_musicxml(midi_path, xml_path)
        if not ok:
            raise HTTPException(status_code=503, detail=err or "Score conversion failed")

    return FileResponse(
        xml_path,
        media_type="application/vnd.recordare.musicxml+xml",
        filename="score.musicxml",
    )


@app.get("/api/runs/{run_id}/run.json")
def get_run_meta(run_id: str):
    path = OUTPUTS / run_id / "run.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(path.read_text())


if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
