"""Research lab API — inference + static React UI."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference import (
    _search_roots,
    generate_tokens,
    get_device,
    list_checkpoints,
    load_model,
    resolve_checkpoint,
)
from score import MAX_MEASURES, midi_to_musicxml, score_backend_available
from utils.midi_fmt import MidiTokenizer

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


def _token_stats(tokenizer: MidiTokenizer, tokens: list[int]) -> dict:
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
    }


@app.get("/api/checkpoints")
def checkpoints():
    paths = list_checkpoints()
    items = []
    for p in paths:
        path = Path(p)
        items.append({
            "path": p,
            "name": path.name,
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
        (model, tokenizer), ckpt_path = _get_model(checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_id = str(uuid.uuid4())[:8]
    run_dir = OUTPUTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_path = None
    if seed_midi and seed_midi.filename:
        seed_path = run_dir / f"seed_{seed_midi.filename}"
        seed_path.write_bytes(await seed_midi.read())

    device = get_device()
    tokens = generate_tokens(
        model,
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
