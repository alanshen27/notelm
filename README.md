# notelm

A pop co-writer with its own synthesizer. Sketch a chord progression or melody
on a piano roll; a Transformer continues it; everything plays through a
polyphonic subtractive synth written from scratch as a Web Audio AudioWorklet.

```
sketch (piano roll) ──► FastAPI /api/continue ──► Transformer
        ▲                                              │
        └── accept / edit / continue again ◄── MIDI continuation
                              │
                    custom AudioWorklet synth ──► playback + WAV export
```

## Quickstart

```bash
./scripts/setup.sh --fetch-pop909 --lab   # uv + Python 3.13 + deps + POP909 + UI
./scripts/run_lab.sh                      # build UI, serve on :8000
```

Open http://localhost:8000 for the **site**. The instrument is **clavier**
at http://localhost:8000/app/ — the only place that runs a model.

Desktop (Electron):

```bash
npm run desktop
```

More training MIDI (EMOPIA emotion clips, Pop1K7, ADL piano, ASAP scores):

```bash
./scripts/setup.sh --fetch-extra
# GigaMIDI is gated + huge; optional piano subset:
#   uv pip install -e '.[data]'
#   export HF_TOKEN=hf_...   # accept terms on the Hub first
#   ./scripts/setup.sh --fetch-gigamidi
```

## The co-writer

- **Playground** (`/playground/`): one click, no grid. Prelude writes a phrase
  and plays it.
- Click notes onto the grid (or stamp a progression: I–V–vi–IV etc.).
- **Continue with AI** primes the model on your notes and returns a
  continuation (amber). **Accept into sketch** merges it and lets you iterate.
- **Emotion** (Q1–Q4) is an EMOPIA-style condition. It only changes the notes
  after you train a checkpoint that includes the extra `EMOTION_*` tokens
  (current POP909-only weights ignore it).
- The synthesizer is an AudioWorklet inside clavier (`apps/web/public/synth/worklet.js`). **Export WAV** uses the same DSP.

## Model

Decoder-only Transformer, 6×512, 8 heads, ~21M params, next-token CE on REMI
tokens (bar, position, pitch, velocity, duration) plus optional emotion prefix.
Shipped weights: `src/checkpoints/transformer/remi/{prelude,etude}.pt`.

## Data

| Name | Flag | What it is |
|---|---|---|
| `pop909` | `--fetch-pop909` | 909 pop piano arrangements (default) |
| `pop1k7` | `--fetch-extra` | 1,747 pop piano transcriptions |
| `emopia` | `--fetch-extra` | pop piano clips with Q1–Q4 emotion in the filename |
| `adl` | `--fetch-extra` | ~11k piano MIDIs (Lakh piano family + scrapes) |
| `asap` | `--fetch-extra` | aligned classical piano scores |
| `maestro` / `maestro_full` | `--fetch-maestro` | MAESTRO performances (2004 / all years) |
| `gigamidi` | `--fetch-gigamidi` | piano-ish GigaMIDI pull (gated, CC BY-NC; skip for a public app) |
| `giantmidi` | `--fetch-ccby` | GiantMIDI-Piano (~10k classical transcriptions, CC BY) |
| `atepp` | `--fetch-ccby` | ATEPP (~11k expressive piano performances) |
| `pdmx` | `--fetch-ccby` | PDMX public-domain scores, all instruments (MIDI only; cap 40k) |
| `pop` | union | pop909 + pop1k7 + emopia |
| `pretrain` | union | maestro_full + adl + asap + piano CC-BY sets |
| `instruments` | union | PDMX + piano corpora (canon pretrain, all insts) |
| `piano` | union | piano-only mix (canon finetune) |
| `canon` | union | instruments mix (single-run fallback) |
| `all` | union | everything present on disk |

Split is by **file path** (90/10). Emotion tokens are inferred from EMOPIA-style
`Q1_…` names. Instrument tokens (`INST_piano`, …) are inferred from GM programs
in the MIDI. Clavier sends `INST_piano`.

Canon trains in two stages: `--dataset instruments` then `--dataset piano`.

```bash
cd src && python train.py --dataset pop --epochs 40
python train.py --dataset pretrain --epochs 20
python train.py --dataset pop --epochs 30 --lr 1e-4 \
  --weights checkpoints/transformer/remi/prelude.pt --start-epoch 0
```

## Cloud (RunPod)

```bash
python3 scripts/runpod_train.py check
python3 scripts/runpod_train.py train --recipe pop --epochs 40
# or: --recipe pretrain-finetune
python3 scripts/runpod_train.py fetch-gigamidi   # 50k piano-ish GigaMIDI for canon
python3 scripts/runpod_train.py monitor --budget 200 --max-hours 48
python3 scripts/runpod_train.py terminate   # you stop billing
```

## Cloud (Render)

Inference only — Render has no GPU. The ~21M Transformer fits in RAM and
runs on CPU. Dashboard: New → Blueprint (`render.yaml`), or sync the Blueprint
if `notelm` already exists.

Two services:

| Service | Image | Plan | What it does |
|---|---|---|---|
| `notelm` | `Dockerfile` | starter | Next static site; proxies `/api` to the API |
| `notelm-api` | `Dockerfile.api` | `4c-8g` | FastAPI + PyTorch CPU. Free 512 MB will OOM. |

Public URL stays https://notelm.onrender.com. The browser still calls `/api/...`
on that host; the site container forwards those requests to
https://notelm-api.onrender.com.

Shipped checkpoints are `prelude.pt` and `etude.pt` (REMI).

## API

- `POST /api/continue` — `{notes, emotion?}` → continuation. Co-writer.
- `POST /api/generate` — sampling from scratch (playground) or optional seed MIDI.

## Layout

```
apps/web                  site + playground + clavier
apps/afterbar             clavier (legacy vite app)
src/train.py              Transformer training
src/api.py                FastAPI — site + /app + /api
src/models/transformer.py
src/utils/data.py         corpus registry
scripts/fetch_datasets.py extra MIDI downloads
```
