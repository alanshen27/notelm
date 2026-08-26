# notelm

A pop co-writer with its own synthesizer. Sketch a chord progression or melody
on a piano roll; a Transformer trained on pop piano arrangements continues it;
everything plays through a polyphonic subtractive synth written from scratch as
a Web Audio AudioWorklet — and bounces to WAV through the same DSP.

```
sketch (piano roll) ──► FastAPI /api/continue ──► Transformer (POP909)
        ▲                                              │
        └── accept / edit / continue again ◄── MIDI continuation
                              │
                    custom AudioWorklet synth ──► playback + WAV export
```

## Quickstart

```bash
./scripts/setup.sh --fetch-pop909 --lab   # uv + Python 3.13 + deps + data + UI
./scripts/run_lab.sh                      # build UI, serve on :8000
```

Open http://localhost:8000 — the **Co-writer** tab is the instrument, the
**Research lab** tab is the original sampling/analysis UI.

## The co-writer

- Click notes onto the grid (or stamp a progression: I–V–vi–IV etc.).
- **Continue with AI** primes the model on your notes and returns a
  continuation (amber). **Accept into sketch** merges it and lets you iterate.
- The synthesizer panel is a real subtractive synth, hand-written DSP running
  in an AudioWorklet: 2 polyBLEP oscillators, TPT state-variable lowpass with
  envelope + LFO modulation, ADSR amp envelope, ping-pong delay, Schroeder
  reverb, soft-clip master. Presets: Neon Keys, Soft Pad, Pluck, Warm Bass.
- **Export WAV** renders the sketch offline through the same signal chain
  (`ui/src/synth/worklet.js` is the whole sound engine).

## Model

Given MIDI event tokens \(x_{1:T}\), train next-token prediction:

\[
\mathcal{L} = -\sum_{t=1}^{T-1} \log p_\theta(x_{t+1} \mid x_{\leq t})
\]

| Architecture | Spec |
|---|---|
| `transformer` (default for co-writing) | decoder-only, 6 layers, d_model 512, 8 heads, pre-norm, weight-tied head, ~19M params, AdamW 3e-4, bf16 autocast on CUDA |
| `lstm` (baseline) | 1 layer, hidden 512, Adam 1e-3 |

Checkpoints: `checkpoints/{model}/{tokenizer}/epoch-N/` + final `weights.pt`.
Transformer checkpoints are plain state dicts; architecture is re-inferred
from tensor shapes on load.

## Data

| Dataset | Flag | Contents |
|---|---|---|
| `pop909` (default) | `--fetch-pop909` (~23 MB) | 909 pop piano arrangements (melody/bridge/piano merged; alternate versions excluded from the split) |
| `maestro` | `--fetch-maestro` (~120 MB) | MAESTRO v3, 2004 subset (legacy default) |
| `maestro_full` | `--fetch-maestro` | all MAESTRO years — used for pretraining |

Global grid `TIMESTEP_MS = 20`. Tokenizers: `event` (default), `raw`, `remi`
(bars/positions — steadiest rhythm), `piano_roll`. Sequence window 4096
tokens, stride 2048, split by file.

## Training

```bash
cd src && python train.py --model transformer --dataset pop909 --tokenizer event
python train.py --model transformer --dataset maestro_full --epochs 20   # pretrain
python train.py --model transformer --dataset pop909 --epochs 60 \
  --lr 1e-4 --weights <pretrain.pt> --start-epoch 0                      # fine-tune
python train.py --all-tokenizers                                        # comparisons
```

Useful flags: `--seq-len N`, `--limit-files N` (smoke tests), `--lr`,
`--epoch N` (resume). Long runs: `./scripts/train_tmux.sh <args>`.

### Cloud training (RunPod)

One command per lifecycle step, or fully automatic — provisions a 1x RTX A5000
(24 GB, secure cloud), syncs the repo, sets up, trains in tmux, monitors with
a hard budget guard, downloads checkpoints, and terminates the pod:

```bash
export RUNPOD_API_KEY=...        # or put it in .env
python3 scripts/runpod_train.py check                      # read-only: key, balance, pods
python3 scripts/runpod_train.py full --recipe pretrain-finetune --budget 25
python3 scripts/runpod_train.py status | logs | terminate  # manual control
```

The `pretrain-finetune` recipe runs: MAESTRO-full pretrain (20 epochs) →
POP909 fine-tune (event) → POP909 REMI. Expected cost ~$4 at $0.27/hr.

## Inference & API

- `POST /api/continue` — JSON `{notes: [{pitch, start, duration, velocity}]}`
  → continuation notes + MIDI. Powers the co-writer.
- `POST /api/generate` — free-form sampling (checkpoint, temperature, top-k,
  optional seed MIDI upload). Powers the research lab.
- Every run is logged to `outputs/<run_id>/` (MIDI + full token list + params).

Dev mode: `cd src && uvicorn api:app --reload --port 8000` and
`cd ui && npm run dev` (proxy on :5173).

## Portfolio

`portfolio/` contains the Stanford Arts Portfolio materials: project
description, demo video script, and a music résumé template.

## Layout

```
src/
  train.py                 # training CLI (models × tokenizers × datasets)
  api.py                   # FastAPI: /api/continue + research lab
  inference.py             # checkpoint loading + generation (lstm + transformer)
  models/{lstm,transformer}.py
  utils/data.py            # dataset registry + windowing
  utils/tokenizers/        # event, raw, remi, piano_roll
ui/src/
  CoWriter.jsx             # piano roll + co-writing flow
  synth/worklet.js         # the synthesizer (hand-written DSP)
  synth/engine.js          # worklet host + offline WAV render
scripts/
  setup.sh                 # bootstrap (+ --fetch-pop909 / --fetch-maestro / --cuda)
  runpod_train.py          # cloud GPU lifecycle with budget guard
```

## Email notification

`cp .env.example .env` and fill in SMTP credentials (`NOTIFY_EMAIL`,
`SMTP_PASS`; optional `SMTP_HOST`/`SMTP_PORT`/`SMTP_USER`). Training emails on
completion or failure.
