# notelm

Autoregressive MIDI generation with an LSTM over a discrete event vocabulary. MIDI files are tokenized into note-on/off, velocity, and time-shift events, then modeled as next-token prediction.

## Approach

Given a sequence of MIDI events \(x_{1:T}\), we train a language model to minimize cross-entropy on next-token targets:

\[
\mathcal{L} = -\sum_{t=1}^{T-1} \log p_\theta(x_{t+1} \mid x_{\leq t})
\]

Training uses teacher forcing on sliding windows extracted from [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro).

## Tokenization

Events are derived from `pretty_midi` and mapped to a fixed vocabulary:

| Token type | Description |
|---|---|
| `NOTE_ON_{pitch}` / `NOTE_OFF_{pitch}` | Piano range A0–C8 (21–108) |
| `VELOCITY_{bin}` | 16-bin quantization of MIDI velocity |
| `TIME_SHIFT_{steps}` | Relative time, 20 ms steps (max 2 s) |
| `BOS`, `EOS`, `PAD`, `UNK` | Sequence control |

Default sequence length: **4096** tokens. Windows use stride **2048** (half overlap). Train/val split is by MIDI file, not by window.

## Model

| Component | Spec |
|---|---|
| Embedding | `vocab_size → 128` |
| LSTM | 1 layer, hidden 512, `batch_first=True` |
| Head | Linear → `vocab_size` |
| Optimizer | Adam, lr = 1e-3 |
| Loss | Cross-entropy (token-level) |

Checkpoints are saved per epoch under `checkpoints/epoch-{n}/`.

## Setup

One-shot bootstrap (installs [uv](https://docs.astral.sh/uv/), Python 3.13, deps, training dirs):

```bash
./scripts/setup.sh --fetch-maestro
```

On an NVIDIA GPU machine, add CUDA PyTorch wheels:

```bash
./scripts/setup.sh --fetch-maestro --cuda
```

Other flags: `--lab` (music21 + npm UI deps), `--cpu` (skip CUDA wheels). Run `./scripts/setup.sh --help` for details.

MAESTRO can also be placed manually under `data/maestro-v3.0.0/` (gitignored). Training reads from `data/maestro-v3.0.0/2004/` by default (edit `DATA_DIR` in `src/utils/data.py` to change).

## Training

```bash
cd src && python train.py
```

Long runs (detachable session + log file):

```bash
brew install tmux   # once
./scripts/train_tmux.sh
tmux attach -t notelm-train
```

Final weights are written to `src/weights.pt`.

## Inference lab (React)

Research-oriented UI: sampling controls, token statistics, in-browser MIDI playback, run logs.

```bash
./scripts/run_lab.sh
# open http://localhost:8000
```

Dev mode (hot reload UI):

```bash
# terminal 1
cd src && uvicorn api:app --reload --port 8000

# terminal 2
cd ui && npm install && npm run dev
# open http://localhost:5173
```

Each run saves `outputs/<run_id>/generated.midi` and `run.json` (full token list + params).

**Sheet music:** §4 Notation renders the first 32 measures via MusicXML (requires `uv pip install music21`). Automatic MIDI→score transcription; fine for inspection, not publication quality.

Legacy Gradio UI: `cd src && python app.py` (port 7860).

## Email notification

Copy `.env.example` to `.env` and fill in your SMTP credentials. Training sends an email on success or failure.

```bash
cp .env.example .env
# Gmail: use an app password (Google Account → Security → App passwords)
```

Required vars: `NOTIFY_EMAIL`, `SMTP_PASS`. Optional: `SMTP_HOST` (default `smtp.gmail.com`), `SMTP_PORT` (default `587`), `SMTP_USER` (defaults to `NOTIFY_EMAIL`).

## Layout

```
src/
  train.py          # entry point
  api.py            # FastAPI inference lab backend
  app.py            # legacy Gradio UI
  inference.py      # load checkpoint + generate
  models/lstm.py    # LSTM + training loop
  utils/
    midi_fmt.py     # tokenizer + dataset
    data.py         # train/val split
scripts/
  setup.sh          # uv + deps + optional MAESTRO/CUDA
  train_tmux.sh     # background training
```
