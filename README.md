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

Default sequence length: **4096** tokens. Windows are extracted with stride 1.

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

```bash
uv venv && source .venv/bin/activate
uv pip install torch pretty_midi tqdm
```

Download MAESTRO and place files under `data/maestro-v3.0.0/` (gitignored). Training currently reads from `data/maestro-v3.0.0/test/`.

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

## Layout

```
src/
  train.py          # entry point
  models/lstm.py    # LSTM + training loop
  utils/
    midi_fmt.py     # tokenizer + dataset
    data.py         # train/val split
scripts/
  train_tmux.sh     # background training
```
