# notelm — a pop co-writer with its own synthesizer

*Project description for the Stanford Arts Portfolio (Music, Science & Technology).*
*Target length when exported to PDF: 1–2 pages. Sections marked [FILL IN] are personal
statements only you can write — keep them honest and specific.*

---

## Creative concept

notelm is a browser instrument for co-writing pop music with a neural network.
You sketch a fragment — a chord progression, a melody, a few notes — on a piano
roll, and a Transformer language model trained on pop piano arrangements
continues it. The continuation comes back onto the same canvas, where you can
accept it into your sketch, edit it, and ask the model to continue again:
composition as a conversation.

Everything you hear is played by a polyphonic subtractive synthesizer written
from scratch as a Web Audio AudioWorklet — two band-limited oscillators, a
state-variable filter, ADSR envelopes, an LFO, ping-pong delay, and a Schroeder
reverb, every sample computed by hand-written DSP code rather than by presets
or samples. The instrument can render any co-written sketch offline to a WAV
file through the same signal chain.

The project asks a question I care about as a songwriter: can a model be a
writing partner rather than a replacement — something that answers your idea
with a phrase you wouldn't have played, but that still sounds like your song?

## What the system is

- **Model.** A 19M-parameter decoder-only Transformer (6 layers, d_model 512,
  8 heads, pre-norm, weight-tied embeddings) trained with next-token
  cross-entropy on symbolic music, with an LSTM retained as a baseline.
- **Data.** POP909 (909 pop-song piano arrangements: melody, sub-melody, and
  accompaniment tracks), optionally preceded by pretraining on the full
  MAESTRO v3 corpus of classical performance and fine-tuning onto pop.
- **Representation.** MIDI is quantized to a 20 ms grid and encoded as event
  tokens (note-on/off, velocity bins, time shifts). Three alternative
  tokenizations (raw messages, REMI bars/positions, piano roll) are
  implemented for comparison; REMI's explicit bar structure is the fallback
  when rhythmic steadiness matters most.
- **Instrument.** React piano-roll interface, FastAPI inference server, and
  the custom AudioWorklet synthesizer with offline WAV export.
- **Infrastructure.** Training runs on a rented cloud GPU via a scripted
  pipeline (provision, sync, train, retrieve checkpoints, terminate) with a
  hard budget guard.

## Individual contribution

[FILL IN — Stanford asks each submission to clearly identify which parts you
completed. Be specific and honest, e.g.: which parts you designed and wrote
yourself, which parts started from existing work (the original notelm LSTM
lab), what tools — including AI coding assistants — you used and how you
directed, verified, and iterated on the work. Reviewers value clarity here
far more than inflated claims.]

## Technology used

Python, PyTorch (custom Transformer implementation, mixed-precision
training), pretty_midi, FastAPI; JavaScript, React, Web Audio API
(AudioWorklet DSP written by hand: polyBLEP oscillators, TPT state-variable
filter, Schroeder reverb); RunPod cloud GPUs (RTX A5000) driven by a scripted
training pipeline; POP909 and MAESTRO datasets.

## How the work developed through iteration

1. **Started as a classical MIDI research lab.** The first version was an
   LSTM trained on MAESTRO with four competing tokenizations and a research
   UI for comparing them — useful, but it generated meandering classical
   piano, and it wasn't the music I write.
2. **Retargeted to pop.** Swapped the corpus to POP909 and made the dataset
   switchable, keeping the train/validation split clean by excluding
   alternate versions of the same song.
3. **Upgraded the model.** Replaced the single-layer LSTM with a decoder-only
   Transformer; kept the LSTM as a baseline to compare against.
4. **Turned generation into conversation.** Built prompt-conditioned
   continuation (the model primes on your notes and answers), then an
   accept-and-continue loop so a sketch grows through repeated exchanges.
5. **Built the sound.** Replaced library playback with a synthesizer written
   sample-by-sample in an AudioWorklet, so the instrument has its own voice
   and any sketch can be bounced to audio through it.
6. **Made training reproducible.** Scripted the entire cloud training
   lifecycle with cost caps, so every checkpoint can be traced to a command.

## Honest limitations

The event representation has no explicit meter, so continuations can drift
rhythmically at high sampling temperatures (the REMI tokenization and lower
temperatures mitigate this). With 909 songs, the model can occasionally
reproduce near-verbatim fragments of training data; continuations are best
treated as suggestions to edit, not finished writing.

## Links

- Code repository: [FILL IN — GitHub URL]
- Demo video: [FILL IN — link or file in this portfolio]
- Audio examples: exported through the built-in synthesizer (WAV files in
  this portfolio)
