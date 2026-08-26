# Demo video — shot list & narration

Target: **2:00–2:30**, screen capture + voiceover. Record at 1080p or higher
with system audio captured (macOS: QuickTime + BlackHole/Loopback, or OBS with
desktop audio). Keep the energy of showing a friend, not reading a paper.

Before recording:

1. `./scripts/run_lab.sh` and open http://localhost:8000 (use the final
   trained checkpoint, not the sanity one).
2. Practice the take once — the model call takes a few seconds; either keep it
   (it's honest) or trim it in the edit.
3. Browser at 100% zoom, close other tabs, mute notifications.

---

| # | Time | On screen | Say (roughly) |
|---|------|-----------|----------------|
| 1 | 0:00–0:15 | Co-writer tab, empty grid | "This is notelm — a co-writer for pop music. I sketch an idea, a Transformer I trained on nine hundred pop songs answers, and everything you'll hear is played by a synthesizer I built from scratch in the browser." |
| 2 | 0:15–0:35 | Click in a short melody by hand (don't use the stamp for the first take — hand entry reads as musicianship). Hit Play. | "I'll start with a four-bar idea." *(let it play — audience hears your DSP synth immediately)* |
| 3 | 0:35–1:00 | Click **Continue with AI**. Amber notes appear. Play the whole thing. | "Now I ask the model to continue. The amber notes are its answer — it picked up the key and the rhythm from my sketch." |
| 4 | 1:00–1:20 | Click **Accept into sketch**, tweak/delete a couple of amber-now-blue notes, continue again. | "It's a conversation: I keep what I like, edit what I don't, and ask again. I stay the writer; it's the collaborator." |
| 5 | 1:20–1:45 | Open the Synthesizer panel. Sweep the filter cutoff *while playing*, switch preset (e.g. Neon Keys → Pluck). | "The sound engine is my own DSP — polyBLEP oscillators, a state-variable filter, delay and reverb, computed sample by sample in an AudioWorklet. No samples, no presets I didn't design." |
| 6 | 1:45–2:05 | Click **Export WAV**; show the file landing in Downloads / drop it into a DAW timeline. | "Any sketch bounces to audio through the same synth — this WAV went straight into my session." |
| 7 | 2:05–2:20 | (Optional) Research lab tab, one glance at token statistics / tokenizer dropdown. | "Under the hood it's a research project too — I compare four MIDI tokenizations and two architectures." |
| 8 | 2:20–2:30 | Back to the grid, final playthrough. | "notelm — write a bar, get a bar back." *(end on the music)* |

Editing notes:

- Keep at least one full uninterrupted listen of a co-written phrase — the
  reviewers are musicians first.
- Show the mouse deliberately; don't rush cursor movements.
- If the model produces something bad on camera, keep one retry in the video —
  "sometimes it misses, so I just ask again" is a *strength* (it shows you
  understand the system).
