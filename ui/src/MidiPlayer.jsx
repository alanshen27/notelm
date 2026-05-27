import { useCallback, useEffect, useRef, useState } from "react";
import * as Tone from "tone";
import { Midi } from "@tonejs/midi";

/** Matches MidiTokenizerConfig.time_step_ms in src/utils/midi_fmt.py */
const MODEL_STEP_MS = 20;

const SYNTH_PRESETS = {
  synth: {
    label: "Synth",
    create: () =>
      new Tone.PolySynth(Tone.Synth, {
        envelope: { attack: 0.02, decay: 0.1, sustain: 0.3, release: 0.8 },
      }),
  },
  fm: {
    label: "FM",
    create: () =>
      new Tone.PolySynth(Tone.FMSynth, {
        envelope: { attack: 0.01, decay: 0.2, sustain: 0.2, release: 0.6 },
        modulationIndex: 8,
        harmonicity: 2,
      }),
  },
  am: {
    label: "AM",
    create: () =>
      new Tone.PolySynth(Tone.AMSynth, {
        envelope: { attack: 0.01, decay: 0.2, sustain: 0.25, release: 0.7 },
      }),
  },
  pluck: {
    label: "Pluck",
    create: () =>
      new Tone.PolySynth(Tone.PluckSynth, {
        attackNoise: 0.5,
        dampening: 2800,
        resonance: 0.85,
      }),
  },
};

function formatBpm(tempos) {
  if (!tempos?.length) return "—";
  const uniq = [...new Set(tempos.map((t) => Math.round(t.bpm)))];
  return uniq.length === 1 ? `${uniq[0]} BPM` : uniq.map((b) => `${b} BPM`).join(" → ");
}

export default function MidiPlayer({ url }) {
  const partsRef = useRef([]);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [sound, setSound] = useState("synth");
  const [meta, setMeta] = useState(null);

  const stop = useCallback(() => {
    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.playbackRate = 1;
    partsRef.current.forEach((p) => p.dispose());
    partsRef.current = [];
    setPlaying(false);
  }, []);

  useEffect(() => {
    if (!url) {
      setMeta(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(url);
        const buf = await res.arrayBuffer();
        const midi = new Midi(buf);
        if (cancelled) return;
        const markerBpm = midi.header.tempos[0]?.bpm ?? 120;
        setMeta({
          duration: midi.duration,
          markerBpm,
          tempoLabel: formatBpm(midi.header.tempos),
          noteCount: midi.tracks.reduce((n, t) => n + t.notes.length, 0),
        });
        setError(null);
      } catch (e) {
        if (!cancelled) setMeta(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [url]);

  const play = useCallback(async () => {
    if (!url) return;
    stop();
    setError(null);

    try {
      await Tone.start();
      const res = await fetch(url);
      const buf = await res.arrayBuffer();
      const midi = new Midi(buf);

      const preset = SYNTH_PRESETS[sound] ?? SYNTH_PRESETS.synth;

      midi.tracks.forEach((track) => {
        if (!track.notes.length) return;
        const synth = preset.create().toDestination();

        const part = new Tone.Part(
          (time, note) => {
            synth.triggerAttackRelease(note.name, note.duration, time, note.velocity);
          },
          track.notes.map((n) => ({
            time: n.time,
            name: n.name,
            duration: n.duration,
            velocity: n.velocity,
          }))
        ).start(0);

        partsRef.current.push(part, synth);
      });

      Tone.Transport.seconds = 0;
      Tone.Transport.playbackRate = playbackRate;

      const duration = (midi.duration || 30) / playbackRate;
      Tone.Transport.scheduleOnce(() => {
        stop();
      }, duration + 0.5);

      Tone.Transport.start();
      setPlaying(true);
      setReady(true);

      setMeta({
        duration: midi.duration,
        markerBpm: midi.header.tempos[0]?.bpm ?? 120,
        tempoLabel: formatBpm(midi.header.tempos),
        noteCount: midi.tracks.reduce((n, t) => n + t.notes.length, 0),
      });
    } catch (e) {
      setError(e.message);
      setPlaying(false);
    }
  }, [url, stop, playbackRate, sound]);

  if (!url) {
    return <p className="muted">No output yet. Run inference to render audio.</p>;
  }

  const wallDuration = meta?.duration
    ? (meta.duration / playbackRate).toFixed(1)
    : null;

  return (
    <div className="player">
      <div className="player-controls">
        <button type="button" onClick={play} disabled={playing}>
          {playing ? "Playing…" : "▶ Play"}
        </button>
        <button type="button" onClick={stop} disabled={!playing && !ready}>
          ■ Stop
        </button>
        <a href={url} download="generated.midi" className="link-btn">
          Download MIDI
        </a>
      </div>

      <div className="player-tuning">
        <div className="field compact">
          <label htmlFor="playback-rate">Playback speed</label>
          <input
            id="playback-rate"
            type="range"
            min={0.25}
            max={2}
            step={0.05}
            value={playbackRate}
            disabled={playing}
            onChange={(e) => setPlaybackRate(parseFloat(e.target.value))}
          />
          <span className="value mono">
            {playbackRate.toFixed(2)}×
            {wallDuration != null && ` · ~${wallDuration}s`}
          </span>
        </div>
        <div className="field compact">
          <label htmlFor="sound-preset">Sound</label>
          <select
            id="sound-preset"
            value={sound}
            disabled={playing}
            onChange={(e) => setSound(e.target.value)}
          >
            {Object.entries(SYNTH_PRESETS).map(([id, p]) => (
              <option key={id} value={id}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {meta && (
        <dl className="player-meta mono">
          <div>
            <dt>MIDI tempo marker</dt>
            <dd>{meta.tempoLabel}</dd>
          </div>
          <div>
            <dt>File duration</dt>
            <dd>{meta.duration.toFixed(1)}s · {meta.noteCount} notes</dd>
          </div>
          <div>
            <dt>Model grid</dt>
            <dd>{MODEL_STEP_MS} ms per TIME_SHIFT token</dd>
          </div>
        </dl>
      )}

      {error && <p className="error">{error}</p>}
      <p className="muted">
        Playback uses absolute note times from the file (not Transport BPM). Generated
        MIDI has no musical tempo — only a default 120 BPM tag from the writer. Use
        speed above if it feels too fast or slow. Click Play once to unlock audio.
      </p>
    </div>
  );
}
