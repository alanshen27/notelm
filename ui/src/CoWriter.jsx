import { useEffect, useMemo, useRef, useState } from "react";
import { continueNotes } from "./api.js";
import { PRESETS, SynthEngine } from "./synth/engine.js";

const PITCH_TOP = 84; // C6
const PITCH_BOTTOM = 48; // C3
const CELL_W = 26;
const CELL_H = 15;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const PROGRESSIONS = {
  "I – V – vi – IV": [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]],
  "vi – IV – I – V": [[57, 60, 64], [53, 57, 60], [60, 64, 67], [55, 59, 62]],
  "I – vi – ii – V": [[60, 64, 67], [57, 60, 64], [50, 53, 57], [55, 59, 62]],
};

const SYNTH_CONTROLS = [
  {
    group: "Oscillators",
    fields: [
      { key: "osc1Wave", label: "osc 1 wave", type: "wave" },
      { key: "osc2Wave", label: "osc 2 wave", type: "wave" },
      { key: "osc2Detune", label: "detune ¢", min: 0, max: 30, step: 1 },
      { key: "osc2Coarse", label: "osc 2 semi", min: -12, max: 12, step: 12 },
      { key: "oscMix", label: "mix", min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    group: "Filter (SVF lowpass)",
    fields: [
      { key: "cutoff", label: "cutoff Hz", min: 60, max: 8000, step: 10 },
      { key: "resonance", label: "resonance", min: 0, max: 1, step: 0.01 },
      { key: "envAmount", label: "env amt (oct)", min: 0, max: 5, step: 0.1 },
      { key: "filtA", label: "f attack", min: 0.001, max: 2, step: 0.001 },
      { key: "filtD", label: "f decay", min: 0.01, max: 2, step: 0.01 },
      { key: "filtS", label: "f sustain", min: 0, max: 1, step: 0.01 },
      { key: "filtR", label: "f release", min: 0.01, max: 3, step: 0.01 },
    ],
  },
  {
    group: "Amp envelope",
    fields: [
      { key: "ampA", label: "attack", min: 0.001, max: 2, step: 0.001 },
      { key: "ampD", label: "decay", min: 0.01, max: 2, step: 0.01 },
      { key: "ampS", label: "sustain", min: 0, max: 1, step: 0.01 },
      { key: "ampR", label: "release", min: 0.01, max: 3, step: 0.01 },
    ],
  },
  {
    group: "LFO",
    fields: [
      { key: "lfoTarget", label: "target", type: "lfoTarget" },
      { key: "lfoRate", label: "rate Hz", min: 0.1, max: 12, step: 0.1 },
      { key: "lfoDepth", label: "depth", min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    group: "Delay + Reverb",
    fields: [
      { key: "delayTime", label: "delay s", min: 0.05, max: 1.2, step: 0.005 },
      { key: "delayFeedback", label: "feedback", min: 0, max: 0.85, step: 0.01 },
      { key: "delayMix", label: "delay mix", min: 0, max: 1, step: 0.01 },
      { key: "reverbSize", label: "reverb size", min: 0, max: 1, step: 0.01 },
      { key: "reverbMix", label: "reverb mix", min: 0, max: 1, step: 0.01 },
      { key: "masterGain", label: "master", min: 0, max: 1.2, step: 0.01 },
    ],
  },
];

const WAVE_NAMES = ["saw", "square", "triangle", "sine"];
const LFO_TARGETS = ["off", "pitch", "cutoff"];

function pitchName(p) {
  return `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
}

let noteId = 1;

export default function CoWriter() {
  const engineRef = useRef(null);
  const [params, setParams] = useState({ ...PRESETS["Neon Keys"] });
  const [preset, setPreset] = useState("Neon Keys");
  const [tempo, setTempo] = useState(100);
  const [bars, setBars] = useState(2);
  const [noteLen, setNoteLen] = useState(2); // cells (16ths)
  const [velocity, setVelocity] = useState(100); // for newly entered notes
  const [seed, setSeed] = useState([]); // {id, pitch, cell, cells}
  const [generated, setGenerated] = useState([]); // {pitch, start, duration, velocity} sec
  const [genMeta, setGenMeta] = useState(null);
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(400);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [playPos, setPlayPos] = useState(0);
  const [error, setError] = useState(null);

  const spb = 60 / tempo / 4; // seconds per 16th cell
  const seedCols = bars * 16;

  if (!engineRef.current) engineRef.current = new SynthEngine();
  const engine = engineRef.current;

  useEffect(() => {
    engine.setParams(params);
  }, [engine, params]);

  useEffect(() => {
    engine.onPos = (s) => setPlayPos(s);
    engine.onEnded = () => {
      setPlaying(false);
      setPlayPos(0);
    };
  }, [engine]);

  const seedSeconds = useMemo(
    () =>
      seed.map((n) => ({
        pitch: n.pitch,
        start: n.cell * spb,
        duration: n.cells * spb,
        velocity: n.velocity ?? 100,
      })),
    [seed, spb]
  );

  const genEnd = generated.length
    ? Math.max(...generated.map((n) => n.start + n.duration))
    : seedCols * spb;
  const totalCols = Math.max(seedCols, Math.ceil(genEnd / spb) + 1);
  const rows = PITCH_TOP - PITCH_BOTTOM + 1;

  const gridClick = async (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / CELL_W);
    const row = Math.floor((e.clientY - rect.top) / CELL_H);
    const pitch = PITCH_TOP - row;
    if (col >= seedCols || pitch < PITCH_BOTTOM || pitch > PITCH_TOP) return;

    const hit = seed.find(
      (n) => n.pitch === pitch && col >= n.cell && col < n.cell + n.cells
    );
    if (hit) {
      setSeed((s) => s.filter((n) => n.id !== hit.id));
      return;
    }
    setSeed((s) => [
      ...s,
      { id: noteId++, pitch, cell: col, cells: noteLen, velocity },
    ]);
    await engine.noteOn(pitch, velocity);
    setTimeout(() => engine.noteOff(pitch), 180);
  };

  const stampProgression = (name) => {
    const chords = PROGRESSIONS[name];
    const cellsPerChord = seedCols / chords.length;
    const stamped = [];
    chords.forEach((chord, i) => {
      chord.forEach((pitch) => {
        stamped.push({
          id: noteId++,
          pitch,
          cell: Math.round(i * cellsPerChord),
          cells: Math.round(cellsPerChord),
          velocity,
        });
      });
    });
    setSeed(stamped);
    setGenerated([]);
  };

  const runContinue = async () => {
    if (!seedSeconds.length) {
      setError("Add some notes first (click the grid or stamp a progression).");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const out = await continueNotes({
        notes: seedSeconds,
        max_new_tokens: maxTokens,
        temperature,
      });
      setGenerated(out.notes);
      setGenMeta(out);
      if (!out.notes.length) {
        setError("Model returned an empty continuation — try again or raise max tokens.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const acceptContinuation = () => {
    const merged = generated.map((n) => ({
      id: noteId++,
      pitch: n.pitch,
      cell: Math.round(n.start / spb),
      cells: Math.max(1, Math.round(n.duration / spb)),
      velocity: n.velocity ?? 100,
    }));
    const newBars = Math.ceil(
      Math.max(...merged.map((n) => n.cell + n.cells), seedCols) / 16
    );
    setBars(newBars);
    setSeed((s) => [...s, ...merged]);
    setGenerated([]);
    setGenMeta(null);
  };

  const allNotes = useMemo(
    () => [...seedSeconds, ...generated],
    [seedSeconds, generated]
  );

  const togglePlay = async () => {
    if (playing) {
      engine.stop();
      return;
    }
    if (!allNotes.length) return;
    setPlaying(true);
    await engine.play(allNotes);
  };

  const exportWav = async () => {
    if (!allNotes.length) return;
    setBusy(true);
    try {
      const blob = await engine.renderWav(allNotes);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `notelm-cowrite-${Date.now()}.wav`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setBusy(false);
    }
  };

  const setParam = (key) => (value) => setParams((p) => ({ ...p, [key]: value }));

  const applyPreset = (name) => {
    setPreset(name);
    setParams({ ...PRESETS[name] });
  };

  return (
    <div className="cowriter">
      <section className="panel">
        <h2>§1 Sketch</h2>
        <div className="cw-toolbar">
          <div className="field compact">
            <label>tempo {tempo} bpm</label>
            <input
              type="range"
              min={60}
              max={160}
              value={tempo}
              onChange={(e) => setTempo(parseInt(e.target.value, 10))}
            />
          </div>
          <div className="field compact">
            <label>bars</label>
            <select value={bars} onChange={(e) => setBars(parseInt(e.target.value, 10))}>
              {[1, 2, 4].map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </div>
          <div className="field compact">
            <label>note length</label>
            <select
              value={noteLen}
              onChange={(e) => setNoteLen(parseInt(e.target.value, 10))}
            >
              <option value={1}>1/16</option>
              <option value={2}>1/8</option>
              <option value={4}>1/4</option>
              <option value={8}>1/2</option>
            </select>
          </div>
          <div className="field compact">
            <label>velocity {velocity}</label>
            <input
              type="range"
              min={30}
              max={127}
              value={velocity}
              onChange={(e) => setVelocity(parseInt(e.target.value, 10))}
            />
          </div>
          <div className="field compact">
            <label>stamp progression</label>
            <div className="cw-stamps">
              {Object.keys(PROGRESSIONS).map((name) => (
                <button key={name} type="button" onClick={() => stampProgression(name)}>
                  {name}
                </button>
              ))}
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              setSeed([]);
              setGenerated([]);
              setGenMeta(null);
            }}
          >
            Clear
          </button>
        </div>

        <div className="cw-roll-wrap">
          <div className="cw-pitchcol">
            {Array.from({ length: rows }, (_, r) => {
              const p = PITCH_TOP - r;
              return (
                <div
                  key={p}
                  className={`cw-pitchlabel ${p % 12 === 0 ? "c" : ""}`}
                  style={{ height: CELL_H }}
                >
                  {p % 12 === 0 ? pitchName(p) : ""}
                </div>
              );
            })}
          </div>
          <div
            className="cw-roll"
            style={{ width: totalCols * CELL_W, height: rows * CELL_H }}
            onClick={gridClick}
          >
            {Array.from({ length: rows }, (_, r) => {
              const p = PITCH_TOP - r;
              const black = [1, 3, 6, 8, 10].includes(p % 12);
              return (
                <div
                  key={p}
                  className={`cw-rowbg ${black ? "black" : ""}`}
                  style={{ top: r * CELL_H, height: CELL_H, width: totalCols * CELL_W }}
                />
              );
            })}
            {Array.from({ length: totalCols + 1 }, (_, c) => (
              <div
                key={c}
                className={`cw-gridline ${c % 16 === 0 ? "bar" : c % 4 === 0 ? "beat" : ""}`}
                style={{ left: c * CELL_W, height: rows * CELL_H }}
              />
            ))}
            {seedCols < totalCols && (
              <div
                className="cw-genregion"
                style={{
                  left: seedCols * CELL_W,
                  width: (totalCols - seedCols) * CELL_W,
                  height: rows * CELL_H,
                }}
              />
            )}
            {seed.map((n) => (
              <div
                key={n.id}
                className="cw-note seed"
                title={`${pitchName(n.pitch)} v${n.velocity ?? 100}`}
                style={{
                  left: n.cell * CELL_W + 1,
                  top: (PITCH_TOP - n.pitch) * CELL_H + 1,
                  width: n.cells * CELL_W - 2,
                  height: CELL_H - 2,
                  opacity: 0.45 + 0.55 * ((n.velocity ?? 100) / 127),
                }}
              />
            ))}
            {generated.map((n, i) => (
              <div
                key={`g${i}`}
                className="cw-note gen"
                title={`${pitchName(n.pitch)} v${n.velocity ?? 100}`}
                style={{
                  left: (n.start / spb) * CELL_W + 1,
                  top: (PITCH_TOP - n.pitch) * CELL_H + 1,
                  width: Math.max(4, (n.duration / spb) * CELL_W - 2),
                  height: CELL_H - 2,
                  opacity: 0.45 + 0.55 * ((n.velocity ?? 100) / 127),
                }}
              />
            ))}
            {playing && (
              <div
                className="cw-playhead"
                style={{ left: (playPos / spb) * CELL_W, height: rows * CELL_H }}
              />
            )}
          </div>
        </div>
        <p className="muted">
          Click to add / remove notes (seed, blue). AI continuation appears in amber.
        </p>
      </section>

      <section className="panel">
        <h2>§2 Co-write</h2>
        <div className="cw-toolbar">
          <div className="field compact">
            <label>continuation tokens {maxTokens}</label>
            <input
              type="range"
              min={100}
              max={1200}
              step={50}
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
            />
          </div>
          <div className="field compact">
            <label>temperature {temperature.toFixed(2)}</label>
            <input
              type="range"
              min={0.5}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>
          <button type="button" className="primary" disabled={busy} onClick={runContinue}>
            {busy ? "Thinking…" : "Continue with AI"}
          </button>
          {generated.length > 0 && (
            <button type="button" onClick={acceptContinuation}>
              Accept into sketch
            </button>
          )}
          <button type="button" onClick={togglePlay} disabled={!allNotes.length}>
            {playing ? "Stop" : "Play"}
          </button>
          <button type="button" onClick={exportWav} disabled={busy || !allNotes.length}>
            Export WAV
          </button>
        </div>
        {genMeta && (
          <p className="muted mono">
            {genMeta.model}/{genMeta.tokenizer} · {generated.length} notes ·{" "}
            <a href={genMeta.midi_url}>download MIDI</a>
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="panel">
        <h2>§3 Synthesizer</h2>
        <div className="field compact cw-preset">
          <label>preset</label>
          <select value={preset} onChange={(e) => applyPreset(e.target.value)}>
            {Object.keys(PRESETS).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <div className="cw-synth-grid">
          {SYNTH_CONTROLS.map((group) => (
            <div key={group.group} className="cw-synth-group">
              <h3>{group.group}</h3>
              {group.fields.map((f) => (
                <div key={f.key} className="field compact">
                  {f.type === "wave" || f.type === "lfoTarget" ? (
                    <>
                      <label>{f.label}</label>
                      <select
                        value={params[f.key]}
                        onChange={(e) => setParam(f.key)(parseInt(e.target.value, 10))}
                      >
                        {(f.type === "wave" ? WAVE_NAMES : LFO_TARGETS).map((w, i) => (
                          <option key={w} value={i}>
                            {w}
                          </option>
                        ))}
                      </select>
                    </>
                  ) : (
                    <>
                      <label>
                        {f.label}{" "}
                        <span className="value">
                          {Number(params[f.key]).toFixed(f.step >= 1 ? 0 : 2)}
                        </span>
                      </label>
                      <input
                        type="range"
                        min={f.min}
                        max={f.max}
                        step={f.step}
                        value={params[f.key]}
                        onChange={(e) => setParam(f.key)(parseFloat(e.target.value))}
                      />
                    </>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
