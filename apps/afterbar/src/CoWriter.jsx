import { useEffect, useMemo, useRef, useState } from "react";
import { continueNotes, fetchCheckpoints } from "./api.js";
import { checkpointLabel, cowriterCheckpoints, preferCheckpoint } from "./checkpoints.js";
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

function pitchName(p) {
  return `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
}

let noteId = 1;

export default function CoWriter() {
  const engineRef = useRef(null);
  const [tempo, setTempo] = useState(100);
  const [bars, setBars] = useState(2);
  const [noteLen, setNoteLen] = useState(2); // cells (16ths)
  const [velocity, setVelocity] = useState(100); // for newly entered notes
  const [seed, setSeed] = useState([]); // {id, pitch, cell, cells}
  const [generated, setGenerated] = useState([]); // {pitch, start, duration, velocity} sec
  const [genMeta, setGenMeta] = useState(null);
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(400);
  const [emotion, setEmotion] = useState("none");
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [playPos, setPlayPos] = useState(0);
  const [error, setError] = useState(null);

  const spb = 60 / tempo / 4; // seconds per 16th cell
  const seedCols = bars * 16;

  if (!engineRef.current) engineRef.current = new SynthEngine();
  const engine = engineRef.current;

  useEffect(() => {
    engine.setParams(PRESETS["Neon Keys"]);
  }, [engine]);

  useEffect(() => {
    fetchCheckpoints()
      .then((data) => {
        const raw = data.checkpoints || [];
        const list = cowriterCheckpoints(raw);
        setCheckpoints(list);
        setCheckpoint((cur) => cur || preferCheckpoint(raw));
      })
      .catch((err) => setError(err.message));
  }, []);

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
        checkpoint: checkpoint || undefined,
        max_new_tokens: maxTokens,
        temperature,
        emotion,
        instrument: "piano",
        tempo,
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

  return (
    <div className="cowriter pg">
      <section className="panel">
        <h2>Sketch</h2>
        <p className="pg-lead">Click the roll to add or remove notes. Yours stay black.</p>
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
          Seed notes are black. Continuation from the model is amber.
        </p>
      </section>

      <section className="panel">
        <h2>Continue</h2>
        <p className="pg-lead">Prelude extends whatever you sketched.</p>
        <div className="cw-toolbar">
          {checkpoints.length > 0 ? (
          <div className="field compact cw-model">
            <label htmlFor="cw-ckpt">model</label>
            <select
              id="cw-ckpt"
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
              disabled={!checkpoints.length}
            >
              {!checkpoints.length && <option value="">no checkpoints found</option>}
              {checkpoints.map((c) => (
                <option key={c.path} value={c.path}>
                  {checkpointLabel(c)}
                </option>
              ))}
            </select>
          </div>
          ) : null}
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
          <div className="field compact">
            <label>emotion</label>
            <select value={emotion} onChange={(e) => setEmotion(e.target.value)}>
              <option value="none">none</option>
              <option value="Q1">Q1 happy / excited</option>
              <option value="Q2">Q2 tense / angry</option>
              <option value="Q3">Q3 sad / dark</option>
              <option value="Q4">Q4 calm / peaceful</option>
            </select>
          </div>
          <button type="button" className="primary" disabled={busy} onClick={runContinue}>
            {busy ? "Thinking…" : "Continue"}
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
            {genMeta.checkpoint
              ? genMeta.checkpoint.split("/").pop().replace(/\.pt$/i, "")
              : `${genMeta.model}/${genMeta.tokenizer}`}
            {genMeta.emotion && genMeta.emotion !== "none" ? ` · ${genMeta.emotion}` : ""}{" "}
            · {generated.length} notes ·{" "}
            <a href={genMeta.midi_url}>download MIDI</a>
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </section>
    </div>
  );
}
