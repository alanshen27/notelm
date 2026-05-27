import { useCallback, useEffect, useState } from "react";
import { fetchCheckpoints, fetchHealth, generate } from "./api.js";
import MidiPlayer from "./MidiPlayer.jsx";
import ScoreViewer from "./ScoreViewer.jsx";
import "./App.css";

const DEFAULTS = {
  max_new_tokens: 512,
  temperature: 1.0,
  top_k: 40,
  context_len: 256,
};

export default function App() {
  const [health, setHealth] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [searchRoots, setSearchRoots] = useState([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [customPath, setCustomPath] = useState("");
  const [params, setParams] = useState(DEFAULTS);
  const [seedFile, setSeedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    const [h, c] = await Promise.all([fetchHealth(), fetchCheckpoints()]);
    setHealth(h);
    setCheckpoints(c.checkpoints);
    setSearchRoots(c.search_roots);
    if (!checkpoint && c.checkpoints.length) {
      setCheckpoint(c.checkpoints[0].path);
    }
  }, [checkpoint]);

  useEffect(() => {
    refresh().catch((e) => setError(e.message));
  }, []);

  const activeCheckpoint = customPath.trim() || checkpoint;

  const run = async (e) => {
    e.preventDefault();
    if (!activeCheckpoint) {
      setError("Select or paste a checkpoint path.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const out = await generate({
        checkpoint: activeCheckpoint,
        ...params,
        seed_midi: seedFile,
      });
      setResult(out);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const setParam = (key) => (e) => {
    const v = e.target.type === "range" ? parseFloat(e.target.value) : e.target.value;
    setParams((p) => ({ ...p, [key]: v }));
  };

  return (
    <div className="lab">
      <header className="header">
        <div>
          <p className="kicker">notelm · autoregressive MIDI</p>
          <h1>Inference lab</h1>
        </div>
        <div className="header-meta mono">
          {health && (
            <>
              <span>device: {health.device}</span>
              <span>ui: {health.ui_built ? "built" : "dev proxy"}</span>
            </>
          )}
        </div>
      </header>

      <form className="grid" onSubmit={run}>
        <section className="panel">
          <h2>§1 Checkpoint</h2>
          <div className="field">
            <label htmlFor="ckpt-select">Detected weights</label>
            <select
              id="ckpt-select"
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
            >
              <option value="">— select —</option>
              {checkpoints.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.parent}/{c.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="ckpt-path">Or absolute path</label>
            <input
              id="ckpt-path"
              type="text"
              placeholder="/notelm/checkpoints/epoch-1/….pt"
              value={customPath}
              onChange={(e) => setCustomPath(e.target.value)}
            />
          </div>
          <button type="button" onClick={() => refresh().catch((e) => setError(e.message))}>
            Refresh index
          </button>
          {searchRoots.length > 0 && (
            <details className="roots">
              <summary>Search roots</summary>
              <ul className="mono">
                {searchRoots.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </details>
          )}
        </section>

        <section className="panel">
          <h2>§2 Sampling</h2>
          <div className="field">
            <label htmlFor="max-tokens">max_new_tokens</label>
            <input
              id="max-tokens"
              type="range"
              min={64}
              max={2048}
              step={64}
              value={params.max_new_tokens}
              onChange={setParam("max_new_tokens")}
            />
            <span className="value">{params.max_new_tokens}</span>
          </div>
          <div className="field">
            <label htmlFor="temp">temperature τ</label>
            <input
              id="temp"
              type="range"
              min={0.1}
              max={2}
              step={0.05}
              value={params.temperature}
              onChange={setParam("temperature")}
            />
            <span className="value">{params.temperature.toFixed(2)}</span>
          </div>
          <div className="field">
            <label htmlFor="topk">top_k (0 = off)</label>
            <input
              id="topk"
              type="range"
              min={0}
              max={100}
              step={1}
              value={params.top_k}
              onChange={setParam("top_k")}
            />
            <span className="value">{params.top_k}</span>
          </div>
          <div className="field">
            <label htmlFor="ctx">seed context_len</label>
            <input
              id="ctx"
              type="range"
              min={32}
              max={1024}
              step={32}
              value={params.context_len}
              onChange={setParam("context_len")}
            />
            <span className="value">{params.context_len}</span>
          </div>
        </section>

        <section className="panel">
          <h2>§3 Conditioning</h2>
          <div className="field">
            <label htmlFor="seed">Seed MIDI (optional)</label>
            <input
              id="seed"
              type="file"
              accept=".midi,.mid"
              onChange={(e) => setSeedFile(e.target.files?.[0] ?? null)}
            />
            {seedFile && <span className="value">{seedFile.name}</span>}
          </div>
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Running inference…" : "Run inference"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>
      </form>

      {result && (
        <section className="panel output">
          <h2>§4 Output — run {result.run_id}</h2>
          <div className="meta-grid mono">
            <div>
              <span className="meta-label">checkpoint</span>
              <span>{result.params.checkpoint}</span>
            </div>
            <div>
              <span className="meta-label">device</span>
              <span>{result.device}</span>
            </div>
            <div>
              <span className="meta-label">tokens</span>
              <span>{result.stats.length}</span>
            </div>
            <div>
              <span className="meta-label">unique</span>
              <span>{result.stats.unique}</span>
            </div>
          </div>

          <h3>Token families</h3>
          <table className="stats-table mono">
            <thead>
              <tr>
                <th>family</th>
                <th>count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.stats.families).map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3>Playback</h3>
          <MidiPlayer url={result.midi_url} />

          <h3>Notation</h3>
          <ScoreViewer url={result.score_url} note={result.score_note} />

          <h3>Token stream (prefix)</h3>
          <pre className="token-stream">{result.tokens_preview}</pre>

          <p className="muted mono">
            Full run logged at{" "}
            <a href={`/api/runs/${result.run_id}/run.json`}>runs/{result.run_id}/run.json</a>
          </p>
        </section>
      )}

      <footer className="footer mono">
        LSTM · event vocabulary · teacher forcing · MAESTRO
      </footer>
    </div>
  );
}
