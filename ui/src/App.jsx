import { useState } from "react";
import CoWriter from "./CoWriter.jsx";
import Lab from "./Lab.jsx";
import "./App.css";

const TABS = [
  { id: "cowriter", label: "Co-writer" },
  { id: "lab", label: "Research lab" },
];

export default function App() {
  const [tab, setTab] = useState("cowriter");

  return (
    <div className="lab">
      <header className="header">
        <div>
          <p className="kicker">notelm · a pop co-writer with its own synthesizer</p>
          <h1>{tab === "cowriter" ? "Co-writer" : "Inference lab"}</h1>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={tab === t.id ? "tab active" : "tab"}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === "cowriter" ? <CoWriter /> : <Lab />}

      <footer className="footer mono">
        transformer + lstm · event / raw / remi / piano_roll · POP909 + MAESTRO ·
        custom AudioWorklet DSP synth
      </footer>
    </div>
  );
}
