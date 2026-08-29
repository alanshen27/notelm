import CoWriter from "./CoWriter.jsx";

export default function App() {
  const electron = typeof navigator !== "undefined" && /Electron/i.test(navigator.userAgent);

  return (
    <div className={`plugin ${electron ? "plugin-electron" : ""}`}>
      <header className="plugin-titlebar">
        {!electron && (
          <div className="plugin-lights" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
        )}
        <div className="plugin-title">
          <strong>clavier</strong>
          <span className="plugin-sep">/</span>
          <span>MIDI continuation</span>
        </div>
        <div className="plugin-meta">
          <span>host notate</span>
          <span>prelude</span>
          <span>48 kHz</span>
        </div>
        <a className="plugin-lab" href="/">
          notate
        </a>
      </header>
      <div className="plugin-body">
        <CoWriter />
      </div>
      <footer className="plugin-status">
        <span>insert · MIDI in</span>
        <span>continue writes after the seed</span>
        <span>notate / clavier</span>
      </footer>
    </div>
  );
}
