import { useEffect, useRef, useState } from "react";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";

export default function ScoreViewer({ url, note }) {
  const containerRef = useRef(null);
  const osmdRef = useRef(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!url || !containerRef.current) return;

    let cancelled = false;
    setStatus("loading");
    setError(null);
    containerRef.current.innerHTML = "";

    const osmd = new OpenSheetMusicDisplay(containerRef.current, {
      autoResize: true,
      drawTitle: false,
      drawingParameters: "compacttight",
    });
    osmdRef.current = osmd;

    (async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || res.statusText);
        }
        const xml = await res.text();
        if (cancelled) return;
        await osmd.load(xml);
        if (cancelled) return;
        await osmd.render();
        if (!cancelled) setStatus("ready");
      } catch (e) {
        if (!cancelled) {
          setError(e.message);
          setStatus("error");
        }
      }
    })();

    return () => {
      cancelled = true;
      osmdRef.current = null;
    };
  }, [url]);

  if (!url) {
    return <p className="muted">Run inference to render notation.</p>;
  }

  return (
    <div className="score-block">
      {note && <p className="muted mono score-note">{note}</p>}
      {status === "loading" && <p className="muted">Engraving score…</p>}
      {error && (
        <p className="error">
          {error}
          {error.includes("music21") && (
            <> — on the server: <code>uv pip install music21</code></>
          )}
        </p>
      )}
      <div
        ref={containerRef}
        className={`score-canvas ${status === "ready" ? "ready" : ""}`}
      />
    </div>
  );
}
