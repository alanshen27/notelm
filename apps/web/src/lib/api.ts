import { apiUrl } from "./routes";

export type Checkpoint = {
  path: string;
  name: string;
  model?: string;
  tokenizer?: string;
  parent?: string;
};

export type SynthNote = {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
};

function requestUrls(path: string) {
  const p = path.startsWith("/") ? path : `/${path}`;
  const urls = [apiUrl(p)];
  const baked = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/+$/, "");
  if (baked) urls.push(`${baked}${p}`);
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host !== "localhost" && host !== "127.0.0.1") {
      urls.push(`https://notelm-api.onrender.com${p}`);
    }
  }
  return [...new Set(urls.filter(Boolean))];
}

export async function fetchCheckpoints(): Promise<{ checkpoints: Checkpoint[] }> {
  let last = "Could not load models.";
  const urls = [
    ...requestUrls("/api/checkpoints"),
    "https://notelm-api.onrender.com/api/checkpoints",
  ];
  for (const url of [...new Set(urls)]) {
    try {
      const r = await fetch(url);
      if (!r.ok) {
        last = r.statusText || last;
        continue;
      }
      return r.json();
    } catch (err) {
      last = err instanceof Error ? err.message : last;
    }
  }
  throw new Error(last);
}

export async function continueNotes(payload: {
  notes: SynthNote[];
  checkpoint?: string;
  max_new_tokens: number;
  temperature: number;
  emotion: string;
  instrument?: string;
  tempo: number;
  range_start?: number;
  range_end?: number;
}) {
  const r = await fetch(apiUrl("/api/continue"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json() as Promise<{
    notes: SynthNote[];
    checkpoint?: string;
    model?: string;
    tokenizer?: string;
    emotion?: string;
    instrument?: string;
    midi_url?: string;
  }>;
}

function readError(err: unknown, fallback: string) {
  if (typeof err === "string" && err.trim()) return err;
  if (err && typeof err === "object" && "detail" in err) {
    const detail = (err as { detail: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail) && detail[0] && typeof detail[0] === "object" && "msg" in detail[0]) {
      return String((detail[0] as { msg: unknown }).msg);
    }
  }
  return fallback;
}

function generateEndpoints() {
  return requestUrls("/api/generate");
}

export async function generatePhrase(payload?: {
  checkpoint?: string;
  max_new_tokens?: number;
  temperature?: number;
  emotion?: string;
  instrument?: string;
}) {
  const form = () => {
    const fd = new FormData();
    fd.append("max_new_tokens", String(payload?.max_new_tokens ?? 256));
    fd.append("temperature", String(payload?.temperature ?? 1.05));
    fd.append("top_k", "40");
    fd.append("context_len", "256");
    fd.append("emotion", payload?.emotion ?? "none");
    fd.append("instrument", payload?.instrument ?? "piano");
    if (payload?.checkpoint) fd.append("checkpoint", payload.checkpoint);
    return fd;
  };

  let last = "Could not generate.";
  for (const url of generateEndpoints()) {
    try {
      const r = await fetch(url, { method: "POST", body: form() });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        last = readError(err, r.statusText);
        continue;
      }
      return r.json() as Promise<{
        notes: SynthNote[];
        midi_url?: string;
        model?: string;
        emotion?: string;
      }>;
    } catch (err) {
      last = err instanceof Error ? err.message : last;
    }
  }
  throw new Error(last);
}
