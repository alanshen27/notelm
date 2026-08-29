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

export async function fetchCheckpoints(): Promise<{ checkpoints: Checkpoint[] }> {
  const r = await fetch(apiUrl("/api/checkpoints"));
  if (!r.ok) throw new Error(await r.text());
  return r.json();
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

export async function generatePhrase(payload?: {
  checkpoint?: string;
  max_new_tokens?: number;
  temperature?: number;
  emotion?: string;
  instrument?: string;
}) {
  const fd = new FormData();
  fd.append("max_new_tokens", String(payload?.max_new_tokens ?? 256));
  fd.append("temperature", String(payload?.temperature ?? 1.05));
  fd.append("top_k", "40");
  fd.append("context_len", "256");
  fd.append("emotion", payload?.emotion ?? "none");
  fd.append("instrument", payload?.instrument ?? "piano");
  if (payload?.checkpoint) fd.append("checkpoint", payload.checkpoint);
  const r = await fetch(apiUrl("/api/generate"), { method: "POST", body: fd });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json() as Promise<{
    notes: SynthNote[];
    midi_url?: string;
    model?: string;
    emotion?: string;
  }>;
}
