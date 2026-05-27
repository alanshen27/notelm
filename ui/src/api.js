const API = "";

export async function fetchHealth() {
  const r = await fetch(`${API}/api/health`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchCheckpoints() {
  const r = await fetch(`${API}/api/checkpoints`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function generate(form) {
  const body = new FormData();
  body.append("checkpoint", form.checkpoint);
  body.append("max_new_tokens", String(form.max_new_tokens));
  body.append("temperature", String(form.temperature));
  body.append("top_k", String(form.top_k));
  body.append("context_len", String(form.context_len));
  if (form.seed_midi) body.append("seed_midi", form.seed_midi);

  const r = await fetch(`${API}/api/generate`, { method: "POST", body });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}
