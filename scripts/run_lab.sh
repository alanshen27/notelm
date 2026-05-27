#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Create venv first: uv venv && source .venv/bin/activate"
  exit 1
fi

source .venv/bin/activate
uv pip install -q fastapi uvicorn python-multipart 2>/dev/null || true
uv pip install -q music21 2>/dev/null || true

if [[ -d ui/node_modules ]]; then
  (cd ui && npm run build)
else
  echo "Building UI (first time installs npm deps)…"
  (cd ui && npm install && npm run build)
fi

echo "Lab UI: http://localhost:8000"
cd src && python api.py
