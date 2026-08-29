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

if [[ ! -d node_modules ]]; then
  echo "Installing JS workspaces…"
  npm install
fi
npm run build

echo "Site:     http://localhost:8000/"
echo "Clavier:  http://localhost:8000/app/"
echo "Dev:      http://localhost:8000/dev/?ckpt=etude"
cd src && python api.py
