#!/usr/bin/env bash
# Run training inside the project venv (avoids system python missing deps).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ ! -d .venv ]]; then
  echo "No .venv — run: ./scripts/setup.sh --cuda --fetch-maestro" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import tqdm, pretty_midi, torch" 2>/dev/null; then
  echo "==> Missing packages in .venv — running uv sync ..."
  uv sync
fi

cd src
echo "Using $(which python)"
exec python -u train.py "$@"
