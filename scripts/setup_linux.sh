#!/usr/bin/env bash
# Linux GPU server bootstrap: system tools + uv + CUDA torch + MAESTRO.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/setup.sh" --system --fetch-maestro --cuda "$@"
