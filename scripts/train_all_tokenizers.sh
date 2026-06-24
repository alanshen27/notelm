#!/usr/bin/env bash
# Train event, raw, remi, and piano_roll sequentially (separate checkpoints each).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
[[ -f .env ]] && set -a && source .env && set +a

cd src
python -u train.py --all-tokenizers "$@"
