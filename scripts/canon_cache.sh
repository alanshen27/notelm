#!/bin/bash
# Full-corpus CPU encode → token_cache (instruments then piano). No GPU.
set -euo pipefail
cd /workspace/notelm
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
source .venv/bin/activate
mkdir -p logs
exec > >(tee -a logs/canon-cache.log) 2>&1
echo "==> $(date -u +%FT%TZ) canon cache start (process-pool encode, full MIDI)"
cd /workspace/notelm/src
python -u train.py --model canon --tokenizer remi --dataset instruments --cache-only
python -u train.py --model canon --tokenizer remi --dataset piano --cache-only
echo "==> $(date -u +%FT%TZ) canon cache done"
echo 0 > /workspace/notelm/logs/canon-cache.exit
