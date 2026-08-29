#!/bin/bash
# Runs entirely on the pod. Pretrain on all instruments, then piano finetune.
# Does not start train.py until GiantMIDI-Piano, ATEPP, and PDMX are on disk.
set -euo pipefail
cd /workspace/notelm
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
mkdir -p logs data
chmod +x scripts/*.sh 2>/dev/null || true
exec > >(tee -a logs/canon-pipeline.log) 2>&1
echo "==> $(date -u +%FT%TZ) canon pipeline start (instruments pretrain → piano finetune)"

wait_setup() {
  echo "==> waiting for setup to finish fetching POP909 / extra / MAESTRO"
  while tmux has-session -t notelm-setup 2>/dev/null; do
    tail -n 2 logs/setup.log 2>/dev/null || true
    sleep 20
  done
  if [[ -f logs/setup.exit ]]; then
    code=$(cat logs/setup.exit)
    if [[ "$code" != "0" ]]; then
      echo "setup failed exit=$code"
      exit 1
    fi
    echo "==> setup exit 0"
    return
  fi
  echo "==> running setup.sh"
  ./scripts/setup.sh --system --fetch-pop909 --fetch-extra --fetch-maestro --cuda
  echo $? > logs/setup.exit
  [[ "$(cat logs/setup.exit)" == "0" ]]
}

require_base_midi() {
  echo "==> checking base MIDI corpora are on disk"
  source .venv/bin/activate
  python - <<'PY'
import sys
sys.path.insert(0, "src")
from utils.data import dataset_files

need = {
    "pop909": 800,
    "pop1k7": 1500,
    "emopia": 800,
    "adl": 8000,
    "asap": 900,
    "maestro_full": 1000,
}
bad = []
for name, min_n in need.items():
    n = len(dataset_files(name))
    print(f"  {name}: {n:,} (need >= {min_n:,})", flush=True)
    if n < min_n:
        bad.append(f"{name}={n}")
if bad:
    raise SystemExit("base MIDI not ready: " + ", ".join(bad))
print("  base MIDI ok", flush=True)
PY
}

require_ccby() {
  echo "==> checking GiantMIDI-Piano / ATEPP / PDMX"
  python - <<'PY'
import sys
sys.path.insert(0, "src")
from utils.data import dataset_files

need = {"giantmidi": 5000, "atepp": 5000, "pdmx": 8000}
bad = []
for name, min_n in need.items():
    n = len(dataset_files(name))
    print(f"  {name}: {n:,} (need >= {min_n:,})", flush=True)
    if n < min_n:
        bad.append(f"{name}={n}")
if bad:
    raise SystemExit("CC-BY MIDI not ready: " + ", ".join(bad))
print("  CC-BY MIDI ok", flush=True)
PY
}

wait_setup
require_base_midi

source .venv/bin/activate
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
echo "==> installing gdown / huggingface_hub for CC-BY fetches"
if command -v uv >/dev/null 2>&1; then
  uv pip install gdown huggingface_hub
else
  python -m pip install gdown huggingface_hub
fi

echo "==> fetching GiantMIDI-Piano, ATEPP, PDMX (will not train until this finishes)"
python -u scripts/fetch_datasets.py ccby
require_ccby

echo "==> stage 1: canon pretrain on all instruments (INST_* labels on)"
cd /workspace/notelm/src
python -u train.py --model canon --tokenizer remi --dataset instruments --epochs 40 \
  2>&1 | tee ../logs/canon-pretrain.log
python -u ../scripts/select_best_checkpoint.py \
  --log ../logs/canon-pretrain.log \
  --ckpt-dir checkpoints/canon/remi \
  --dest checkpoints/canon/remi/pretrain.pt \
  --also checkpoints/canon/remi/weights.pt

echo "==> stage 2: piano finetune (INST_piano for clavier)"
python -u train.py --model canon --tokenizer remi --dataset piano --epochs 40 \
  --lr 1e-4 --weights checkpoints/canon/remi/pretrain.pt --start-epoch 0 \
  2>&1 | tee ../logs/canon-finetune.log
python -u ../scripts/select_best_checkpoint.py \
  --log ../logs/canon-finetune.log \
  --ckpt-dir checkpoints/canon/remi \
  --dest checkpoints/canon/remi/canon.pt \
  --also checkpoints/canon/remi/weights.pt

echo "==> $(date -u +%FT%TZ) canon pipeline done"
