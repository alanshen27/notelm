#!/bin/bash
# Instruments pretrain → piano finetune. Prefers token_cache if present.
set -euo pipefail
cd /workspace/notelm
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
source .venv/bin/activate
mkdir -p logs
exec > >(tee -a logs/canon-pipeline.log) 2>&1
echo "==> $(date -u +%FT%TZ) canon train start"

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
ngpu=${ngpu:-1}
launch=(python -u)
if [[ "$ngpu" -gt 1 ]]; then
  echo "==> torchrun nproc=$ngpu"
  launch=(torchrun --standalone --nproc_per_node="$ngpu")
fi

echo "==> stage 1: canon pretrain on all instruments (INST_* labels on)"
cd /workspace/notelm/src
"${launch[@]}" train.py --model canon --tokenizer remi --dataset instruments --epochs 40 \
  2>&1 | tee ../logs/canon-pretrain.log
python -u ../scripts/select_best_checkpoint.py \
  --log ../logs/canon-pretrain.log \
  --ckpt-dir checkpoints/canon/remi \
  --dest checkpoints/canon/remi/pretrain.pt \
  --also checkpoints/canon/remi/weights.pt

echo "==> stage 2: piano finetune (INST_piano for clavier)"
"${launch[@]}" train.py --model canon --tokenizer remi --dataset piano --epochs 40 \
  --lr 1e-4 --weights checkpoints/canon/remi/pretrain.pt --start-epoch 0 \
  2>&1 | tee ../logs/canon-finetune.log
python -u ../scripts/select_best_checkpoint.py \
  --log ../logs/canon-finetune.log \
  --ckpt-dir checkpoints/canon/remi \
  --dest checkpoints/canon/remi/canon.pt \
  --also checkpoints/canon/remi/weights.pt

echo "==> $(date -u +%FT%TZ) canon pipeline done"
echo 0 > /workspace/notelm/logs/canon-pipeline.exit
