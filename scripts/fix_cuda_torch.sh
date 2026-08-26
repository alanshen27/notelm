#!/usr/bin/env bash
# Reinstall PyTorch with the right CUDA wheels for this GPU (cu128 for Blackwell).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ ! -d .venv ]]; then
  echo "No .venv — run ./scripts/setup.sh --cuda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pick_index() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local name cap major
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if echo "$name" | grep -qiE 'blackwell|rtx pro 4|rtx 50|5080|5090|5070|5050'; then
      echo "https://download.pytorch.org/whl/cu128"
      return
    fi
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
    major="${cap%%.*}"
    if [[ -n "$major" && "$major" -ge 12 ]]; then
      echo "https://download.pytorch.org/whl/cu128"
      return
    fi
  fi
  echo "https://download.pytorch.org/whl/cu124"
}

INDEX=$(pick_index)
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "==> Reinstalling torch>=2.7 from ${INDEX}"
uv pip uninstall -y torch 2>/dev/null || true
uv pip install --reinstall "torch>=2.7" --index-url "${INDEX}"

echo "==> CUDA smoke test..."
python -c "
import torch
print('torch', torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA not available')
x = torch.randn(2, device='cuda')
print('ok on', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability())
"
