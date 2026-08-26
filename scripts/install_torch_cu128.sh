#!/usr/bin/env bash
# Install PyTorch with CUDA 12.8 wheels (required for Blackwell / sm_120).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

INDEX="https://download.pytorch.org/whl/cu128"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating .venv (Python 3.13)..."
  uv venv --python 3.13
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'no nvidia-smi')"
echo "==> Installing torch>=2.7 from ${INDEX}"
uv pip uninstall -y torch 2>/dev/null || true
uv pip install --reinstall "torch>=2.7" --index-url "${INDEX}"

echo "==> Verifying CUDA..."
python -c "
import torch
print('torch', torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit('ERROR: torch.cuda.is_available() is False')
x = torch.randn(2, device='cuda')
torch.cuda.synchronize()
print('OK on', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability())
"

echo "==> Done. Train with:  cd src && python train.py"
