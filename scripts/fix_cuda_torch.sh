#!/usr/bin/env bash
# Reinstall PyTorch cu124 and remove orphaned CUDA 13 NVIDIA wheels from a bad PyPI install.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ ! -d .venv ]]; then
  echo "No .venv — run ./scripts/setup.sh --cuda first." >&2
  exit 1
fi

# Leftovers when PyPI torch (CUDA 13) was installed before cu124 wheels.
CUDA13_ORPHANS=(
  cuda-bindings
  cuda-pathfinder
  cuda-toolkit
  nvidia-cuda-cupti
  nvidia-cuda-nvrtc
  nvidia-cuda-runtime
  nvidia-cudnn-cu13
  nvidia-cufft
  nvidia-cufile
  nvidia-curand
  nvidia-cusolver
  nvidia-cusparse
  nvidia-cusparselt-cu13
  nvidia-nccl-cu13
  nvidia-nvjitlink
  nvidia-nvshmem-cu13
  nvidia-nvtx
)

echo "==> Reinstalling PyTorch (CUDA 12.4 index)..."
uv pip uninstall -y torch 2>/dev/null || true
uv pip install --reinstall "torch>=2.6" --index-url "https://download.pytorch.org/whl/cu124"

echo "==> Removing orphaned CUDA 13 NVIDIA packages (if present)..."
uv pip uninstall -y "${CUDA13_ORPHANS[@]}" 2>/dev/null || true

echo "==> Removing optional torch extras not used by notelm..."
uv pip uninstall -y torchaudio torchvision 2>/dev/null || true

echo "==> Checking CUDA..."
uv run python -c "
import torch
ok = torch.cuda.is_available()
print('torch', torch.__version__, '| cuda', ok, end='')
if ok:
    print(' |', torch.cuda.get_device_name(0))
else:
    print()
    raise SystemExit(
        'CUDA still unavailable — update the NVIDIA driver or recreate .venv with ./scripts/setup.sh --cuda'
    )
"

echo "Done. Keep only torch + *-cu12 nvidia packages in pip list."
