#!/usr/bin/env bash
# Bootstrap notelm: uv, Python 3.13, project deps, training dirs, optional MAESTRO + CUDA.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FETCH_MAESTRO=false
INSTALL_LAB=false
CPU_ONLY=false
FORCE_CUDA=false

MAESTRO_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_ZIP="$ROOT/data/maestro-v3.0.0-midi.zip"
MAESTRO_DIR="$ROOT/data/maestro-v3.0.0"
PYTHON_VERSION="3.13"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [OPTIONS]

Install uv (if missing), Python 3.13, sync dependencies from pyproject.toml,
create training directories, and verify imports.

Options:
  --cuda          Install PyTorch with CUDA 12.4 wheels (use on NVIDIA GPUs)
  --cpu           Keep CPU/MPS PyTorch from PyPI (no CUDA wheel reinstall)
  --fetch-maestro Download MAESTRO v3.0.0 MIDI (~120 MB) into data/
  --lab           Also install music21 + build the React inference lab UI
  -h, --help      Show this help

After setup:
  source .venv/bin/activate
  cd src && python train.py

Long training session:
  ./scripts/train_tmux.sh
EOF
}

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

ensure_path() {
  export PATH="${HOME}/.local/bin:${PATH}"
}

install_uv() {
  if have uv; then
    log "uv already installed ($(uv --version))"
    return
  fi

  log "Installing uv..."
  if have brew; then
    brew install uv
  else
    ensure_path
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ensure_path
  fi

  if ! have uv; then
    echo "uv install finished but 'uv' is not on PATH." >&2
    echo "Add ~/.local/bin to PATH, then re-run this script." >&2
    exit 1
  fi
  log "uv installed ($(uv --version))"
}

install_python() {
  ensure_path
  log "Ensuring Python ${PYTHON_VERSION}..."
  uv python install "${PYTHON_VERSION}"
}

sync_deps() {
  ensure_path
  log "Creating venv and installing dependencies (uv sync)..."
  uv sync

  if $INSTALL_LAB; then
    log "Installing optional score deps (music21)..."
    uv sync --extra score
  fi
}

install_cuda_torch() {
  ensure_path
  log "Installing PyTorch CUDA wheels (cu124)..."
  uv pip install --upgrade "torch>=2.0" --index-url "${TORCH_CUDA_INDEX}"
}

want_cuda() {
  if $CPU_ONLY; then
    return 1
  fi
  if $FORCE_CUDA; then
    return 0
  fi
  have nvidia-smi && nvidia-smi >/dev/null 2>&1
}

fetch_maestro() {
  if [[ -d "$MAESTRO_DIR/2004" ]] && compgen -G "$MAESTRO_DIR/2004/*.midi" >/dev/null; then
    log "MAESTRO already present at $MAESTRO_DIR"
    return
  fi

  for cmd in curl unzip; do
    if ! have "$cmd"; then
      echo "Missing '$cmd' (needed for --fetch-maestro)." >&2
      exit 1
    fi
  done

  mkdir -p "$ROOT/data"
  log "Downloading MAESTRO v3.0.0 MIDI..."
  curl -fL --progress-bar -o "$MAESTRO_ZIP" "$MAESTRO_URL"

  log "Extracting MAESTRO..."
  unzip -q -o "$MAESTRO_ZIP" -d "$ROOT/data"

  # Zip root is usually maestro-v3.0.0/; normalize if nested differently.
  if [[ ! -d "$MAESTRO_DIR" ]]; then
    nested="$(find "$ROOT/data" -maxdepth 2 -type d -name 'maestro-v3.0.0' | head -1)"
    if [[ -n "$nested" && "$nested" != "$MAESTRO_DIR" ]]; then
      mv "$nested" "$MAESTRO_DIR"
    fi
  fi

  if ! compgen -G "$MAESTRO_DIR/2004/*.midi" >/dev/null; then
    warn "MAESTRO extracted but no files in $MAESTRO_DIR/2004 — check layout."
  else
    log "MAESTRO ready under $MAESTRO_DIR"
  fi
}

setup_env_file() {
  if [[ -f "$ROOT/.env" ]]; then
    return
  fi
  if [[ -f "$ROOT/.env.example" ]]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    log "Created .env from .env.example (edit for email notifications)"
  fi
}

setup_dirs() {
  mkdir -p "$ROOT/src/checkpoints" "$ROOT/logs" "$ROOT/data"
}

setup_lab_ui() {
  if ! have npm; then
    warn "--lab skipped UI build: npm not found (install Node.js)"
    return
  fi
  log "Installing UI dependencies..."
  (cd "$ROOT/ui" && npm install)
  log "UI deps installed (run ./scripts/run_lab.sh to build and serve)"
}

verify_training() {
  ensure_path
  log "Verifying training imports..."
  uv run python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

import pretty_midi
import torch
from utils.data import DATA_DIR, SEQ_LEN

print(f"  Python:  {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA:    {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f" ({torch.cuda.get_device_name(0)})")
else:
    print()

midi_count = len(list(DATA_DIR.glob("*.midi"))) if DATA_DIR.is_dir() else 0
print(f"  Data:    {DATA_DIR} ({midi_count} .midi files)")
print(f"  seq_len: {SEQ_LEN}")

if midi_count == 0:
    print("\nNo training MIDI found. Run with --fetch-maestro or add files under data/maestro-v3.0.0/")
    sys.exit(1)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) FORCE_CUDA=true; shift ;;
    --cpu) CPU_ONLY=true; shift ;;
    --fetch-maestro) FETCH_MAESTRO=true; shift ;;
    --lab) INSTALL_LAB=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

log "notelm setup (root: $ROOT)"
install_uv
install_python
sync_deps

if want_cuda; then
  install_cuda_torch
else
  log "Using PyPI PyTorch (CPU/MPS). Pass --cuda on NVIDIA machines for GPU training."
fi

$FETCH_MAESTRO && fetch_maestro
setup_env_file
setup_dirs
$INSTALL_LAB && setup_lab_ui
verify_training

cat <<EOF

Setup complete.

  source .venv/bin/activate
  cd src && python train.py

Optional:
  ./scripts/setup.sh --fetch-maestro   # download MAESTRO if you skipped it
  ./scripts/setup.sh --cuda            # force CUDA PyTorch wheels
  ./scripts/train_tmux.sh              # detached training + log
  ./scripts/run_lab.sh                 # inference lab (needs --lab or npm install)

EOF
