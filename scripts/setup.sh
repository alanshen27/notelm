#!/usr/bin/env bash
# Bootstrap notelm: uv, Python 3.13, project deps, training dirs, optional MAESTRO + CUDA.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FETCH_MAESTRO=false
FETCH_POP909=false
FETCH_EXTRA=false
FETCH_CCBY=false
FETCH_GIGAMIDI=false
GIGAMIDI_MAX="${GIGAMIDI_MAX:-50000}"
EXTRA_NAMES=()
INSTALL_LAB=false
INSTALL_SYSTEM=false
CPU_ONLY=false
FORCE_CUDA=false

MAESTRO_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_ZIP="$ROOT/data/maestro-v3.0.0-midi.zip"
MAESTRO_DIR="$ROOT/data/maestro-v3.0.0"
POP909_URL="https://raw.githubusercontent.com/music-x-lab/POP909-Dataset/master/POP909.zip"
POP909_ZIP="$ROOT/data/POP909.zip"
POP909_DIR="$ROOT/data/POP909"
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
  --fetch-pop909  Download POP909 pop-song MIDI (~23 MB) into data/
  --fetch-extra   EMOPIA + Pop1K7 + ADL piano + ASAP (needs git, ~hours of MIDI)
  --fetch-ccby    GiantMIDI-Piano + ATEPP + PDMX (CC BY, for canon)
  --fetch-gigamidi Piano-ish GigaMIDI subset (needs HF_TOKEN + datasets pkg; NC)
  --gigamidi-max N Cap GigaMIDI files (default 50000)
  --fetch-emopia|--fetch-pop1k7|--fetch-adl|--fetch-asap|--fetch-giantmidi|--fetch-atepp|--fetch-pdmx
                  Individual extra corpora (see scripts/fetch_datasets.py)
  --system        Install OS packages (Linux: tmux, curl, unzip via apt/dnf/apk)
  --lab           Also install music21 + npm UI deps
  -h, --help      Show this help

Linux GPU server (typical):
  ./scripts/setup.sh --system --fetch-pop909 --cuda

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

configure_uv() {
  ensure_path
  # Avoid slow hardlink-then-copy when cache and .venv are on different filesystems.
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
}

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
  configure_uv
  local -a uv_args=()
  if want_cuda; then
    # PyPI Linux torch pulls many NVIDIA libs; install one CUDA wheel instead.
    uv_args=(--no-install-package torch)
    log "Installing dependencies (CUDA PyTorch follows separately)..."
    log "PyTorch CUDA wheel is ~2–3 GB — 5–15 minutes on slow or network storage is normal."
  else
    log "Installing dependencies (uv sync)..."
    log "On Linux, the PyTorch wheel is large; first install may take several minutes."
  fi

  uv sync "${uv_args[@]}"

  if want_cuda; then
    install_cuda_torch
  fi

  if $INSTALL_LAB; then
    log "Installing optional score deps (music21)..."
    uv sync --extra score "${uv_args[@]}"
  fi
}

pick_torch_cuda_index() {
  # Blackwell (sm_120, RTX 50 / RTX PRO) needs PyTorch built with CUDA 12.8+.
  if have nvidia-smi; then
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

install_cuda_torch() {
  configure_uv
  TORCH_CUDA_INDEX="$(pick_torch_cuda_index)"
  # cu124 wheels top out at 2.6.x; 2.7+ lives on cu128 (Blackwell).
  if [[ "${TORCH_CUDA_INDEX}" == *cu128* ]]; then
    torch_spec="torch>=2.7"
  else
    torch_spec="torch==2.6.0"
  fi
  log "Installing ${torch_spec} from ${TORCH_CUDA_INDEX} (auto-selected for this GPU)..."
  uv pip uninstall -y torch 2>/dev/null || true
  uv pip install --reinstall "${torch_spec}" --index-url "${TORCH_CUDA_INDEX}"
  log "CUDA smoke test..."
  uv run python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    torch.randn(2, device='cuda')
    print('GPU:', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability())
"
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

install_system_packages() {
  local pkgs=(tmux curl unzip ca-certificates git)
  local missing=()
  for cmd in tmux curl unzip git; do
    have "$cmd" || missing+=("$cmd")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    log "System tools already present (tmux, curl, unzip)"
    return
  fi

  log "Installing system packages: ${pkgs[*]} ..."

  run_as_root() {
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
      "$@"
    elif have sudo; then
      sudo "$@"
    else
      warn "Need root or sudo to install: ${pkgs[*]}"
      warn "On Debian/Ubuntu: sudo apt-get install -y ${pkgs[*]}"
      return 1
    fi
  }

  if have apt-get; then
    run_as_root apt-get update -qq
    run_as_root apt-get install -y "${pkgs[@]}"
  elif have dnf; then
    run_as_root dnf install -y tmux curl unzip ca-certificates
  elif have yum; then
    run_as_root yum install -y tmux curl unzip ca-certificates
  elif have apk; then
    run_as_root apk add --no-cache tmux curl unzip ca-certificates
  else
    warn "Unknown package manager — install manually: ${pkgs[*]}"
    return 1
  fi

  log "System packages installed"
}

fetch_maestro() {
  if [[ -d "$MAESTRO_DIR" ]]; then
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

fetch_pop909() {
  if [[ -d "$POP909_DIR" ]]; then
    log "POP909 already present at $POP909_DIR"
    return
  fi

  for cmd in curl unzip; do
    if ! have "$cmd"; then
      echo "Missing '$cmd' (needed for --fetch-pop909)." >&2
      exit 1
    fi
  done

  mkdir -p "$ROOT/data"
  log "Downloading POP909 (pop-song MIDI, ~23 MB)..."
  curl -fL --progress-bar -o "$POP909_ZIP" "$POP909_URL"

  log "Extracting POP909..."
  unzip -q -o "$POP909_ZIP" -d "$ROOT/data"

  # Zip root is usually POP909/; normalize if nested differently.
  if [[ ! -d "$POP909_DIR" ]]; then
    nested="$(find "$ROOT/data" -maxdepth 3 -type d -name 'POP909' | head -1)"
    if [[ -n "$nested" && "$nested" != "$POP909_DIR" ]]; then
      mv "$nested" "$POP909_DIR"
    fi
  fi

  if ! compgen -G "$POP909_DIR/001/001.mid" >/dev/null; then
    warn "POP909 extracted but $POP909_DIR/001/001.mid missing — check layout."
  else
    log "POP909 ready under $POP909_DIR ($(find "$POP909_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') songs)"
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
  for tok in event remi; do
    mkdir -p "$ROOT/src/checkpoints/transformer/$tok"
  done
  mkdir -p "$ROOT/src/checkpoints/canon/remi"
}

setup_lab_ui() {
  if ! have npm; then
    warn "--lab skipped UI build: npm not found (install Node.js)"
    return
  fi
  log "Installing UI dependencies (npm registry: npmmirror)..."
  export ELECTRON_MIRROR="${ELECTRON_MIRROR:-https://npmmirror.com/mirrors/electron/}"
  export ELECTRON_BUILDER_BINARIES_MIRROR="${ELECTRON_BUILDER_BINARIES_MIRROR:-https://npmmirror.com/mirrors/electron-builder-binaries/}"
  (cd "$ROOT" && npm install)
  log "JS workspaces installed (run ./scripts/run_lab.sh to build and serve)"
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
from utils.data import ATOMIC_DATASETS, dataset_dir, dataset_files, seq_len_for

print(f"  Python:  {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA:    {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f" ({torch.cuda.get_device_name(0)})")
    try:
        torch.randn(2, device="cuda")
    except RuntimeError as e:
        if "no kernel image" in str(e):
            print(
                "\nERROR: GPU detected but PyTorch has no kernels for this card "
                "(Blackwell / RTX 50 needs cu128).\n"
                "Fix:  ./scripts/fix_cuda_torch.sh\n"
                "  or:  uv pip install --reinstall 'torch>=2.7' "
                "--index-url https://download.pytorch.org/whl/cu128\n"
            )
            sys.exit(1)
        raise
else:
    print()

import shutil

if not torch.cuda.is_available() and shutil.which("nvidia-smi"):
    print(
        "\nERROR: NVIDIA GPU detected but PyTorch CUDA is not working.\n"
        "Linux PyPI torch often needs CUDA 13 drivers; this project uses cu124 instead.\n"
        "Fix:  ./scripts/setup.sh --cuda\n"
        "  or:  uv pip install --reinstall 'torch==2.12.0' "
        "--index-url https://download.pytorch.org/whl/cu124\n"
    )
    sys.exit(1)

total = 0
for name in ATOMIC_DATASETS:
    count = len(dataset_files(name))
    total += count
    print(f"  Data:    {name} -> {dataset_dir(name)} ({count} files)")
print(f"  seq_len: {seq_len_for('event')} (event tokenizer)")

if total == 0:
    print("\nNo training MIDI found. Run with --fetch-pop909 (or --fetch-extra).")
    sys.exit(1)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) FORCE_CUDA=true; shift ;;
    --cpu) CPU_ONLY=true; shift ;;
    --fetch-maestro) FETCH_MAESTRO=true; shift ;;
    --fetch-pop909) FETCH_POP909=true; shift ;;
    --fetch-extra) FETCH_EXTRA=true; shift ;;
    --fetch-ccby) FETCH_CCBY=true; shift ;;
    --fetch-gigamidi) FETCH_GIGAMIDI=true; shift ;;
    --gigamidi-max)
      GIGAMIDI_MAX="$2"
      shift 2
      ;;
    --fetch-emopia|--fetch-pop1k7|--fetch-adl|--fetch-asap|--fetch-giantmidi|--fetch-atepp|--fetch-pdmx)
      EXTRA_NAMES+=("${1#--fetch-}"); shift ;;
    --system) INSTALL_SYSTEM=true; shift ;;
    --lab) INSTALL_LAB=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

log "notelm setup (root: $ROOT)"
configure_uv
$INSTALL_SYSTEM && install_system_packages
install_uv
install_python
sync_deps

if ! want_cuda; then
  log "Using PyPI PyTorch (CPU/MPS). Pass --cuda on NVIDIA machines for GPU training."
fi

$FETCH_MAESTRO && fetch_maestro
$FETCH_POP909 && fetch_pop909
if $FETCH_EXTRA; then
  python3 "$ROOT/scripts/fetch_datasets.py" extra
fi
if $FETCH_CCBY; then
  python3 "$ROOT/scripts/fetch_datasets.py" ccby
fi
if ((${#EXTRA_NAMES[@]})); then
  python3 "$ROOT/scripts/fetch_datasets.py" "${EXTRA_NAMES[@]}"
fi
if $FETCH_GIGAMIDI; then
  log "Installing Hugging Face datasets extra for GigaMIDI..."
  uv pip install datasets huggingface_hub
  python3 "$ROOT/scripts/fetch_datasets.py" gigamidi --max-files "$GIGAMIDI_MAX"
fi
setup_env_file
setup_dirs
$INSTALL_LAB && setup_lab_ui
verify_training

cat <<EOF

Setup complete.

  source .venv/bin/activate
  cd src && python train.py

Optional:
  ./scripts/setup.sh --fetch-pop909    # download POP909 (pop training data)
  ./scripts/setup.sh --fetch-extra     # EMOPIA + Pop1K7 + ADL + ASAP
  ./scripts/setup.sh --fetch-ccby      # GiantMIDI-Piano + ATEPP + PDMX
  ./scripts/setup.sh --fetch-maestro   # classical piano pretrain
  ./scripts/setup.sh --fetch-gigamidi  # 50k piano-ish GigaMIDI (needs HF_TOKEN, NC)
  ./scripts/setup.sh --cuda            # force CUDA PyTorch wheels
  ./scripts/train_tmux.sh              # detached training + log
  ./scripts/run_lab.sh                 # build UI + serve (needs --lab or npm install)

EOF
