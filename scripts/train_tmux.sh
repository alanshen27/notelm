#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="notelm-train"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/train-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/checkpoints"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed."
  echo "Install with: brew install tmux"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running."
  echo "  attach:  tmux attach -t $SESSION"
  echo "  kill:    tmux kill-session -t $SESSION"
  exit 1
fi

CMD="cd '$ROOT' && source .venv/bin/activate && cd src && python -u train.py 2>&1 | tee -a '$LOG_FILE'"

tmux new-session -d -s "$SESSION" "$CMD"

echo "Started training in tmux session '$SESSION'"
echo "  log:     $LOG_FILE"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b then d"
echo "  kill:    tmux kill-session -t $SESSION"
