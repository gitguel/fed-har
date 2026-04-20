#!/bin/bash
# Starts LFR training inside a detached tmux session.
# Usage: bash scripts/run_lfr_tmux.sh
#
# Attach:  tmux attach -t lfr_training
# Detach:  Ctrl-b d
# Log:     tail -f logs/lfr_train.log

set -euo pipefail

SESSION="lfr_training"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$REPO_DIR/logs/lfr_train.log"

mkdir -p "$REPO_DIR/logs"

# Kill existing session with the same name if any
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists — killing it first."
    tmux kill-session -t "$SESSION"
fi

# Create detached session (wide terminal so tqdm bars render correctly)
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Send training command
tmux send-keys -t "$SESSION" \
    "cd '$REPO_DIR' && poetry run python scripts/train_lfr.py 2>&1 | tee '$LOG_FILE'" \
    Enter

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " LFR training started in tmux session: $SESSION"
echo ""
echo "  Attach to session :  tmux attach -t $SESSION"
echo "  Detach (stay alive): Ctrl-b  then  d"
echo "  Stream log          :  tail -f $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
