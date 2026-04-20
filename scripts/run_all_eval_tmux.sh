#!/bin/bash
# Runs eval_lfr.py followed by train_supervised.py in a detached tmux session.
# Usage: bash scripts/run_all_eval_tmux.sh
#
# Attach:  tmux attach -t eval_all
# Detach:  Ctrl-b d
# Log:     tail -f logs/eval_all.log

set -euo pipefail

SESSION="eval_all"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$REPO_DIR/logs/eval_all.log"

mkdir -p "$REPO_DIR/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists — killing it first."
    tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" \
    "cd '$REPO_DIR' && poetry run python scripts/eval_lfr.py 2>&1 | tee logs/eval_lfr.log && poetry run python scripts/train_supervised.py 2>&1 | tee logs/eval_supervised.log && echo 'ALL DONE'" \
    Enter

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Evaluation pipeline started in tmux session: $SESSION"
echo ""
echo "  Attach      :  tmux attach -t $SESSION"
echo "  Detach      :  Ctrl-b then d"
echo "  LFR log     :  tail -f logs/eval_lfr.log"
echo "  Supervised  :  tail -f logs/eval_supervised.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
