#!/usr/bin/env bash
# Headless self-improving loop: one fresh Opus orchestrator process per iteration.
# Usage: scripts/run_loop.sh [iterations]      Stop early: touch experiments/STOP
set -u
cd "$(dirname "$0")/.."
N=${1:-12}
mkdir -p experiments/runs
for i in $(seq 1 "$N"); do
  if [ -f experiments/STOP ]; then echo "STOP: $(cat experiments/STOP)"; break; fi
  case "$(git branch --show-current)" in loop/*) ;; *) echo "not on a loop/* branch"; exit 1;; esac
  echo "[$(date +%T)] iteration $i/$N"
  claude -p "Run exactly ONE iteration of the loop protocol in CLAUDE.md (iteration $i of $N), waiting for each subagent's result before the next step. Then stop. If the stopping rule is met, write the reason to experiments/STOP instead of experimenting." \
    --model opus --permission-mode acceptEdits \
    > "experiments/runs/iter_$(date +%Y%m%d_%H%M%S).log" 2>&1
  tail -1 experiments/log.tsv
done
