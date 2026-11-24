#!/usr/bin/env bash
# Headless self-improving loop: one fresh Opus orchestrator process per iteration.
# Usage: scripts/run_loop.sh [iterations]      Stop early: touch experiments/STOP
set -u
cd "$(dirname "$0")/.."
N=${1:-12}
mkdir -p experiments/runs

# Preflight: project allow rules are ignored until the folder has been trusted interactively once.
if claude -p "Reply with the single word ok." --model haiku 2>&1 | grep -q "not been trusted"; then
  echo "Workspace not trusted: run 'claude' interactively in $(pwd) once and accept the trust dialog."
  exit 1
fi

rows() { wc -l < experiments/log.tsv; }
idle=0
for i in $(seq 1 "$N"); do
  if [ -f experiments/STOP ]; then echo "STOP: $(cat experiments/STOP)"; break; fi
  case "$(git branch --show-current)" in loop/*) ;; *) echo "not on a loop/* branch"; exit 1;; esac
  before=$(rows)
  echo "[$(date +%T)] iteration $i/$N"
  claude -p "Run exactly ONE iteration of the loop protocol in CLAUDE.md (iteration $i of $N), waiting for each subagent's result before the next step. Then stop. If the stopping rule is met, write the reason to experiments/STOP instead of experimenting." \
    --model opus --permission-mode acceptEdits \
    > "experiments/runs/iter_$(date +%Y%m%d_%H%M%S).log" 2>&1
  tail -1 experiments/log.tsv
  # Fail fast: an iteration that logs nothing usually means a setup problem, not a bad idea.
  if [ "$(rows)" -eq "$before" ]; then idle=$((idle + 1)); else idle=0; fi
  if [ "$idle" -ge 2 ]; then echo "2 iterations in a row logged nothing; read experiments/runs/ before continuing."; exit 1; fi
done
