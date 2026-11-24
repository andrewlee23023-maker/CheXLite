# Loop notes (maintained by the orchestrator)

## What helps

## What hurts / no effect

## Untried ideas
Ranked by the iteration-7 analyst, all still untested:
1. `img_size: 320` (budget max). Many misses are subtle Cardiomegaly/Atelectasis/Effusion. Same checkpoint size.
2. `uncertain_policy` set to `ignore` for Cardiomegaly and Consolidation, instead of U-zeros.
3. `pos_weight` = clamp(neg/pos, max=5) in BCE.

## Infra blockers
- 2026-11-24, iteration 7: `.venv/Scripts/python src/train.py` and `git checkout` need permission
  approval in the headless session. The experimenter and the orchestrator were both denied, so idea 1 never
  trained. Its edit was reverted by hand and it is NOT a failed idea. This probably explains why iterations
  1-6 added no rows to log.tsv. Fix: allow the venv python and git commands in .claude/settings.
- 2026-11-24, iteration 8: STILL BLOCKED. The `cd` prefix was not the cause: the exact allowlisted string
  `.venv/Scripts/python src/train.py` (no cd, no pipes) was denied with "This command requires approval",
  and so was `git checkout -- src/`. Meanwhile `git status` works. So the allow rules in `.claude/settings.json`
  are not being applied to these commands. The likely causes are Windows path or prefix matching of
  `.venv/Scripts/python` and `--` inside `git checkout`, or a permission mode / settings source that overrides
  the project allows. The chosen idea (Cardiomegaly uncertain `zeros` -> `ignore`, a 1-line config change;
  encode_labels already supports ignore) was reverted by hand and is NOT a failed idea. Run it first once
  training works.
  Try: run `scripts/run_loop.sh` with an explicit `--allowedTools` / `--permission-mode` that covers the venv
  python, or add wildcard rules such as `Bash(.venv/Scripts/python *)` and `Bash(git checkout *)`, then check
  them with one manual headless call before spending more iterations.
- 2026-11-24, iteration 9: STILL BLOCKED, and `.claude/settings.json` is unchanged since iteration 8.
  `.venv/Scripts/python --version` was denied ("This command requires approval"), so no experiment was
  attempted and no analyst or experimenter was spawned. The orchestrator wrote `experiments/STOP` to halt
  `run_loop.sh`. This is an infra halt, NOT the stopping rule: log.tsv still has only the baseline row.
  Delete STOP after fixing permissions. Check the fix with one manual headless call first, e.g.
  `claude -p "run .venv/Scripts/python --version" --permission-mode acceptEdits`.
