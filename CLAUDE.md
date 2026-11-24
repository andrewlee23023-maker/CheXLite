# ChexLite: offline chest X-ray second look for clinics without a radiologist

## Product framing (keep every claim consistent with this)
- User: a clinician (GP, clinical officer, nurse) at a clinic with no radiologist.
- Job: flag 5 findings (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion),
  highlight regions to check, and recommend **refer to a radiologist** or not.
- It is a triage aid, not a diagnosis. A radiologist can always be consulted later.
- The harm that matters most is a missed finding that isn't referred, so we report sensitivity
  at a fixed operating point and missed studies per 1,000, not just AUROC.
- Radiologists are a **reference point**, never a claimed target. Never write "radiologist-level".
- The clinician note comes from a fine-tuned Gemma 3 270M that sees ONLY the classifier's structured
  output (probabilities, thresholds, refer decision, heatmap regions). Every note is checked by a
  deterministic faithfulness verifier in the app. If it fails, the app shows the templated note instead.
  Report the LLM's faithfulness rate in the evaluation; never claim the LLM adds diagnostic information.

## Hard rules (for you and every subagent)
1. Never edit anything under `eval/`. Never run `--split test` or set `CHEX_ALLOW_TEST`.
   Never run `--split valid` during the loop either; it is reserved for the human.
2. Never change `seed` in `src/config.yaml` or the budget in `eval/budget.yaml`.
3. Never special-case image paths, patients or studies, drop samples, or read labels at predict time.
4. Loop work happens only on a `loop/*` branch. Never touch `main`, never push.
5. One idea per experiment, one `exp:` commit per experiment.

## Loop protocol (you are the orchestrator; subagents run on Haiku)
Run iterations **sequentially**: there is one 8 GB GPU.

Each iteration:
1. Read `experiments/log.tsv` and `experiments/NOTES.md`.
2. Spawn **analyst** → 3 ranked hypotheses. Pick one, or replace it with a better idea of your own.
   Prefer ideas that keep the model phone-sized (fp32 checkpoint < ~25 MB).
3. Spawn **experimenter** with exactly that one hypothesis. It edits `src/`, trains, and reports.
4. If it reports a crash after two fixes: `git checkout -- src/`, log the idea as failed in NOTES.md, move on.
5. `git add src && git commit -m "exp: <short idea>"`
6. Spawn **auditor** on the commit. On FAIL: `git reset --hard HEAD~1`, note the reason in NOTES.md, move on.
7. `.venv/Scripts/python scripts/ratchet.py "<short idea>"`. It keeps or reverts the commit and logs the result. Don't second-guess it.
8. When the number of rows in `experiments/log.tsv` is a multiple of 5, update `experiments/NOTES.md` (what helps, what hurts, what's left to try) and commit it.

Stop when any of these holds: the iteration budget given in the prompt is used up, 6 iterations in a
row haven't improved, or an improvement falls inside the previous best's CI three times in a row
(that's noise, not progress). To stop, write the reason to `experiments/STOP`.
The headless runner is `scripts/run_loop.sh N` (one fresh process per iteration).

## Useful commands
Always use the venv interpreter `.venv/Scripts/python` and a Bash timeout of 600000 for train, eval and ratchet.
- Train (loop budget): `.venv/Scripts/python src/train.py`
- Dev eval: `.venv/Scripts/python eval/evaluate.py --split dev` (last line is JSON; `metric` = mean AUROC)
- Hardest dev errors: `experiments/last_dev_errors.csv`
