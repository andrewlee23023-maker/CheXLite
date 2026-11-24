"""Keep-or-revert gate for one loop experiment.

    python scripts/ratchet.py --baseline "DenseNet-free MobileNetV3 baseline"
    python scripts/ratchet.py "idea description"      # HEAD must be an `exp:` commit on loop/*

1. Refuses if eval/ differs from tag `eval-frozen` or split hashes != MANIFEST (tamper).
2. Runs the frozen dev evaluation on checkpoints/model.pt.
3. Keeps the commit if dev mean AUROC beats the best kept score by MARGIN, saving
   the checkpoint as best.pt. Otherwise `git reset --hard HEAD~1` and restores best.pt.
4. Appends a row to experiments/log.tsv either way. Failures are evidence too.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "experiments" / "log.tsv"
CK = REPO / "checkpoints"
MARGIN = 0.002
HEADER = ["commit", "idea", "dev", "ci_lo", "ci_hi", "referral_rate", "kept", "time"]


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)


def eval_untouched() -> str | None:
    if git("rev-parse", "-q", "--verify", "eval-frozen").returncode != 0:
        return "tag eval-frozen does not exist; freeze eval first"
    if git("diff", "--quiet", "eval-frozen", "--", "eval/").returncode != 0:
        return "eval/ differs from eval-frozen"
    manifest = json.loads((REPO / "eval" / "MANIFEST.json").read_text())
    for f, h in manifest["sha256"].items():
        if hashlib.sha256((REPO / "eval" / "splits" / f).read_bytes()).hexdigest() != h:
            return f"split {f} hash mismatch"
    return None


def best_kept() -> float:
    if not LOG.exists():
        return float("-inf")
    rows = [r for r in csv.DictReader(LOG.open(), delimiter="\t") if r["kept"] == "1"]
    return max((float(r["dev"]) for r in rows), default=float("-inf"))


def log(row: dict):
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, HEADER, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)


def revert():
    saved = LOG.read_text() if LOG.exists() else None  # reset --hard must not eat the log
    git("reset", "--hard", "HEAD~1")
    if saved is not None:
        LOG.write_text(saved)
    if (CK / "best.pt").exists():
        shutil.copy2(CK / "best.pt", CK / "model.pt")


def commit_log(summary: str):
    git("add", str(LOG))
    git("commit", "-m", f"log: {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()

    sha = git("rev-parse", "--short", "HEAD").stdout.strip()
    now = dt.datetime.now().isoformat(timespec="seconds")
    if not a.baseline:
        branch = git("branch", "--show-current").stdout.strip()
        msg = git("log", "-1", "--format=%s").stdout.strip()
        if not branch.startswith("loop/") or not msg.startswith("exp:"):
            sys.exit(f"Refusing: need an `exp:` commit on a loop/* branch (got {branch!r}, {msg!r}).")

    problem = eval_untouched()
    if problem:
        if not a.baseline:
            revert()
        log({"commit": sha, "idea": a.idea, "dev": "nan", "ci_lo": "", "ci_hi": "",
             "referral_rate": "", "kept": "0", "time": now})
        commit_log(f"{a.idea} -> REJECTED ({problem})")
        sys.exit(f"TAMPER/SETUP: {problem}. Experiment reverted.")

    r = subprocess.run([sys.executable, "eval/evaluate.py", "--split", "dev"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        if not a.baseline:
            revert()
        log({"commit": sha, "idea": a.idea + " [EVAL FAILED]", "dev": "nan", "ci_lo": "", "ci_hi": "",
             "referral_rate": "", "kept": "0", "time": now})
        commit_log(f"{a.idea} -> EVAL FAILED")
        sys.exit("Evaluation failed, reverted:\n" + r.stderr[-2000:])

    res = json.loads(r.stdout.strip().splitlines()[-1])
    score, best = res["metric"], best_kept()
    kept = a.baseline or score > best + MARGIN
    if kept:
        shutil.copy2(CK / "model.pt", CK / "best.pt")
    else:
        revert()
    log({"commit": sha, "idea": a.idea, "dev": f"{score:.4f}", "ci_lo": f"{res['metric_ci95'][0]:.4f}",
         "ci_hi": f"{res['metric_ci95'][1]:.4f}", "referral_rate": f"{res['referral']['referral_rate']:.3f}",
         "kept": str(int(kept)), "time": now})
    commit_log(f"{a.idea} -> dev {score:.4f} ({'KEPT' if kept else 'reverted'})")
    print(json.dumps({"score": score, "previous_best": best, "kept": kept, "ci95": res["metric_ci95"],
                      "per_label": res["per_label_auroc"], "referral": res["referral"]}))


if __name__ == "__main__":
    main()
