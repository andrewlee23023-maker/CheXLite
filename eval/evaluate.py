"""Frozen evaluator. The loop only ever runs `--split dev`.

    python eval/evaluate.py --split dev      # loop metric (noisy NLP labels, patient hold-out)
    python eval/evaluate.py --split valid    # radiologist labels, 200 studies (human only)
    CHEX_ALLOW_TEST=1 python eval/evaluate.py --split test   # final, human only, logged

Runs src/predict.py itself so predictions can't be hand-made, checks that every study
was predicted, aggregates images to studies (max), and reports:
  * per-label AUROC and mean AUROC (primary metric) with a bootstrap 95% CI
  * clinic referral operating point: refer if any-finding score >= t, where t gives
    target sensitivity. Reports referral rate, specificity, and missed studies per 1,000.
    For the test split, t is chosen on VALID and applied unchanged (no peeking).
The last stdout line is a single JSON object.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG = yaml.safe_load((HERE / "eval_config.yaml").read_text())
LABELS = CFG["labels"]
STUDY_RE = r"^(.*?patient\d+/study\d+)/"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()  # average ranks handle ties
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def predict(split_csv: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "preds.csv"
        subprocess.run([sys.executable, str(REPO / "src" / "predict.py"),
                        "--csv", str(split_csv), "--out", str(out)], check=True, cwd=REPO)
        return pd.read_csv(out)


def study_table(split: str) -> pd.DataFrame:
    split_csv = HERE / "splits" / f"{split}.csv"
    gt = pd.read_csv(split_csv)
    preds = predict(split_csv)

    if set(preds["Path"]) != set(gt["Path"]) or len(preds) != len(gt):
        sys.exit("EVAL ERROR: predictions must cover exactly the rows of the split.")
    p = preds[LABELS].to_numpy(float)
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        sys.exit("EVAL ERROR: predictions must be probabilities in [0, 1].")

    # Labels: blank = not mentioned -> 0; -1 = uncertain -> excluded for that label.
    y = gt[["Path"] + LABELS].copy()
    y[LABELS] = y[LABELS].fillna(0.0).replace(-1.0, np.nan)
    y["study"] = y["Path"].str.extract(STUDY_RE)[0]
    preds = preds.copy()
    preds["study"] = preds["Path"].str.extract(STUDY_RE)[0]
    if y["study"].isna().any():
        sys.exit("EVAL ERROR: could not parse study ids from Path.")

    ys = y.groupby("study")[LABELS].first()
    ps = preds.groupby("study")[LABELS].max().loc[ys.index]
    return ys.add_prefix("y_").join(ps.add_prefix("p_"))


def per_label_auroc(t: pd.DataFrame) -> dict:
    out = {}
    for lab in LABELS:
        m = t[f"y_{lab}"].notna().to_numpy()
        out[lab] = auroc(t[f"y_{lab}"].to_numpy()[m], t[f"p_{lab}"].to_numpy()[m])
    return out


def any_finding(t: pd.DataFrame):
    yv = t[[f"y_{l}" for l in LABELS]]
    pos = (yv == 1).any(axis=1)
    neg = (yv == 0).all(axis=1)  # uncertain-only studies are excluded
    keep = (pos | neg).to_numpy()
    score = t[[f"p_{l}" for l in LABELS]].max(axis=1).to_numpy()
    return pos.to_numpy()[keep].astype(int), score[keep]


def threshold_at_sensitivity(y, s, target) -> float:
    pos = np.sort(s[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def referral(y, s, t) -> dict:
    refer = s >= t
    n = len(y)
    fn = int(((y == 1) & ~refer).sum())
    return {
        "threshold": t,
        "sensitivity": float(refer[y == 1].mean()),
        "specificity": float((~refer)[y == 0].mean()),
        "referral_rate": float(refer.mean()),
        "missed_per_1000": 1000.0 * fn / n,
        "n_studies": n,
        "prevalence": float(y.mean()),
    }


def bootstrap_mean_auroc(t: pd.DataFrame, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        b = t.iloc[rng.integers(0, len(t), len(t))].reset_index(drop=True)
        vals.append(np.nanmean(list(per_label_auroc(b).values())))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def dump_errors(t: pd.DataFrame, y, s):
    """Hardest misses and false alarms, for the analyst agent (dev only)."""
    yv = t[[f"y_{l}" for l in LABELS]]
    keep = ((yv == 1).any(axis=1) | (yv == 0).all(axis=1)).to_numpy()
    d = t[keep].copy()
    d["any_true"], d["any_score"] = y, s
    fn = d[d.any_true == 1].nsmallest(50, "any_score").assign(kind="missed")
    fp = d[d.any_true == 0].nlargest(50, "any_score").assign(kind="false_alarm")
    (REPO / "experiments").mkdir(exist_ok=True)
    pd.concat([fn, fp]).to_csv(REPO / "experiments" / "last_dev_errors.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "valid", "test"], required=True)
    args = ap.parse_args()

    if args.split == "test":
        if os.environ.get("CHEX_ALLOW_TEST") != "1":
            sys.exit("Test split is human-only: set CHEX_ALLOW_TEST=1. Every run is logged.")
        with open(REPO / "experiments" / "test_runs.log", "a") as f:
            f.write(f"{dt.datetime.now().isoformat()}\n")

    t = study_table(args.split)
    pl = per_label_auroc(t)
    y, s = any_finding(t)

    if args.split == "test":
        vt = study_table("valid")
        thr = threshold_at_sensitivity(*any_finding(vt), CFG["target_sensitivity"])
        thr_source = "valid"
    else:
        thr = threshold_at_sensitivity(y, s, CFG["target_sensitivity"])
        thr_source = args.split
    if args.split == "dev":
        dump_errors(t, y, s)

    result = {
        "split": args.split,
        "metric": float(np.nanmean(list(pl.values()))),
        "metric_name": "mean_auroc_5",
        "metric_ci95": bootstrap_mean_auroc(t, CFG["bootstrap"][args.split], CFG["bootstrap"]["seed"]),
        "per_label_auroc": pl,
        "referral": {**referral(y, s, thr), "threshold_chosen_on": thr_source,
                     "target_sensitivity": CFG["target_sensitivity"]},
        "n_studies": int(len(t)),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
