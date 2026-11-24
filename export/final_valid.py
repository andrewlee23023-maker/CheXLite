"""One-time final evaluation on CheXpert VALID (3-radiologist majority labels), out of sample.

Thresholds are the ones the app ships (app/model/meta.json), chosen on the dev split BEFORE valid was
ever scored. Valid is predicted once and reported with:
  * mean / per-label AUROC with bootstrap 95% CI (frozen eval code: eval/evaluate.py helpers)
  * the shipped referral threshold applied unchanged: sensitivity, specificity, referral rate,
    missed studies per 1,000, with Wilson CIs
  * per-finding flag sensitivity/specificity at the shipped flag thresholds
  * for comparison only: the in-sample valid threshold at 95% sensitivity (what evaluate.py reports)
Every run appends a line to experiments/valid_runs.log in the main checkout.

    python export/final_valid.py
"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO.parent / "chexlite"
sys.path.insert(0, str(MAIN / "eval"))
import evaluate as E  # noqa: E402  (frozen evaluation helpers, unmodified)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def op_point(y, s, t):
    pred = s >= t
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    n = len(y)
    return {"threshold": round(float(t), 4), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": round(tp / max(1, tp + fn), 4), "sensitivity_ci95": wilson(tp, tp + fn),
            "specificity": round(tn / max(1, tn + fp), 4), "specificity_ci95": wilson(tn, tn + fp),
            "referral_rate": round((tp + fp) / n, 4), "missed_per_1000": round(1000 * fn / n, 1)}


def main():
    meta = json.loads((REPO / "app" / "model" / "meta.json").read_text())
    with open(MAIN / "experiments" / "valid_runs.log", "a") as f:
        f.write(f"{dt.datetime.now().isoformat()} final_valid.py arch={meta['arch']} thresholds={meta.get('threshold_source')}\n")

    t = E.study_table("valid")
    pl = E.per_label_auroc(t)
    y, s = E.any_finding(t)
    keep = ~np.isnan(y)
    y, s = y[keep].astype(int), s[keep]

    flags = {}
    for lab in E.LABELS:
        yl, pl_ = t[f"y_{lab}"].to_numpy(), t[f"p_{lab}"].to_numpy()
        m = ~np.isnan(yl)
        flags[lab] = {"n_pos": int(yl[m].sum()), **op_point(yl[m].astype(int), pl_[m], meta["flag_thresholds"][lab])}

    res = {
        "split": "valid", "n_studies": int(len(t)), "n_any_finding_scored": int(keep.sum()),
        "prevalence_any": round(float(y.mean()), 4),
        "mean_auroc_5": round(float(np.nanmean(list(pl.values()))), 4),
        "mean_auroc_ci95": [round(x, 4) for x in E.bootstrap_mean_auroc(t, E.CFG["bootstrap"]["valid"], E.CFG["bootstrap"]["seed"])],
        "per_label_auroc": {k: round(v, 4) for k, v in pl.items()},
        "referral_shipped_threshold_from_dev": op_point(y, s, meta["refer_threshold"]),
        "referral_in_sample_valid_95": op_point(y, s, E.threshold_at_sensitivity(y, s, E.CFG["target_sensitivity"])),
        "flags_shipped_thresholds_from_dev": flags,
        "model": {"arch": meta["arch"], "onnx_mb": round(meta["onnx_bytes"] / 1e6, 1)},
    }
    out = REPO / "experiments_results" / "final_valid.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
