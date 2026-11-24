"""Choose the app's operating thresholds and write them to app/model/meta.json.

  * referral threshold: "any finding" score (max of 5 probs) at the target sensitivity
  * per-finding flag thresholds: each finding at --flag-target sensitivity (lower than the referral
    target, because flags only guide where to look; at 95% per finding nearly everything is flagged)
Per the eval protocol, final thresholds come from the radiologist-labelled VALID split, run by a
human. `--split dev` gives provisional thresholds for development.

    python export/calibrate.py --split dev --ckpt ../chexlite/checkpoints/best.pt
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAIN = REPO.parent / "chexlite"  # main checkout holds data, splits and venv
STUDY_RE = r"^(.*?patient\d+/study\d+)/"


def thr_at_sens(y, s, target):
    pos = np.sort(s[y == 1])
    return float(pos[min(int(np.floor((1 - target) * len(pos))), len(pos) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "valid"], required=True)
    ap.add_argument("--ckpt", default=str(MAIN / "checkpoints" / "best.pt"))
    ap.add_argument("--target", type=float, default=0.95, help="referral sensitivity")
    ap.add_argument("--flag-target", type=float, default=0.80, help="per-finding flag sensitivity")
    a = ap.parse_args()

    labels = yaml.safe_load((MAIN / "eval" / "eval_config.yaml").read_text())["labels"]
    split_csv = MAIN / "eval" / "splits" / f"{a.split}.csv"
    gt = pd.read_csv(split_csv)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "p.csv"
        subprocess.run([sys.executable, str(MAIN / "src" / "predict.py"), "--csv", str(split_csv),
                        "--out", str(out), "--ckpt", a.ckpt], check=True, cwd=MAIN)
        pr = pd.read_csv(out)

    gt["study"] = gt["Path"].str.extract(STUDY_RE)[0]
    pr["study"] = pr["Path"].str.extract(STUDY_RE)[0]
    y = gt.groupby("study")[labels].first().fillna(0.0).replace(-1.0, np.nan)
    p = pr.groupby("study")[labels].max().loc[y.index]

    per = {}
    for l in labels:
        m = y[l].notna().to_numpy()
        per[l] = thr_at_sens(y[l].to_numpy()[m], p[l].to_numpy()[m], a.flag_target)
    pos, neg = (y == 1).any(axis=1), (y == 0).all(axis=1)
    keep = (pos | neg).to_numpy()
    refer = thr_at_sens(pos.to_numpy()[keep].astype(int), p.max(axis=1).to_numpy()[keep], a.target)

    meta_path = REPO / "app" / "model" / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"flag_thresholds": per, "refer_threshold": refer, "target_sensitivity": a.target, "flag_target_sensitivity": a.flag_target,
                 "threshold_source": a.split if a.split == "valid" else "dev (provisional)"})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({"refer_threshold": refer, "flag_thresholds": per, "source": a.split}, indent=2))


if __name__ == "__main__":
    main()
