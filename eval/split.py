"""Run ONCE, by a human, before `git tag eval-frozen`.

Builds a patient-level train/dev split of CheXpert train.csv (no patient appears in
both), copies the radiologist-labelled valid set, and the CheXlocalize test set if
present. Writes SHA-256 hashes to eval/MANIFEST.json so later tampering is detected.

Split CSVs contain CheXpert labels, which the data use agreement doesn't let you
redistribute, so eval/splits/ is gitignored. Only the manifest is committed.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "eval_config.yaml").read_text())


def data_root() -> Path:
    return Path(os.environ.get("CHEXPERT_ROOT", CFG["data_root"]))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = data_root()
    src = root / CFG["chexpert_dir"]
    train = pd.read_csv(src / "train.csv")
    valid = pd.read_csv(src / "valid.csv")

    pid = train["Path"].str.extract(r"(patient\d+)")[0]
    patients = np.array(sorted(pid.unique()))
    rng = np.random.default_rng(CFG["split_seed"])
    rng.shuffle(patients)
    dev_patients = set(patients[: int(len(patients) * CFG["dev_fraction"])])
    is_dev = pid.isin(dev_patients)

    out = HERE / "splits"
    out.mkdir(exist_ok=True)
    train[~is_dev].to_csv(out / "train.csv", index=False)
    train[is_dev].to_csv(out / "dev.csv", index=False)
    valid.to_csv(out / "valid.csv", index=False)
    files = ["train.csv", "dev.csv", "valid.csv"]

    test_csv = root / CFG["test_labels_csv"]
    if test_csv.exists():
        pd.read_csv(test_csv).to_csv(out / "test.csv", index=False)
        files.append("test.csv")
    else:
        print(f"WARNING: no test set at {test_csv}; rerun split.py once it is downloaded.")

    manifest = {
        "sha256": {f: sha256(out / f) for f in files},
        "rows": {f: int(len(pd.read_csv(out / f))) for f in files},
        "patients": {"train": int(len(patients) - len(dev_patients)), "dev": len(dev_patients)},
        "split_seed": CFG["split_seed"],
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
