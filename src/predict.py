"""Predict per-image probabilities for every row of a split CSV.
Called by eval/evaluate.py; outputs CSV with Path + one probability column per label."""
import argparse

import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from common import LABELS, REPO, CXRDataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(REPO / "checkpoints" / "model.pt"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(cfg["arch"], pretrained=False, num_classes=len(LABELS))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(dev)

    df = pd.read_csv(args.csv)
    dl = DataLoader(CXRDataset(df, cfg, train=False, with_labels=False),
                    batch_size=cfg["batch_size"] * 2, num_workers=cfg["num_workers"])
    probs = torch.empty(len(df), len(LABELS))
    for x, idx in dl:
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=cfg["amp"] and dev == "cuda"):
            probs[idx] = torch.sigmoid(model(x.to(dev)).float()).cpu()

    out = pd.DataFrame(probs.numpy(), columns=LABELS)
    out.insert(0, "Path", df["Path"].values)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
