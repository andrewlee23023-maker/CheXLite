"""Train a phone-sized multi-label CXR classifier.

    python src/train.py           # loop budget from eval/budget.yaml (used by agents)
    python src/train.py --full    # final model, all data (human only)
Writes checkpoints/model.pt.
"""
import argparse
import math
import time

import pandas as pd
import timm
import torch
import yaml
from torch.utils.data import DataLoader

from common import LABELS, REPO, CXRDataset, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "src" / "config.yaml"))
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    budget = yaml.safe_load((REPO / "eval" / "budget.yaml").read_text())["full" if args.full else "loop"]
    if cfg["img_size"] > budget["max_img_size"]:
        raise SystemExit(f"img_size {cfg['img_size']} exceeds budget {budget['max_img_size']}")
    seed_everything(cfg["seed"])

    df = pd.read_csv(REPO / "eval" / "splits" / "train.csv")
    if cfg["frontal_only"]:
        df = df[df["Frontal/Lateral"] == "Frontal"]
    if budget["train_subset"] and budget["train_subset"] < len(df):
        df = df.sample(n=budget["train_subset"], random_state=cfg["seed"])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CXRDataset(df, cfg, train=True)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["num_workers"], pin_memory=dev == "cuda",
                    persistent_workers=cfg["num_workers"] > 0)

    model = timm.create_model(cfg["arch"], pretrained=cfg["pretrained"], num_classes=len(LABELS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = budget["epochs"] * len(dl)
    warm = max(1, int(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = cfg["amp"] and dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    t0 = time.time()
    for ep in range(budget["epochs"]):
        model.train()
        run = 0.0
        for step, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            mask = ~torch.isnan(y)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = (bce(logits.float(), torch.nan_to_num(y)) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if (step + 1) % 100 == 0:
                print(f"ep {ep} step {step + 1}/{len(dl)} loss {run / 100:.4f} ({time.time() - t0:.0f}s)", flush=True)
                run = 0.0

    (REPO / "checkpoints").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "full": args.full}, REPO / "checkpoints" / "model.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved checkpoints/model.pt | {n_params / 1e6:.2f}M params | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
