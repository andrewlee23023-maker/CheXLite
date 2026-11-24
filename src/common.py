import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
EVAL_CFG = yaml.safe_load((REPO / "eval" / "eval_config.yaml").read_text())
LABELS = EVAL_CFG["labels"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def data_root() -> Path:
    p = Path(os.environ.get("CHEXPERT_ROOT", EVAL_CFG["data_root"]))
    return p if p.is_absolute() else REPO / p


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(cfg: dict, train: bool):
    s = cfg["img_size"]
    t = [transforms.Grayscale(num_output_channels=3), transforms.Resize((s, s))]
    if train:
        a = cfg["augment"]
        t.append(transforms.RandomAffine(a["degrees"], translate=(a["translate"],) * 2,
                                         scale=tuple(a["scale"])))
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


def encode_labels(df: pd.DataFrame, policy: dict) -> np.ndarray:
    """blank -> 0; -1 -> 1 (ones), 0 (zeros) or NaN (ignore, masked in the loss)."""
    out = []
    for lab in LABELS:
        col = df[lab].fillna(0.0).astype(float)
        u = {"ones": 1.0, "zeros": 0.0, "ignore": np.nan}[policy[lab]]
        out.append(col.replace(-1.0, u).to_numpy())
    return np.stack(out, axis=1).astype(np.float32)


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, train: bool, with_labels: bool = True):
        self.root = data_root()
        self.paths = df["Path"].tolist()
        self.tf = build_transform(cfg, train)
        self.y = encode_labels(df, cfg["uncertain_policy"]) if with_labels else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            x = self.tf(im.convert("L"))
        if self.y is None:
            return x, i
        return x, torch.from_numpy(self.y[i])
