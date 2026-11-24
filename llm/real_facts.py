"""Build LLM evaluation inputs from REAL classifier outputs on dev-split images.

Runs the exported ONNX model (the exact graph the app runs: probs + in-graph Grad-CAM) on frontal
dev images, preprocessed like training, and converts outputs to facts with the same logic as the app.
Dev patients are never in the classifier's training data, and these facts are never in the LLM's
training data (which is synthetic).

    python llm/real_facts.py --n 400 --seed 0
    python llm/real_facts.py --n 400 --seed 1 --exclude llm/data/real_facts.jsonl --out llm/data/real_facts_heldout.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from note_rules import build_facts, facts_to_prompt

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app" / "model"
MAIN = HERE.parent.parent / "chexlite"


def preprocess(path, s, mean, std):
    im = Image.open(path).convert("L").convert("RGB").resize((s, s), Image.BILINEAR)
    x = (np.asarray(im, dtype=np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return x.transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "data" / "real_facts.jsonl"))
    ap.add_argument("--exclude", nargs="*", default=[], help="facts files whose images must not be reused")
    a = ap.parse_args()

    meta = json.loads((APP / "meta.json").read_text())
    sess = ort.InferenceSession(str(APP / "chexlite.onnx"), providers=["CPUExecutionProvider"])
    dev = pd.read_csv(MAIN / "eval" / "splits" / "dev.csv")
    used = {json.loads(l)["path"] for f in a.exclude for l in open(f, encoding="utf-8")}
    dev = dev[(dev["Frontal/Lateral"] == "Frontal") & ~dev["Path"].isin(used)].sample(n=a.n, random_state=a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_refer = n_flag = 0
    with out.open("w", encoding="utf-8") as fh:
        for p in dev["Path"]:
            x = preprocess(MAIN / "data" / "chexpert" / p, meta["img_size"], meta["mean"], meta["std"])
            probs, cams = sess.run(None, {"image": x})
            f = build_facts(probs[0], cams[0], meta)
            n_refer += f["refer"]
            n_flag += sum(x["flagged"] for x in f["findings"])
            fh.write(json.dumps({"path": p, "facts": f, "prompt": facts_to_prompt(f)}) + "\n")
    print(json.dumps({"n": a.n, "refer_rate": n_refer / a.n, "mean_flags": n_flag / a.n, "out": str(out)}))


if __name__ == "__main__":
    main()
