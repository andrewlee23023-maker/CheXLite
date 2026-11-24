"""LoRA fine-tune Gemma 3 270M (instruction-tuned) to turn classifier output into a clinician note.

Data: template+paraphrase pairs (make_pairs.py) and verified Claude-written pairs (claude_pairs.py).
Loss is computed only on the note tokens. LoRA adapters are merged into the base weights at the end,
so the exported model is a plain Gemma checkpoint (for ONNX/transformers.js export).

    python llm/finetune.py --out llm/runs/gemma270m-v1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE = "google/gemma-3-270m-it"


def load_pairs(paths, n_template, seed):
    rng = random.Random(seed)
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"skip missing {p}")
            continue
        rs = [json.loads(l) for l in p.open(encoding="utf-8")]
        if "template" in p.name and n_template and len(rs) > n_template:
            rs = rng.sample(rs, n_template)
        rows += rs
        print(f"{p.name}: {len(rs)} pairs")
    rng.shuffle(rows)
    return rows


def encode(tok, prompt, note, max_len):
    # Render to text, then tokenize: identical to how eval_llm.py builds inference prompts.
    p_text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
    n_ids = tok(note + "<end_of_turn>", add_special_tokens=False)["input_ids"]
    ids = (p_ids + n_ids)[:max_len]
    labels = ([-100] * len(p_ids) + n_ids)[:max_len]
    return ids, labels


def batches(data, bs, pad_id, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id)
        lab = torch.full((len(chunk), L), -100)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, (a, b) in enumerate(chunk):
            ids[k, :len(a)] = torch.tensor(a)
            lab[k, :len(b)] = torch.tensor(b)
            att[k, :len(a)] = 1
        yield ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(HERE / "data" / "template_pairs.jsonl"),
                                                  str(HERE / "data" / "claude_pairs.jsonl")])
    ap.add_argument("--n-template", type=int, default=3000, help="template pairs to sample (0 = all)")
    ap.add_argument("--out", default=str(HERE / "runs" / "gemma270m-v1"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = bs*accum)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    rows = load_pairs(a.data, a.n_template, a.seed)
    enc = [encode(tok, r["prompt"], r["note"], a.max_len) for r in rows]
    n_val = max(50, int(len(enc) * a.val_frac))
    val, train = enc[:n_val], enc[n_val:]
    print(f"train {len(train)}  val {len(val)}  max len {max(len(x[0]) for x in enc)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = a.epochs * math.ceil(len(train) / (a.bs * a.accum))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def val_loss():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for ids, lab, att in batches(val, a.bs, pad, False, rng):
                out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev))
                k = (lab != -100).sum().item()
                tot += out.loss.item() * k
                n += k
        model.train()
        return tot / n

    print(f"val loss before: {val_loss():.4f}")
    step, micro, t0, hist = 0, 0, time.time(), []
    model.train()
    for ep in range(a.epochs):
        for ids, lab, att in batches(train, a.bs, pad, True, rng):
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev))
            (out.loss / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{total} loss {out.loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        vl = val_loss()
        hist.append({"epoch": ep, "val_loss": vl})
        print(f"epoch {ep} val loss {vl:.4f}")

    out_dir = Path(a.out)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    (out_dir / "train_info.json").write_text(json.dumps({**vars(a), "n_train": len(train), "n_val": len(val),
                                                         "history": hist, "seconds": round(time.time() - t0)}, indent=2))
    print(f"saved merged model to {out_dir}")


if __name__ == "__main__":
    main()
