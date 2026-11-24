"""Faithfulness evaluation of the clinician-note LLM.

Metric: verifier pass rate, i.e. the share of generated notes that satisfy every rule in note_rules.verify()
(referral decision matches, every flagged finding reported, no unflagged finding asserted, regions and
numbers match the model output, no out-of-scope conditions). Reported with a Wilson 95% CI, failure types,
length, and generation latency. In the app, a failing note is replaced by the template, so
pass rate = the share of results where the LLM note is actually shown.

    python llm/eval_llm.py --model llm/runs/gemma270m-v1 --facts llm/data/real_facts.jsonl
    python llm/eval_llm.py --model google/gemma-3-270m-it --zero-shot      # baseline with written rules
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from claude_pairs import SYSTEM
from note_rules import verify

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--facts", default=str(HERE / "data" / "real_facts.jsonl"))
    ap.add_argument("--zero-shot", action="store_true", help="prepend the written rules (for the untuned baseline)")
    ap.add_argument("--n", type=int, default=0, help="limit examples (0 = all)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.facts, encoding="utf-8")]
    if a.n:
        rows = rows[:a.n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()

    prompts = []
    for r in rows:
        content = (SYSTEM + "\n\n" + r["prompt"]) if a.zero_shot else r["prompt"]
        prompts.append(tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False))

    notes, t0 = [], time.time()
    for i in range(0, len(prompts), a.bs):
        enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False,
                                 eos_token_id=[tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id])
        for g in gen[:, enc["input_ids"].shape[1]:]:
            notes.append(tok.decode(g, skip_special_tokens=True).strip())
    secs = time.time() - t0

    ok, fails, words, examples = 0, {}, [], []
    for r, note in zip(rows, notes):
        v = verify(note, r["facts"])
        ok += v["ok"]
        words.append(len(note.split()))
        for p in v["problems"]:
            k = " ".join(p.split(" ")[:2])
            fails[k] = fails.get(k, 0) + 1
        if not v["ok"] and len(examples) < 5:
            examples.append({"prompt": r["prompt"], "note": note, "problems": v["problems"]})

    res = {"model": a.model, "facts": Path(a.facts).name, "zero_shot": a.zero_shot, "n": len(rows),
           "pass_rate": round(ok / len(rows), 4), "pass_ci95": wilson(ok, len(rows)),
           "failure_types": dict(sorted(fails.items(), key=lambda x: -x[1])),
           "mean_words": round(sum(words) / len(words), 1), "gpu_seconds_per_note": round(secs / len(rows), 3),
           "failed_examples": examples}
    print(json.dumps({k: v for k, v in res.items() if k != "failed_examples"}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
