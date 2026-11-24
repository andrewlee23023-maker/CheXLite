"""Coverage check for the browser export's restricted output vocabulary (export_browser.py restrict_lm_head).

The export keeps only the 453 tokens in the SFT notes and prompts. This greedy-decodes the final model on
DEVELOPMENT prompts only: the dev facts it was tuned on (real_facts.jsonl) plus synthetic prompts from
both training samplers, and records which tokens it emits (result: 132 distinct tokens over 2,400 notes,
4 of them outside the kept set, one a stray CJK character). The held-out eval facts (real_facts_heldout.jsonl) are never used here.

    python llm/note_vocab.py --model llm/runs/gemma270m-grpo
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_pairs import sample_facts
from note_rules import facts_to_prompt

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "runs" / "gemma270m-grpo"))
    ap.add_argument("--synthetic", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=str(HERE / "data" / "note_vocab_gen.json"))
    a = ap.parse_args()

    rng = random.Random(7)
    prompts = [json.loads(l)["prompt"] for l in open(HERE / "data" / "real_facts.jsonl", encoding="utf-8")]
    prompts += [facts_to_prompt(sample_facts(rng, balanced=(i % 2 == 0))) for i in range(a.synthetic)]

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda().eval()
    stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]
    ids = set()
    for i in range(0, len(prompts), a.batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + a.batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:].tolist():
            ids.update(t for t in row if t != tok.pad_token_id)
        print(f"{min(i + a.batch, len(prompts))}/{len(prompts)} prompts, {len(ids)} distinct tokens", flush=True)
    Path(a.out).write_text(json.dumps({"model": Path(a.model).name, "n_prompts": len(prompts),
                                       "sources": ["real_facts.jsonl (dev)", "synthetic default+balanced samplers"],
                                       "ids": sorted(ids)}))


if __name__ == "__main__":
    main()
