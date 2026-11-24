"""GRPO on top of the SFT note model, with the deterministic faithfulness verifier as reward.

Prompts are fresh synthetic facts (new seed, balanced sampler), never the real-output eval sets. For each prompt we sample
G notes, score each with note_rules.verify(), and use group-normalized advantages
(r - mean) / std. Groups with zero variance carry no signal and are skipped. Single on-policy update
per batch (importance ratio = 1, so no clipping is needed), plus a per-token k3 KL penalty against the
frozen SFT model (LoRA adapter disabled), which stops the policy drifting into regex-gaming text.

Reward: +1 if the note passes every rule; otherwise -0.25 per violated rule (floor -1).
Notes over 80 words get an extra -0.5.

    python llm/grpo.py --init llm/runs/gemma270m-v1 --out llm/runs/gemma270m-grpo
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_pairs import sample_facts
from note_rules import facts_to_prompt, verify

HERE = Path(__file__).resolve().parent


def reward(note, facts):
    v = verify(note, facts)
    r = 1.0 if v["ok"] else max(-1.0, -0.25 * len(v["problems"]))
    if len(note.split()) > 80:
        r -= 0.5
    return r, v["ok"]


def completion_logprobs(model, seqs, attn, n_comp):
    """Per-token log-probs of the last n_comp positions of each sequence (completion region)."""
    # Left-padded prompts: derive RoPE positions from the mask, exactly as generate() does.
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=seqs, attention_mask=attn, position_ids=pos, logits_to_keep=n_comp + 1)
    logits = out.logits[:, :-1, :].float()               # predicts tokens at positions -n_comp..-1
    targets = seqs[:, -n_comp:]
    return torch.gather(F.log_softmax(logits, -1), 2, targets.unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(HERE / "runs" / "gemma270m-v1"))
    ap.add_argument("--out", default=str(HERE / "runs" / "gemma270m-grpo"))
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts", type=int, default=8, help="prompts per step")
    ap.add_argument("--group", type=int, default=8, help="samples per prompt")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl", type=float, default=0.04)
    ap.add_argument("--micro", type=int, default=8, help="sequences per backward micro-batch")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.init)
    tok.padding_side = "left"
    stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    base = AutoModelForCausalLM.from_pretrained(a.init, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                            "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    log, t0 = [], time.time()

    for step in range(a.steps):
        facts = [sample_facts(rng, balanced=True) for _ in range(a.prompts)]
        texts = [tok.apply_chat_template([{"role": "user", "content": facts_to_prompt(f)}],
                                         add_generation_prompt=True, tokenize=False) for f in facts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        model.eval()
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=a.max_new,
                                 num_return_sequences=a.group, eos_token_id=stop, pad_token_id=pad)
        P = enc["input_ids"].shape[1]
        comp = gen[:, P:]
        # completion mask: tokens up to and including the first stop token
        is_stop = torch.zeros_like(comp, dtype=torch.bool)
        for s in stop:
            is_stop |= comp == s
        after = (is_stop.cumsum(1) - is_stop.long()) > 0
        cmask = (~after).float()
        notes = [tok.decode(c[m.bool()], skip_special_tokens=True).strip() for c, m in zip(comp, cmask)]

        rs, oks = zip(*[reward(n, facts[i // a.group]) for i, n in enumerate(notes)])
        r = torch.tensor(rs, device=dev).view(a.prompts, a.group)
        std = r.std(1, keepdim=True)
        adv = ((r - r.mean(1, keepdim=True)) / (std + 1e-4)).view(-1)
        keep = (std.view(-1) > 1e-6).repeat_interleave(a.group)

        prompt_attn = enc["attention_mask"].repeat_interleave(a.group, 0)
        attn = torch.cat([prompt_attn, cmask.long()], 1)
        idx = keep.nonzero().view(-1).tolist()
        model.train()
        opt.zero_grad(set_to_none=True)
        tot_kl = 0.0
        n_tok = cmask[idx].sum().clamp(min=1) if idx else torch.tensor(1.0, device=dev)
        for i in range(0, len(idx), a.micro):
            b = idx[i:i + a.micro]
            seqs, am, m = gen[b], attn[b], cmask[b]
            lp = completion_logprobs(model, seqs, am, comp.shape[1])
            with torch.no_grad(), model.disable_adapter():
                ref = completion_logprobs(model, seqs, am, comp.shape[1])
            d = ref - lp
            kl = torch.exp(d) - d - 1                       # k3 estimator, >= 0
            loss = ((-adv[b].unsqueeze(1) * lp + a.kl * kl) * m).sum() / n_tok
            loss.backward()
            tot_kl += (kl * m).sum().item()
        if idx:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        rec = {"step": step, "pass": round(sum(oks) / len(oks), 3), "reward": round(sum(rs) / len(rs), 3),
               "groups_used": len(idx) // a.group, "kl": round(tot_kl / float(n_tok), 4),
               "secs": round(time.time() - t0)}
        log.append(rec)
        if step % 5 == 0:
            print(json.dumps(rec), flush=True)

    out = Path(a.out)
    model.merge_and_unload().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "grpo_log.json").write_text(json.dumps({"args": vars(a), "log": log}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
