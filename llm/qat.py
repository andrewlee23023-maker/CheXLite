"""Quantization-aware fine-tuning of the note model for the browser's 4-bit export.

The SFT+GRPO model passes the faithfulness verifier on 100% of held-out notes at full precision, but the
4-bit export drops to ~94%: rounding flips a few near-tie greedy choices. GRPO at full precision has no
signal left (rewards are ~all 1), so this trains THROUGH the exact 4-bit rounding instead:

  * every decoder linear computes with W_eff = fake_q4(W + B A * s), straight-through gradients to the
    LoRA factors A, B (the base W and the embedding are frozen). fake_q4 reproduces ONNX Runtime's
    block-32 symmetric RTN (llm/q4sim.py), so merging B A into W and exporting yields exactly the
    weights that were trained.
  * targets: the full-precision model's own greedy notes (self-distillation), kept only if they pass
    the verifier, on NEW development cases (data/dev2_train.jsonl: real classifier outputs on dev images
    in neither the inspected set nor the held-out set) plus synthetic prompts from both samplers.
  * model selection uses data/dev2_select.jsonl only; the held-out set is scored once, at the end.

    python llm/qat.py --out llm/runs/gemma270m-qat
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import q4sim
from make_pairs import sample_facts
from note_rules import facts_to_prompt, verify

HERE = Path(__file__).resolve().parent


class QATLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.weight = base.weight                       # frozen
        self.weight.requires_grad_(False)
        self.bias = base.bias
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.s = alpha / r

    def merged(self):
        return self.weight + (self.B @ self.A) * self.s

    def forward(self, x):
        w = self.merged()
        wq = w + (q4sim.fake_q4(w) - w).detach()        # straight-through estimator
        return F.linear(x, wq, self.bias)


def wrap(model, r, alpha):
    mods = []
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                m = QATLoRALinear(getattr(parent, n), r, alpha)
                setattr(parent, n, m)
                mods.append(m)
    return mods


@torch.no_grad()
def unwrap_merged(model):
    """Replace QAT wrappers by plain Linears holding W + B A s (full precision; the export quantizes it)."""
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                m = getattr(parent, n)
                lin = nn.Linear(m.weight.shape[1], m.weight.shape[0], bias=m.bias is not None, device=m.weight.device)
                lin.weight.copy_(m.merged())
                if m.bias is not None:
                    lin.bias.copy_(m.bias)
                setattr(parent, n, lin)


def build_targets(teacher, tok, prompts_facts, batch, mask):
    notes = q4sim.greedy_notes(teacher, tok, [p for p, _ in prompts_facts], batch=batch, mask=mask)
    keep = [(p, n) for (p, f), n in zip(prompts_facts, notes) if verify(n, f)["ok"]]
    return keep


def encode(tok, prompt, note):
    pre = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    a = tok(pre, add_special_tokens=False)["input_ids"]
    b = tok(note + "<end_of_turn>", add_special_tokens=False)["input_ids"]
    return a + b, [-100] * len(a) + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "runs" / "gemma270m-grpo"))
    ap.add_argument("--out", default=str(HERE / "runs" / "gemma270m-qat"))
    ap.add_argument("--synthetic", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--evals", type=int, default=2, help="selection-set evaluations per epoch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda()
    model.eval()
    mask = q4sim.keep_mask(model.config.vocab_size, "cuda")
    sel = [json.loads(l) for l in open(HERE / "data" / "dev2_select.jsonl", encoding="utf-8")]

    # 1. Targets from the full-precision model (teacher = the same weights before fake quantization).
    tgt_path = out / "targets.jsonl"
    if tgt_path.exists():
        targets = [tuple(json.loads(l)) for l in open(tgt_path, encoding="utf-8")]
    else:
        pf = [(json.loads(l)["prompt"], json.loads(l)["facts"]) for l in open(HERE / "data" / "dev2_train.jsonl", encoding="utf-8")]
        for i in range(a.synthetic):
            f = sample_facts(rng, balanced=(i % 2 == 0))
            pf.append((facts_to_prompt(f), f))
        targets = build_targets(model, tok, pf, batch=32, mask=mask)
        with open(tgt_path, "w", encoding="utf-8") as fh:
            for t in targets:
                fh.write(json.dumps(t) + "\n")
    print(f"targets: {len(targets)} verified notes", flush=True)

    # 2. Fake-quantize the frozen parts exactly as the export does (embedding 4-bit, head fp32).
    q4sim.apply_fake_q4(model)          # also fake-quantizes decoder linears; wrappers re-quantize W + BA
    for p in model.parameters():
        p.requires_grad_(False)
    mods = wrap(model, a.r, 2 * a.r)
    # Loss over the kept output vocabulary only: the export masks every other token, so this is the
    # distribution the app decodes from (and full-vocab logits would not fit in GPU memory).
    keep = (mask == 0).nonzero().squeeze(1)
    head_keep = model.lm_head.weight[keep].detach()          # fp32 rows, as in the export
    to_keep = torch.full((model.config.vocab_size,), -1, dtype=torch.long, device="cuda")
    to_keep[keep] = torch.arange(len(keep), device="cuda")
    # GPU memory: the full fp32 output layer (670 MB) is only needed when generating for evaluation.
    model.lm_head.to("cpu")
    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    # wrappers hold the already-quantized W; fake_q4 is idempotent on it, so at init W_eff == shipped weights
    params = [p for m in mods for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    data = [encode(tok, p, n) for p, n in targets]
    steps = a.epochs * (len(data) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))

    def select_rate():
        model.eval()
        model.lm_head.to("cuda")
        model.config.use_cache = True
        r, _, _ = q4sim.pass_rate(model, tok, sel, mask=mask, batch=16)
        model.config.use_cache = False
        model.lm_head.to("cpu")
        torch.cuda.empty_cache()
        model.train()
        return r

    log = {"base_select_pass": 284 / 300, "steps": steps, "args": vars(a), "n_targets": len(targets), "evals": []}
    # (start value measured with q4sim.py on the same set and confirmed by the first run of this script)
    print("select pass at start (4-bit):", log["base_select_pass"], flush=True)
    per_epoch = len(data) // a.bs
    eval_at = {round(per_epoch * (k + 1) / a.evals) for k in range(a.evals)}
    best = (log["base_select_pass"], -1)
    model.train()
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        rng.shuffle(data)
        for i in range(0, len(data) - a.bs + 1, a.bs):
            batch = data[i:i + a.bs]
            L = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch], device="cuda")
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device="cuda")
            att = (torch.arange(L, device="cuda")[None] < torch.tensor([len(x) for x, _ in batch], device="cuda")[:, None]).long()
            hs = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
            y = lab[:, 1:]
            tsel = y != -100
            yk = to_keep[y[tsel]]
            assert (yk >= 0).all(), "target token outside the kept vocabulary"
            loss = F.cross_entropy(hs[tsel] @ head_keep.T, yk)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s "
                      f"gpu {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            if (step - ep * per_epoch) in eval_at:
                r = select_rate()
                log["evals"].append({"epoch": ep + 1, "step": step, "select_pass": r})
                print(f"step {step}: select pass (4-bit) {r:.4f}", flush=True)
                if r > best[0]:
                    best = (r, step)
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}, out / "lora_best.pt")
                (out / "qat_log.json").write_text(json.dumps(log, indent=2))
    log["best"] = {"select_pass": best[0], "step": best[1]}
    (out / "qat_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log["best"]), flush=True)
    if best[1] < 0:
        print("no improvement over the shipped model on dev2_select; nothing saved")
        return

    # 3. Save a full-precision checkpoint whose 4-bit export equals the trained model. The base W in the
    # wrappers is already fake-quantized, so merged = q4(W0) + BA, and the export re-quantizes that.
    sd = torch.load(out / "lora_best.pt")
    model.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    unwrap_merged(model)
    model.lm_head.to("cuda")
    model.lm_head.weight = model.model.embed_tokens.weight   # re-tie: export reads the fp32 head from the
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)   # embedding, so restore the
    model.model.embed_tokens.weight.data.copy_(base.model.embed_tokens.weight.data.cuda())  # fp32 table
    model.config.tie_word_embeddings = True
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    print("saved", out / "model")


if __name__ == "__main__":
    main()
