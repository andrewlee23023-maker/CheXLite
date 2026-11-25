"""Simulate the browser export's 4-bit weights in PyTorch, so the shipped model's behaviour can be measured
(and trained against) on the GPU instead of decoding one note at a time through ONNX Runtime on CPU.

ORT's symmetric block-wise RTN (MatMulNBits / GatherBlockQuantized, block 32): per block, take the
element with the largest magnitude (keeping its sign) as `amax`, scale = amax / -8 (ties go to the negative extreme), q = clamp(floor(w /
scale + 0.5) + 8, 0, 15), dequantized w' = (q - 8) * scale. `check_against_onnx` confirms this reproduces the
exported weights.

    python llm/q4sim.py --check          # compare simulated weights with app/llm-model/onnx
    python llm/q4sim.py --eval FACTS     # greedy verifier pass rate of the simulated 4-bit model
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from note_rules import verify  # noqa: E402

BLOCK = 32
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def fake_q4(w: torch.Tensor) -> torch.Tensor:
    """Block-wise along the last (input) dim, as ORT quantizes MatMul B along K and Gather along axis 1."""
    shp = w.shape
    b = w.reshape(-1, BLOCK).float()
    mx, mn = b.max(dim=1, keepdim=True).values, b.min(dim=1, keepdim=True).values
    amax = torch.where(mx.abs() > mn.abs(), mx, mn)   # ties go to the negative extreme, as in ORT
    scale = amax / -8.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.floor(b / scale + 0.5), -8, 7) + 8   # ORT rounds halves up
    return ((q - 8) * scale).reshape(shp).to(w.dtype)


def quantized_params(model):
    """(name, param) for every weight the export stores in 4 bits: all decoder linears + the embedding."""
    for name, p in model.named_parameters():
        if (name.endswith(".weight") and any(f".{l}." in name for l in LINEARS)) or name.endswith("embed_tokens.weight"):
            yield name, p


@torch.no_grad()
def apply_fake_q4(model, keep_fp_head=True):
    """In place: replace each 4-bit-exported weight by its dequantized value. Gemma ties lm_head to the
    embedding; the export keeps the head in fp32, so untie first and keep an fp32 copy for it."""
    if keep_fp_head and model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone(), requires_grad=False)
    for _, p in quantized_params(model):
        p.copy_(fake_q4(p))
    return model


def keep_mask(vocab_size, device):
    """Logit mask matching the shipped export's restricted output vocabulary (read from the ONNX graph)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(HERE.parent / "app" / "llm-model" / "onnx" / "model_q4.onnx"), load_external_data=True)
    vmap = next(numpy_helper.to_array(t) for t in m.graph.initializer if t.name == "lm_head_vocab_map")
    keep = torch.tensor((vmap < vmap.max()).nonzero()[0], device=device)
    mask = torch.full((vocab_size,), float("-inf"), device=device)
    mask[keep] = 0
    return mask


@torch.no_grad()
def greedy_notes(model, tok, prompts, batch=32, mask=None, max_new=160):
    from transformers import LogitsProcessor, LogitsProcessorList

    class Mask(LogitsProcessor):
        def __call__(self, ids, scores):
            return scores + mask

    tok.padding_side = "left"
    stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]
    procs = LogitsProcessorList([Mask()]) if mask is not None else None
    notes = []
    for i in range(0, len(prompts), batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stop,
                             pad_token_id=tok.pad_token_id, logits_processor=procs)
        notes += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [n.strip() for n in notes]


def pass_rate(model, tok, rows, mask=None, batch=32):
    notes = greedy_notes(model, tok, [r["prompt"] for r in rows], batch=batch, mask=mask)
    oks = [verify(n, r["facts"])["ok"] for n, r in zip(notes, rows)]
    return sum(oks) / len(oks), oks, notes


def check_against_onnx(model):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(HERE.parent / "app" / "llm-model" / "onnx" / "model_q4.onnx"), load_external_data=True)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    node = next(n for n in m.graph.node if n.op_type == "MatMulNBits" and "layers.0/self_attn/q_proj" in n.name)
    packed, scales = inits[node.input[1]], inits[node.input[2]]            # [N, K/32, 16] uint8, [N, K/32]
    lo, hi = (packed & 0x0F).astype(np.float32), (packed >> 4).astype(np.float32)
    q = np.stack([lo, hi], -1).reshape(packed.shape[0], packed.shape[1], BLOCK)
    deq = ((q - 8) * scales[..., None]).reshape(packed.shape[0], -1)          # [N, K] = [out, in]
    sim = fake_q4(model.model.layers[0].self_attn.q_proj.weight.detach().cpu().float()).numpy()
    print(json.dumps({"q_proj_layer0_max_abs_diff": float(np.abs(deq - sim).max()),
                      "frac_exact": float((np.abs(deq - sim) < 1e-6).mean())}))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "runs" / "gemma270m-grpo"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--eval")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="evaluate without fake quantization")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    if a.check:
        check_against_onnx(model)
        return
    model = model.cuda().eval()
    if not a.fp32:
        apply_fake_q4(model)
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8")]
    rows = rows[:a.n] if a.n else rows
    rate, oks, _ = pass_rate(model, tok, rows, mask=keep_mask(model.config.vocab_size, "cuda"))
    print(json.dumps({"model": Path(a.model).name, "fake_q4": not a.fp32, "facts": Path(a.eval).name,
                      "n": len(rows), "pass_rate": round(rate, 4), "pass": sum(oks)}))


if __name__ == "__main__":
    main()
