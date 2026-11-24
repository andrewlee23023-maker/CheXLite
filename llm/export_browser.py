"""Export the fine-tuned note model for transformers.js: ONNX + 4-bit weights, split into <90 MB files.

GitHub rejects files over 100 MB and GitHub Pages doesn't serve Git LFS, so external weights are
packed into chunk files named the way transformers.js 4.x expects (model_q4.onnx_data,
model_q4.onnx_data_1, ...) and declared in config.json under transformers.js_config.

Runs in the isolated export venv (optimum pins its own transformers):
    PYTHONNOUSERSITE=1 .venv-export/Scripts/python llm/export_browser.py --model llm/runs/gemma270m-v1
Then verifies the quantized model with ONNX Runtime on real eval facts (greedy decoding, KV cache).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

HERE = Path(__file__).resolve().parent
CHUNK_BYTES = 90 * 1024 * 1024


def split_lm_head(model: onnx.ModelProto, parts: int = 2) -> None:
    """Replace logits = h @ W (W: [hidden, vocab]) by Concat([h @ W_0, ..., h @ W_k], -1) along the vocab axis,
    so no single quantized initializer exceeds the per-file limit. Mathematically identical."""
    from onnx import helper, numpy_helper
    inits = {t.name: t for t in model.graph.initializer}
    for i, n in enumerate(model.graph.node):
        if n.op_type == "MatMul" and "lm_head" in n.name and n.input[1] in inits:
            W = numpy_helper.to_array(inits[n.input[1]])
            cuts = np.array_split(np.arange(W.shape[1]), parts)
            new_nodes, outs = [], []
            for k, c in enumerate(cuts):
                wname, oname = f"{n.input[1]}_part{k}", f"{n.output[0]}_part{k}"
                model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W[:, c]), wname))
                new_nodes.append(helper.make_node("MatMul", [n.input[0], wname], [oname], name=f"{n.name}_part{k}"))
                outs.append(oname)
            new_nodes.append(helper.make_node("Concat", outs, [n.output[0]], name=f"{n.name}_concat", axis=-1))
            model.graph.initializer.remove(inits[n.input[1]])
            del model.graph.node[i]
            for k, nn in enumerate(new_nodes):
                model.graph.node.insert(i + k, nn)
            return


def build_note_vocab(tok_dir: Path) -> list:
    """Token ids the note model may emit: every token in the SFT training notes and prompts, plus the
    special tokens. Built from training data only (never the held-out eval facts)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    keep = set(tok.all_special_ids) | {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}
    for f in ("claude_pairs.jsonl", "template_pairs.jsonl"):
        for line in open(HERE / "data" / f, encoding="utf-8"):
            r = json.loads(line)
            for text in (r["note"], " " + r["note"], r["prompt"]):
                keep.update(tok(text, add_special_tokens=False)["input_ids"])
    keep = sorted(i for i in keep if i is not None)
    gen = HERE / "data" / "note_vocab_gen.json"   # coverage check only (llm/note_vocab.py)
    if gen.exists():
        miss = set(json.loads(gen.read_text())["ids"]) - set(keep)
        print(f"coverage: {len(miss)} tokens the model generates on dev prompts are outside the kept set")
    return keep


def restrict_lm_head(model: onnx.ModelProto, keep: list) -> None:
    """logits = h @ W over the 262k vocabulary becomes h @ W[:, keep] (fp32, exact for kept tokens), then
    every other token gets -1e9 so greedy decoding can never pick it. Implemented as
    Pad(sub, +1 column of -1e9) -> Gather(vocab_map) so the output keeps its full [B, S, vocab] shape
    and the browser runtime needs no changes."""
    from onnx import helper, numpy_helper
    inits = {t.name: t for t in model.graph.initializer}
    for i, n in enumerate(model.graph.node):
        if n.op_type == "MatMul" and "lm_head" in n.name and n.input[1] in inits:
            W = numpy_helper.to_array(inits[n.input[1]])            # [hidden, vocab]
            V, K = W.shape[1], len(keep)
            vmap = np.full(V, K, dtype=np.int64)
            vmap[np.array(keep)] = np.arange(K)
            model.graph.initializer.remove(inits[n.input[1]])
            model.graph.initializer.extend([
                numpy_helper.from_array(np.ascontiguousarray(W[:, keep]), "lm_head_keep_W"),
                numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 1], dtype=np.int64), "lm_head_pads"),
                numpy_helper.from_array(np.array(-1e9, dtype=np.float32), "lm_head_neg"),
                numpy_helper.from_array(vmap, "lm_head_vocab_map")])
            new = [helper.make_node("MatMul", [n.input[0], "lm_head_keep_W"], ["lm_head_sub"], name="/lm_head/MatMul_keep"),
                   helper.make_node("Pad", ["lm_head_sub", "lm_head_pads", "lm_head_neg"], ["lm_head_padded"], name="/lm_head/Pad", mode="constant"),
                   helper.make_node("Gather", ["lm_head_padded", "lm_head_vocab_map"], [n.output[0]], name="/lm_head/ToVocab", axis=2)]
            del model.graph.node[i]
            for k, nn in enumerate(new):
                model.graph.node.insert(i + k, nn)
            return
    raise RuntimeError("lm_head MatMul not found")


def int4_gather_to_uint8(model: onnx.ModelProto) -> int:
    """Repack INT4 GatherBlockQuantized data as uint8 with bits=4: symmetric int4 q in [-8, 7] maps to
    uint4 q + 8 (flip each nibble's sign bit, XOR 0x88 per byte) with zero point 8, same weights."""
    from onnx import helper
    inits = {t.name: t for t in model.graph.initializer}
    n_conv = 0
    for n in model.graph.node:
        if n.op_type != "GatherBlockQuantized":
            continue
        assert len(n.input) < 4 or not n.input[3], "zero points not supported here (use symmetric)"
        t = inits[n.input[0]]
        if t.data_type != TensorProto.INT4:
            continue
        raw = np.frombuffer(t.raw_data, dtype=np.uint8) ^ 0x88
        rows, cols = t.dims
        t.data_type = TensorProto.UINT8
        t.dims[:] = [rows, cols // 2]
        t.raw_data = raw.tobytes()
        n.attribute.append(helper.make_attribute("bits", 4))
        n_conv += 1
    return n_conv


def gather_q4_to_standard_ops(model: onnx.ModelProto) -> int:
    """ORT Web's WASM build has no GatherBlockQuantized kernel at all, so rewrite the 4-bit embedding
    lookup with standard ops: Gather packed uint8 rows and their block scales, split each byte into
    (low, high) nibbles with integer division, subtract the zero point 8, multiply by the scale."""
    from onnx import helper, numpy_helper
    int4_gather_to_uint8(model)
    inits = {t.name: t for t in model.graph.initializer}
    out_nodes, n_conv = [], 0
    for n in model.graph.node:
        if n.op_type != "GatherBlockQuantized":
            out_nodes.append(n)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        assert attrs.get("gather_axis", 0) == 0 and attrs.get("quantize_axis", 1) == 1
        bs = attrs["block_size"]
        W, ids, S = n.input[0], n.input[1], n.input[2]
        rows, half = inits[W].dims
        cols = half * 2
        p = n.name.replace("/", "_") + "_dq"
        def c(name, arr):
            model.graph.initializer.append(numpy_helper.from_array(np.array(arr), p + name))
            return p + name
        k16, k8 = c("_16", np.int32(16)), c("_8", np.float32(8))
        shp_pairs, shp_blocks = c("_shp_pairs", np.array([0, 0, half, 1], np.int64)), c("_shp_blocks", np.array([0, 0, cols // bs, bs], np.int64))
        shp_out, ax = c("_shp_out", np.array([0, 0, cols], np.int64)), c("_ax", np.array([-1], np.int64))
        def mk(op, ins, out, **kw):
            out_nodes.append(helper.make_node(op, ins, [p + out], name=p + out, **kw))
            return p + out
        g = mk("Gather", [W, ids], "_g", axis=0)                     # [B, T, half] uint8
        gi = mk("Cast", [g], "_gi", to=TensorProto.INT32)
        hi = mk("Div", [gi, k16], "_hi")
        lo = mk("Sub", [gi, mk("Mul", [hi, k16], "_hi16")], "_lo")
        lo4 = mk("Reshape", [lo, shp_pairs], "_lo4")
        hi4 = mk("Reshape", [hi, shp_pairs], "_hi4")
        pairs = mk("Concat", [lo4, hi4], "_pairs", axis=-1)          # low nibble = even column
        q = mk("Sub", [mk("Cast", [pairs], "_qf", to=TensorProto.FLOAT), k8], "_q")
        qb = mk("Reshape", [q, shp_blocks], "_qb")                    # [B, T, cols/bs, bs]
        sc = mk("Unsqueeze", [mk("Gather", [S, ids], "_s", axis=0), ax], "_s4")
        deq = mk("Mul", [qb, sc], "_deq")
        out_nodes.append(helper.make_node("Reshape", [deq, shp_out], [n.output[0]], name=p + "_out"))
        n_conv += 1
    del model.graph.node[:]
    model.graph.node.extend(out_nodes)
    return n_conv


def quantize(src: Path, symmetric: bool, head_bits: int, keep: list = None) -> onnx.ModelProto:
    """4-bit block-32 weights for the transformer; the output projection (lm_head) at head_bits.
    Greedy decoding takes the argmax over a 262k vocabulary, so lm_head is the most rounding-sensitive."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    model = onnx.load(str(src), load_external_data=True)
    if keep is not None:
        restrict_lm_head(model, keep)   # small fp32 head; excluded from quantization below
    elif head_bits == 8:
        split_lm_head(model, 2)   # 8-bit lm_head is 168 MB; two halves fit the 90 MB file limit
    head = [n.name for n in model.graph.node if n.op_type in ("MatMul", "Gather") and "lm_head" in n.name]
    kw = dict(block_size=32, is_symmetric=symmetric, op_types_to_quantize=("MatMul", "Gather"),
              quant_axes=(("MatMul", 0), ("Gather", 1)))
    q = MatMulNBitsQuantizer(model, bits=4, nodes_to_exclude=head if head_bits != 4 else None, **kw)
    q.process()
    model = q.model.model
    if keep is None and head_bits != 4 and head:
        q = MatMulNBitsQuantizer(model, bits=head_bits, nodes_to_include=head, **kw)
        q.process()
        model = q.model.model
    print("GatherBlockQuantized -> standard ops:", gather_q4_to_standard_ops(model))
    return model


def save_chunked(model: onnx.ModelProto, out_dir: Path, name: str) -> int:
    """Move initializers > 1 KB into chunk files under CHUNK_BYTES each; return the number of chunks."""
    chunks, cur, offset = [], None, 0
    for t in model.graph.initializer:
        raw = t.raw_data if t.HasField("raw_data") else onnx.numpy_helper.to_array(t).tobytes()
        if len(raw) < 1024:
            continue
        if cur is None or offset + len(raw) > CHUNK_BYTES:
            fname = f"{name}_data" if not chunks else f"{name}_data_{len(chunks)}"
            cur = open(out_dir / fname, "wb")
            chunks.append((fname, cur))
            offset = 0
        cur.write(raw)
        for f in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data", "string_data"):
            t.ClearField(f)
        t.ClearField("raw_data")
        t.data_location = TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (("location", chunks[-1][0]), ("offset", str(offset)), ("length", str(len(raw)))):
            e = t.external_data.add()
            e.key, e.value = k, v
        offset += len(raw)
    for _, fh in chunks:
        fh.close()
    onnx.save(model, str(out_dir / name))
    return len(chunks)


def verify_ort(out_dir: Path, name: str, facts_path: Path, n: int):
    """Greedy-decode with ORT + KV cache on real eval facts and run the verifier."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE))
    from note_rules import verify

    tok = AutoTokenizer.from_pretrained(out_dir)
    sess = ort.InferenceSession(str(out_dir / name), providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in sess.get_inputs()}
    past_names = [k for k in ins if k.startswith("past_key_values")]
    cfg = json.loads((out_dir / "config.json").read_text())
    n_kv, hd = cfg["num_key_value_heads"], cfg["head_dim"]
    stop = {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}

    rows = [json.loads(l) for l in open(facts_path, encoding="utf-8")][:n]
    ok = 0
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        feed_ids = np.array([ids], dtype=np.int64)
        past = {k: np.zeros((1, n_kv, 0, hd), dtype=np.float32) for k in past_names}
        out_ids, total = [], len(ids)
        for _ in range(160):
            feed = {"input_ids": feed_ids, "attention_mask": np.ones((1, total), dtype=np.int64), **past}
            if "position_ids" in ins:
                feed["position_ids"] = np.arange(total - feed_ids.shape[1], total, dtype=np.int64)[None]
            outs = sess.run(None, feed)
            names = [o.name for o in sess.get_outputs()]
            logits = outs[names.index("logits")]
            nxt = int(logits[0, -1].argmax())
            if nxt in stop:
                break
            out_ids.append(nxt)
            past = {k: outs[names.index(k.replace("past_key_values", "present"))] for k in past_names}
            feed_ids = np.array([[nxt]], dtype=np.int64)
            total += 1
        note = tok.decode(out_ids, skip_special_tokens=True).strip()
        ok += verify(note, r["facts"])["ok"]
    return ok, len(rows), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "runs" / "gemma270m-v1"))
    ap.add_argument("--out", default=str(HERE.parent / "app" / "llm-model"))
    ap.add_argument("--facts", default=str(HERE / "data" / "real_facts.jsonl"))
    ap.add_argument("--verify-n", type=int, default=60)
    ap.add_argument("--asym", action="store_true", help="asymmetric 4-bit (zero points)")
    ap.add_argument("--head-bits", type=int, default=8, choices=[4, 8],
                    help="lm_head precision. 4-bit: 274 MB but 73%% verifier pass on held-out; 8-bit: 357 MB, 97%%")
    ap.add_argument("--full-vocab", action="store_true",
                    help="keep the 262k-token lm_head (slow in the browser: no fast 8-bit WASM kernel)")
    a = ap.parse_args()

    out = Path(a.out)
    if (out / "onnx").exists():
        shutil.rmtree(out / "onnx")
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    fp32 = Path(a.model) / "onnx_fp32"          # cached full-precision export (gitignored with llm/runs)
    if not (fp32 / "model.onnx").exists():
        exe = Path(sys.executable).parent / "optimum-cli"
        subprocess.run([str(exe), "export", "onnx", "--model", a.model, "--task", "text-generation-with-past", str(fp32)], check=True)
    for f in fp32.iterdir():
        if f.suffix in (".json", ".jinja", ".model") or f.name.startswith("tokenizer"):
            shutil.copy2(f, out / f.name)
    keep = None if a.full_vocab else build_note_vocab(fp32)
    head_desc = f"{a.head_bits}-bit full vocab" if keep is None else f"fp32 over {len(keep)} note tokens"
    print(f"quantizing: 4-bit block 32 ({'asym' if a.asym else 'sym'}), lm_head {head_desc}")
    qm = quantize(fp32 / "model.onnx", not a.asym, a.head_bits, keep)
    n_chunks = save_chunked(qm, out / "onnx", "model_q4.onnx")

    # transformers.js reads the chat template from tokenizer_config.json, not chat_template.jinja
    tc_path, jinja = out / "tokenizer_config.json", out / "chat_template.jinja"
    if jinja.exists():
        tc = json.loads(tc_path.read_text(encoding="utf-8"))
        tc["chat_template"] = jinja.read_text(encoding="utf-8")
        tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg = json.loads((out / "config.json").read_text())
    cfg["transformers.js_config"] = {"use_external_data_format": {"model_q4.onnx": n_chunks}, "dtype": "q4"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted((out / "onnx").iterdir())}
    print(json.dumps({"chunks": n_chunks, "files_mb": sizes, "total_mb": round(sum(sizes.values()), 1)}, indent=2))

    # Verify the exact shipped files. ORT resolves external data next to the .onnx file, and the
    # tokenizer/config live one level up, so copy them beside it temporarily.
    tmp = out / "onnx"
    copied = [shutil.copy2(f, tmp / f.name) for f in out.iterdir() if f.is_file()]
    try:
        ok, n, last = verify_ort(tmp, "model_q4.onnx", Path(a.facts), a.verify_n)
    finally:
        for c in copied:
            Path(c).unlink()
    res = {"quant": {"asym": a.asym, "lm_head": head_desc}, "facts": Path(a.facts).name, "q4_verifier_pass": ok, "n": n, "pass_rate": round(ok / n, 4), "example": last}
    (HERE / "results" / "q4_ort_verify.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
