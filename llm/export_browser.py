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


def quantize(src: Path, symmetric: bool, head_bits: int) -> onnx.ModelProto:
    """4-bit block-32 weights for the transformer; the output projection (lm_head) at head_bits.
    Greedy decoding takes the argmax over a 262k vocabulary, so lm_head is the most rounding-sensitive."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    model = onnx.load(str(src), load_external_data=True)
    if head_bits == 8:
        split_lm_head(model, 2)   # 8-bit lm_head is 168 MB; two halves fit the 90 MB file limit
    head = [n.name for n in model.graph.node if n.op_type == "MatMul" and "lm_head" in n.name]
    kw = dict(block_size=32, is_symmetric=symmetric, op_types_to_quantize=("MatMul", "Gather"),
              quant_axes=(("MatMul", 0), ("Gather", 1)))
    q = MatMulNBitsQuantizer(model, bits=4, nodes_to_exclude=head if head_bits != 4 else None, **kw)
    q.process()
    model = q.model.model
    if head_bits != 4 and head:
        q = MatMulNBitsQuantizer(model, bits=head_bits, nodes_to_include=head, **kw)
        q.process()
        model = q.model.model
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
    print(f"quantizing: 4-bit block 32 ({'asym' if a.asym else 'sym'}), lm_head {a.head_bits}-bit")
    qm = quantize(fp32 / "model.onnx", not a.asym, a.head_bits)
    n_chunks = save_chunked(qm, out / "onnx", "model_q4.onnx")

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
    res = {"quant": {"asym": a.asym, "head_bits": a.head_bits}, "facts": Path(a.facts).name, "q4_verifier_pass": ok, "n": n, "pass_rate": round(ok / n, 4), "example": last}
    (HERE / "results" / "q4_ort_verify.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
