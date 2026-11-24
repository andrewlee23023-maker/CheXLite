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
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

HERE = Path(__file__).resolve().parent
CHUNK_BYTES = 90 * 1024 * 1024


def quantize(src: Path) -> onnx.ModelProto:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    model = onnx.load(str(src), load_external_data=True)
    q = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=True,
                             op_types_to_quantize=("MatMul", "Gather"), quant_axes=(("MatMul", 0), ("Gather", 1)))
    q.process()
    return q.model.model


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
    a = ap.parse_args()

    out = Path(a.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        exe = Path(sys.executable).parent / "optimum-cli"
        subprocess.run([str(exe), "export", "onnx", "--model", a.model, "--task", "text-generation-with-past", td], check=True)
        for f in Path(td).iterdir():
            if f.suffix in (".json", ".jinja", ".model") or f.name.startswith("tokenizer"):
                shutil.copy2(f, out / f.name)
        print("quantizing to 4-bit (MatMul + Gather, block 32)...")
        qm = quantize(Path(td) / "model.onnx")
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
    res = {"q4_verifier_pass": ok, "n": n, "pass_rate": round(ok / n, 4), "example": last}
    (HERE / "results" / "q4_ort_verify.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
