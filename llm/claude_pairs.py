"""Generate varied clinician notes with Claude, keeping only the ones that pass the verifier.

Facts come from the same sampler as make_pairs.py (different seed). Claude sees only the
structured facts and writes natural, varied notes; every note is checked by note_rules.verify()
and failures are dropped. The pass rate is itself a reported result: how faithful a frontier model
is under the same rules the fine-tuned Gemma 3 270M is held to.

Requires ANTHROPIC_API_KEY. The generated JSONL is committed-ready output for reproducibility
(re-running costs money and isn't deterministic).

    python llm/claude_pairs.py --n 1500 --seed 1 --out llm/data/claude_pairs.jsonl
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from make_pairs import sample_facts
from note_rules import facts_to_prompt, verify

HERE = Path(__file__).resolve().parent
MODEL = "claude-opus-5-5"
PER_REQUEST = 20

SYSTEM = """You write short notes for clinicians at clinics without a radiologist. A chest X-ray \
screening model has already run; you only turn its structured output into plain language.

Rules (a script checks every note; notes that break a rule are discarded):
- State the referral decision exactly as given: if refer is yes, recommend referral to a radiologist; \
if no, say referral is not suggested (use a negative word such as "no" or "not" in that sentence).
- Mention every FLAGGED finding as present-to-check, by name, and if a region is given, say to check \
that exact region phrase (e.g. "right lower zone") in the same sentence as that finding.
- Findings marked "not flagged" may only appear in a sentence containing "not" or "no".
- Never mention any condition other than: atelectasis, cardiomegaly, consolidation, edema, pleural effusion.
- Only use numbers that appear in the model output (you may omit numbers entirely). No percentages.
- Never mention a region for a finding that isn't flagged.
- Do not claim certainty or a diagnosis; end with a brief reminder that this is a screening aid.
- 2 to 5 sentences, under 80 words, plain clinical English. Vary wording and sentence order \
across notes: these are training examples, so avoid repeating the same template."""

SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "note": {"type": "string"}},
                "required": ["id", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["notes"],
    "additionalProperties": False,
}


def request_batch(client, batch):
    body = "\n\n".join(f"### id {i}\n{facts_to_prompt(f)}" for i, f in enumerate(batch))
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},  # SDK 0.97 predates the typed `fallbacks` param
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Write one note per model output below.\n\n{body}"}],
    )
    if resp.stop_reason == "refusal":
        return [], resp.usage, "refusal"
    text = next(b.text for b in resp.content if b.type == "text")
    notes = {n["id"]: n["note"] for n in json.loads(text)["notes"]}
    return [(batch[i], notes.get(i)) for i in range(len(batch))], resp.usage, resp.model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "data" / "claude_pairs.jsonl"))
    a = ap.parse_args()

    rng = random.Random(a.seed)
    facts = [sample_facts(rng) for _ in range(a.n)]
    batches = [facts[i:i + PER_REQUEST] for i in range(0, len(facts), PER_REQUEST)]
    client = anthropic.Anthropic(max_retries=4)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = total = missing = 0
    fails = {}
    tin = tout = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(request_batch, client, b) for b in batches]
        for fut in as_completed(futs):
            try:
                pairs, usage, served = fut.result()
            except anthropic.APIStatusError as e:
                print(f"batch failed: {e.status_code} {e.message}")
                continue
            tin += usage.input_tokens
            tout += usage.output_tokens
            for f, note in pairs:
                total += 1
                if not note:
                    missing += 1
                    continue
                v = verify(note, f)
                if v["ok"]:
                    kept += 1
                    fh.write(json.dumps({"facts": f, "prompt": facts_to_prompt(f), "note": note,
                                         "source": "claude", "model": served}) + "\n")
                else:
                    for p in v["problems"]:
                        key = p.split(" ")[0] + " " + (p.split(" ")[1] if len(p.split(" ")) > 1 else "")
                        fails[key] = fails.get(key, 0) + 1
            print(f"{total}/{a.n} notes, kept {kept} ({time.time() - t0:.0f}s)", flush=True)

    cost = tin / 1e6 * 4.0 + tout / 1e6 * 20.0  # claude-opus-5-5 list price, $/MTok in/out
    summary = {"requested": a.n, "returned": total - missing, "kept": kept,
               "verifier_pass_rate": round(kept / max(1, total - missing), 4),
               "failure_types": dict(sorted(fails.items(), key=lambda x: -x[1])),
               "input_tokens": tin, "output_tokens": tout, "est_cost_usd": round(cost, 2), "model": MODEL}
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
