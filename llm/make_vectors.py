"""Build llm/rule_vectors.json: notes with Python verdicts that app/note.js must reproduce.
Includes valid template notes and targeted corruptions, each expected to fail a specific rule."""
import json
import random
from pathlib import Path

from note_rules import template_note, verify

HERE = Path(__file__).resolve().parent


def corruptions(f, note, rng):
    flagged = [x for x in f["findings"] if x["flagged"]]
    clear = [x for x in f["findings"] if not x["flagged"]]
    out = []
    if f["refer"]:
        out.append(("flip_refer", note.replace("Refer to a radiologist", "Do not refer to a radiologist", 1)))
    else:
        out.append(("flip_refer", "Refer to a radiologist now. " + note))
    if clear:
        out.append(("invent_finding", note + f" {clear[0]['name']} is present."))
    if flagged:
        x = flagged[0]
        out.append(("drop_finding", " ".join(s for s in note.split(". ") if x["name"] not in s)))
        out.append(("fake_number", note.replace(f"{x['p']:.2f}", f"{min(0.99, x['p'] + 0.2):.2f}", 1)))
        if x["region"]:
            other = "left upper zone" if x["region"] != "left upper zone" else "right lower zone"
            out.append(("wrong_region", note.replace(x["region"], other, 1)))
    out.append(("forbidden", note + " There may also be pneumonia."))
    out.append(("percent", note + " Overall risk is 73%."))
    return out


def main():
    rng = random.Random(1)
    rows = [json.loads(l) for l in (HERE / "data" / "template_pairs.jsonl").open(encoding="utf-8")][:300]
    vecs = []
    for r in rows:
        f = r["facts"]
        tn = template_note(f)
        vecs.append({"kind": "template", "facts": f, "note": tn, "expect_ok": verify(tn, f)["ok"], "template": tn})
        vecs.append({"kind": "paraphrase", "facts": f, "note": r["note"], "expect_ok": verify(r["note"], f)["ok"]})
        for kind, bad in corruptions(f, tn, rng):
            vecs.append({"kind": kind, "facts": f, "note": bad, "expect_ok": verify(bad, f)["ok"]})
    (HERE / "rule_vectors.json").write_text(json.dumps(vecs))
    by = {}
    for v in vecs:
        k = by.setdefault(v["kind"], [0, 0])
        k[0] += v["expect_ok"]
        k[1] += 1
    print("pass/total by kind:", json.dumps(by))


if __name__ == "__main__":
    main()
