"""Generate (model output -> clinician note) training pairs from templates with paraphrase.

Facts are sampled to cover the space the app produces: the real thresholds from
app/model/meta.json, every combination of flagged findings, every region, refer/no-refer, and the
"flagged but below referral" case. Each note is assembled from interchangeable phrasings and must
pass verify(). Any failure is a bug in the generator or the verifier, so the script aborts.

    python llm/make_pairs.py --n 6000 --seed 0 --out llm/data/template_pairs.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from note_rules import DISPLAY, LABELS, REGIONS, facts_to_prompt, fmt2, r2, verify

HERE = Path(__file__).resolve().parent
META = json.loads((HERE.parent / "app" / "model" / "meta.json").read_text())

REFER = [
    "Refer to a radiologist: the screening score is above the referral threshold.",
    "Recommend referral to a radiologist for a formal read.",
    "This study should be referred to a radiologist.",
    "Please refer this film to a radiologist for review.",
    "Referral to a radiologist is recommended (overall score {any} vs threshold {rt}).",
]
NO_REFER_CLEAR = [
    "No referral suggested: no finding passed the screening threshold. This does not rule out disease.",
    "No referral is suggested by the screen; nothing passed its threshold.",
    "The screen does not suggest referral. A normal screen does not rule out disease.",
    "Referral is not suggested: the overall score {any} is below the threshold of {rt}.",
]
NO_REFER_FLAGGED = [
    "No referral suggested: the overall score is below the referral threshold, but some areas are worth a closer look.",
    "Referral is not suggested by the overall score ({any} vs {rt}), but review the highlighted areas.",
    "The screen does not reach the referral threshold; still, take a closer look at the areas below.",
]
FLAG = [
    "{name} flagged (score {p}): check the {region}.",
    "{name} is flagged with a score of {p}; look closely at the {region}.",
    "Check the {region} for {lname} (score {p}, threshold {thr}).",
    "The model flags {lname} in the {region} (score {p}).",
    "Possible {lname}: review the {region}.",
]
FLAG_NOREGION = [
    "{name} flagged (score {p}).",
    "{name} is flagged with a score of {p}.",
    "Possible {lname} (score {p}, threshold {thr}).",
]
COMBINED_ONLY = [
    "No single finding passed its own threshold, but the combined score did, so a radiologist read is still advised.",
    "None of the five findings crossed its own threshold, yet the combined score is high enough that a specialist read is advised.",
]
CLEAR = [
    "Not flagged: {list}.",
    "No flag for {list}.",
    "The model did not flag {list}.",
]
CLOSER = [
    "This screens for five findings only. It is not a diagnosis and does not replace clinical judgment.",
    "This tool only screens for five findings and is not a diagnosis; use clinical judgment.",
    "Screening aid only, limited to five findings: it does not replace clinical judgment.",
    "Remember this is a five-finding screen, not a diagnosis.",
]


def sample_facts(rng):
    thrs = META["flag_thresholds"]
    rt = r2(META["refer_threshold"])
    mode = rng.random()
    want = {l: rng.random() < (0.15 if mode < 0.35 else 0.4) for l in LABELS}
    findings = []
    for l in LABELS:
        t = r2(thrs[l])
        if want[l]:
            p = r2(rng.uniform(t + 0.01, min(0.99, t + 0.6)))
        else:
            p = r2(rng.uniform(0.0, max(0.0, t - 0.01)))
        flagged = p >= t
        region = None
        if flagged:
            region = "central middle chest" if l == "Cardiomegaly" and rng.random() < 0.6 else rng.choice(REGIONS)
            if rng.random() < 0.05:
                region = None
        findings.append({"label": l, "name": DISPLAY[l], "p": p, "thr": t, "flagged": flagged, "region": region})
    any_score = max(x["p"] for x in findings)
    if rng.random() < 0.08:  # combined-only referral: no single flag but any score above refer threshold
        for x in findings:
            if x["flagged"]:
                x.update(p=r2(max(0.0, x["thr"] - 0.01)), flagged=False, region=None)
        any_score = r2(rng.uniform(rt, rt + 0.15))
    return {"refer": any_score >= rt, "any_score": any_score, "refer_threshold": rt, "findings": findings}


def write_note(f, rng):
    flagged = [x for x in f["findings"] if x["flagged"]]
    clear = [x for x in f["findings"] if not x["flagged"]]
    fill = {"any": fmt2(f["any_score"]), "rt": fmt2(f["refer_threshold"])}
    out = []
    if f["refer"]:
        out.append(rng.choice(REFER).format(**fill))
    elif flagged:
        out.append(rng.choice(NO_REFER_FLAGGED).format(**fill))
    else:
        out.append(rng.choice(NO_REFER_CLEAR).format(**fill))
    order = flagged[:]
    if rng.random() < 0.5:
        order.sort(key=lambda x: -x["p"] / max(x["thr"], 1e-6))
    for x in order:
        d = {"name": x["name"], "lname": x["name"].lower(), "p": fmt2(x["p"]), "thr": fmt2(x["thr"]), "region": x["region"]}
        s = rng.choice(FLAG if x["region"] else FLAG_NOREGION).format(**d)
        out.append(s[0].upper() + s[1:])
    if f["refer"] and not flagged:
        out.append(rng.choice(COMBINED_ONLY))
    if clear and rng.random() < 0.85:
        names = [x["name"].lower() for x in clear]
        lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        out.append(rng.choice(CLEAR).format(list=lst))
    out.append(rng.choice(CLOSER))
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "data" / "template_pairs.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"refer": 0, "flagged_no_refer": 0, "clear": 0, "combined_only": 0}
    with out.open("w", encoding="utf-8") as fh:
        for i in range(a.n):
            f = sample_facts(rng)
            note = write_note(f, rng)
            v = verify(note, f)
            if not v["ok"]:
                raise SystemExit(f"generator/verifier disagree on pair {i}: {v['problems']}\n{note}\n{json.dumps(f)}")
            nflag = sum(x["flagged"] for x in f["findings"])
            key = ("combined_only" if nflag == 0 else "refer") if f["refer"] else ("flagged_no_refer" if nflag else "clear")
            stats[key] += 1
            fh.write(json.dumps({"facts": f, "prompt": facts_to_prompt(f), "note": note, "source": "template"}) + "\n")
    print(json.dumps({"written": a.n, "out": str(out), "mix": stats}))


if __name__ == "__main__":
    main()
