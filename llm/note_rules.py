"""Python twin of app/note.js: template note, prompt format and faithfulness verifier.
llm/test_parity.mjs checks both implementations against llm/rule_vectors.json."""
import math
import re

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
DISPLAY = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
           "Edema": "Edema", "Pleural Effusion": "Pleural effusion"}
SYNONYMS = {
    "Atelectasis": [r"atelecta\w*", r"\bcollapse\w*"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged heart", r"heart (is |appears )?enlarged", r"cardiac enlargement", r"enlarged cardiac"],
    "Consolidation": [r"consolidat\w*"],
    "Edema": [r"\bo?edema\w*", r"fluid in the lungs"],
    "Pleural Effusion": [r"effusion\w*", r"pleural fluid", r"fluid around the lung"],
}
SYN = {k: [re.compile(p, re.I) for p in v] for k, v in SYNONYMS.items()}
FORBIDDEN = re.compile(r"\b(pneumothorax|pneumonia|nodules?|mass(es)?|tumou?rs?|cancers?|malignan\w*|tuberculosis|TB|"
                       r"fractures?|covid\w*|lesions?|infection\w*|emphysema|fibrosis)\b", re.I)
NEGATION = re.compile(r"\b(no|not|none|without|unnecessary|isn't|aren't|don't|doesn't|didn't|nothing)\b", re.I)

REGIONS = []
for lvl in ["upper", "middle", "lower"]:
    REGIONS.append(f"both {lvl} zones")
    REGIONS += [f"right {lvl} zone", f"left {lvl} zone"]
    REGIONS.append(f"central {lvl} chest")


def cam_region(cam):
    """Python twin of camRegion() in app/note.js. cam: 2-D array (h, w) normalised to [0, 1]."""
    h, w = cam.shape
    sy = s = mr = mc = ml = 0.0
    for i in range(h):
        for j in range(w):
            v = float(cam[i, j])
            if v < 0.5:
                continue
            x = (j + 0.5) / w
            sy += v * (i + 0.5) / h
            s += v
            if x < 1 / 3:
                mr += v
            elif x > 2 / 3:
                ml += v
            else:
                mc += v
    if s == 0:
        return None
    y = sy / s
    lvl = "upper" if y < 1 / 3 else "middle" if y < 2 / 3 else "lower"
    if mr >= 0.3 * s and ml >= 0.3 * s:
        return f"both {lvl} zones"
    if mc >= mr and mc >= ml:
        return f"central {lvl} chest"
    return f"right {lvl} zone" if mr > ml else f"left {lvl} zone"


def build_facts(probs, cams, meta):
    """Python twin of buildFacts() in app/note.js. probs: [5], cams: [5, h, w]."""
    findings = []
    for k, label in enumerate(meta["labels"]):
        p, thr = float(probs[k]), float(meta["flag_thresholds"][label])
        findings.append({"label": label, "name": DISPLAY[label], "p": r2(p), "thr": r2(thr), "flagged": p >= thr,
                         "region": cam_region(cams[k])})
    any_score = float(max(probs))
    return {"refer": any_score >= meta["refer_threshold"], "any_score": r2(any_score),
            "refer_threshold": r2(meta["refer_threshold"]), "findings": findings}


def r2(x):
    # JS Math.round(x*100)/100 rounds half up; Python round() is banker's rounding
    return math.floor(x * 100 + 0.5) / 100


def fmt2(x):
    return f"{x:.2f}"


def template_note(f):
    flagged = [x for x in f["findings"] if x["flagged"]]
    clear = [x for x in f["findings"] if not x["flagged"]]
    out = []
    if f["refer"]:
        out.append("Refer to a radiologist: the screening score is above the referral threshold.")
    elif flagged:
        out.append("No referral suggested: the overall score is below the referral threshold, but some areas are worth a closer look.")
    else:
        out.append("No referral suggested: no finding passed the screening threshold. This does not rule out disease.")
    for x in flagged:
        where = f": check the {x['region']}" if x["region"] else ""
        out.append(f"{x['name']} flagged (score {fmt2(x['p'])}){where}.")
    if f["refer"] and not flagged:
        out.append("No single finding passed its own threshold, but the combined score did, so a radiologist read is still advised.")
    if clear:
        out.append("Not flagged: " + ", ".join(x["name"].lower() for x in clear) + ".")
    out.append("This screens for five findings only. It is not a diagnosis and does not replace clinical judgment.")
    return " ".join(out)


def facts_to_prompt(f):
    """Compact, fixed-format input the LLM sees (mirrored in app/llm.js)."""
    lines = [f"refer: {'yes' if f['refer'] else 'no'} (score {fmt2(f['any_score'])}, threshold {fmt2(f['refer_threshold'])})"]
    for x in f["findings"]:
        state = "FLAGGED" if x["flagged"] else "not flagged"
        reg = f", {x['region']}" if x["flagged"] and x["region"] else ""
        lines.append(f"{x['name']}: {fmt2(x['p'])} / {fmt2(x['thr'])} {state}{reg}")
    return "Model output:\n" + "\n".join(lines) + "\nWrite the clinician note."


def _sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]


def _mentions(s, label):
    return any(r.search(s) for r in SYN[label])


def verify(note, f):
    problems = []
    ss = _sentences(note)
    words = len(note.split())
    if words > 120:
        problems.append(f"too long ({words} words)")

    refer_s = [s for s in ss if re.search(r"\brefer\w*", s, re.I)]
    pos = [s for s in refer_s if not NEGATION.search(s)]
    neg = [s for s in refer_s if NEGATION.search(s)]
    if f["refer"] and not pos:
        problems.append("missing referral recommendation")
    if not f["refer"] and pos:
        problems.append("recommends referral but model did not")
    if not f["refer"] and not neg:
        problems.append("missing no-referral statement")

    for x in f["findings"]:
        ms = [s for s in ss if _mentions(s, x["label"])]
        if x["flagged"] and not any(not NEGATION.search(s) for s in ms):
            problems.append(f"flagged {x['label']} not reported")
        if not x["flagged"] and any(not NEGATION.search(s) for s in ms):
            problems.append(f"unflagged {x['label']} reported as present")

    flagged_regions = {x["region"] for x in f["findings"] if x["flagged"] and x["region"]}
    for s in ss:
        for r in REGIONS:
            if r not in s.lower():
                continue
            owners = [x for x in f["findings"] if x["flagged"] and x["region"] == r and _mentions(s, x["label"])]
            if r not in flagged_regions or not owners:
                problems.append(f'region "{r}" not supported')

    allowed = [f["any_score"], f["refer_threshold"]] + [v for x in f["findings"] for v in (x["p"], x["thr"])]
    for m in re.finditer(r"\b0?\.\d+\b|\b1\.0+\b", note):
        v = float(m.group(0))
        if not any(abs(a - v) <= 0.011 for a in allowed):
            problems.append(f"number {m.group(0)} not from model")
    for m in re.finditer(r"(\d{1,3})\s?%", note):
        v = int(m.group(1)) / 100
        if not any(abs(a - v) <= 0.011 for a in allowed):
            problems.append(f"percentage {m.group(0)} not from model")

    fb = FORBIDDEN.search(note)
    if fb:
        problems.append(f'mentions "{fb.group(0)}", which the model does not assess')
    return {"ok": not problems, "problems": problems}
