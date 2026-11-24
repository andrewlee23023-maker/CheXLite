// Clinician note: facts from model outputs, a templated note, and a deterministic faithfulness
// verifier. The LLM note is shown only if verify() passes; otherwise the template is shown.
// llm/note_rules.py is the Python twin of this file, and llm/rule_vectors.json keeps them in sync.

export const DISPLAY = {
  "Atelectasis": "Atelectasis",
  "Cardiomegaly": "Cardiomegaly",
  "Consolidation": "Consolidation",
  "Edema": "Edema",
  "Pleural Effusion": "Pleural effusion",
};

const SYNONYMS = {
  "Atelectasis": [/atelecta\w*/i, /\bcollapse\w*/i],
  "Cardiomegaly": [/cardiomegaly/i, /enlarged heart/i, /heart (is |appears )?enlarged/i, /cardiac enlargement/i, /enlarged cardiac/i],
  "Consolidation": [/consolidat\w*/i],
  "Edema": [/\boe?dema\w*/i, /fluid in the lungs/i],
  "Pleural Effusion": [/effusion\w*/i, /pleural fluid/i, /fluid around the lung/i],
};

const FORBIDDEN = /\b(pneumothorax|pneumonia|nodules?|mass(es)?|tumou?rs?|cancers?|malignan\w*|tuberculosis|TB|fractures?|covid\w*|lesions?|infection\w*|emphysema|fibrosis)\b/i;
const NEGATION = /\b(no|not|none|without|unnecessary|isn't|aren't|don't|doesn't|didn't|nothing)\b/i;

export const REGIONS = [];
for (const lvl of ["upper", "middle", "lower"]) {
  REGIONS.push(`both ${lvl} zones`);  // listed first: "both lower zones" must not match "lower zone" rules
  for (const side of ["right", "left"]) REGIONS.push(`${side} ${lvl} zone`);
  REGIONS.push(`central ${lvl} chest`);
}

// Strongest part of a [h*w] cam (values 0..1) -> named zone. Vertical level from the weighted
// centroid; side from how the activation mass splits across image thirds, so bilateral activation
// reads as "both" rather than averaging to "central". Assumes a standard frontal film, with the
// patient's right on the image's left.
export function camRegion(cam, h, w) {
  let sy = 0, s = 0, mr = 0, mc = 0, ml = 0;
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    const v = cam[i * w + j];
    if (v < 0.5) continue;
    const x = (j + 0.5) / w;
    sy += v * (i + 0.5) / h; s += v;
    if (x < 1 / 3) mr += v; else if (x > 2 / 3) ml += v; else mc += v;
  }
  if (s === 0) return null;
  const y = sy / s;
  const lvl = y < 1 / 3 ? "upper" : y < 2 / 3 ? "middle" : "lower";
  if (mr >= 0.3 * s && ml >= 0.3 * s) return `both ${lvl} zones`;
  if (mc >= mr && mc >= ml) return `central ${lvl} chest`;
  return mr > ml ? `right ${lvl} zone` : `left ${lvl} zone`;
}

export function buildFacts(probs, cams, meta) {
  const [h, w] = meta.cam_size;
  const findings = meta.labels.map((label, k) => {
    const p = probs[k], thr = meta.flag_thresholds[label];
    return { label, name: DISPLAY[label], p: round2(p), thr: round2(thr), flagged: p >= thr,
             region: camRegion(cams.subarray(k * h * w, (k + 1) * h * w), h, w) };
  });
  const any = Math.max(...probs);
  return { refer: any >= meta.refer_threshold, any_score: round2(any), refer_threshold: round2(meta.refer_threshold), findings };
}

const round2 = (x) => Math.round(x * 100) / 100;

export function templateNote(f) {
  const flagged = f.findings.filter((x) => x.flagged);
  const clear = f.findings.filter((x) => !x.flagged);
  const out = [];
  if (f.refer) out.push("Refer to a radiologist: the screening score is above the referral threshold.");
  else out.push("No referral suggested: no finding passed the screening threshold. This does not rule out disease.");
  for (const x of flagged) {
    out.push(`${x.name} flagged (score ${x.p.toFixed(2)})${x.region ? `: check the ${x.region}` : ""}.`);
  }
  if (f.refer && flagged.length === 0) {
    out.push("No single finding passed its own threshold, but the combined score did, so a radiologist read is still advised.");
  }
  if (clear.length) out.push(`Not flagged: ${clear.map((x) => x.name.toLowerCase()).join(", ")}.`);
  out.push("This screens for five findings only. It is not a diagnosis and does not replace clinical judgment.");
  return out.join(" ");
}

const sentences = (t) => t.split(/(?<=[.!?])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
const mentions = (s, label) => SYNONYMS[label].some((re) => re.test(s));

export function verify(note, f) {
  const problems = [];
  const ss = sentences(note);
  const words = note.trim().split(/\s+/).length;
  if (words > 120) problems.push(`too long (${words} words)`);

  // R1 referral statement must match the decision
  const referS = ss.filter((s) => /\brefer\w*/i.test(s));
  const positive = referS.filter((s) => !NEGATION.test(s));
  const negative = referS.filter((s) => NEGATION.test(s));
  if (f.refer && positive.length === 0) problems.push("missing referral recommendation");
  if (!f.refer && positive.length > 0) problems.push("recommends referral but model did not");
  if (!f.refer && negative.length === 0) problems.push("missing no-referral statement");

  // R2 every flagged finding stated positively; unflagged findings only under negation
  for (const x of f.findings) {
    const ms = ss.filter((s) => mentions(s, x.label));
    if (x.flagged && !ms.some((s) => !NEGATION.test(s))) problems.push(`flagged ${x.label} not reported`);
    if (!x.flagged && ms.some((s) => !NEGATION.test(s))) problems.push(`unflagged ${x.label} reported as present`);
  }

  // R3 regions: only regions of flagged findings, and in that finding's sentence
  const flaggedRegions = new Set(f.findings.filter((x) => x.flagged && x.region).map((x) => x.region));
  for (const s of ss) {
    for (const r of REGIONS) {
      if (!s.toLowerCase().includes(r)) continue;
      const owners = f.findings.filter((x) => x.flagged && x.region === r && mentions(s, x.label));
      if (!flaggedRegions.has(r) || owners.length === 0) problems.push(`region "${r}" not supported`);
    }
  }

  // R4 numbers must match model outputs
  const allowed = [f.any_score, f.refer_threshold, ...f.findings.flatMap((x) => [x.p, x.thr])];
  for (const m of note.matchAll(/\b0?\.\d+\b|\b1\.0+\b/g)) {
    const v = parseFloat(m[0]);
    if (!allowed.some((a) => Math.abs(a - v) <= 0.011)) problems.push(`number ${m[0]} not from model`);
  }
  for (const m of note.matchAll(/(\d{1,3})\s?%/g)) {
    const v = parseInt(m[1], 10) / 100;
    if (!allowed.some((a) => Math.abs(a - v) <= 0.011)) problems.push(`percentage ${m[0]} not from model`);
  }

  // R5 no conditions outside the five the model screens for
  const fb = note.match(FORBIDDEN);
  if (fb) problems.push(`mentions "${fb[0]}", which the model does not assess`);

  return { ok: problems.length === 0, problems };
}
