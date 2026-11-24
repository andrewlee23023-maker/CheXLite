# ChexLite

An offline chest X-ray second look for clinics without a radiologist. A phone-sized image model
(MobileNetV3, 17 MB) flags five findings, highlights where to look, and recommends whether to refer
the study to a radiologist. An optional 270M-parameter language model, running on the device, turns
that output into a short note. Every note is checked against the model output before it is shown.
Everything runs in the browser with no server, and it keeps working offline after the first visit.

> Status: work in progress for the AI for Medicine Global Student Hackathon (due 2026-12-04).
> Numbers marked _TBD_ are filled in by the final run.

## Problem and who has it
Many clinics that take chest X-rays have no radiologist on site. Films are read by a GP, clinical
officer or nurse, or wait days for a remote read. The question at the point of care is usually not
"what is the diagnosis" but **"does this film need a radiologist, and where should I look?"**

ChexLite answers that question for five findings: atelectasis, cardiomegaly, consolidation, edema and
pleural effusion (the CheXpert competition labels). It is a triage aid, not a diagnosis, and a
radiologist can always be consulted later. It is built for phone-level hardware with no connectivity:
the image model is 17 MB and runs on the CPU in the browser (latency: _TBD_).

## What the clinician sees
1. **Decision:** "Refer to a radiologist" or "No referral suggested", with the any-finding score and
   threshold. If a finding is flagged but the overall score stays below the referral threshold, the
   title says "Take a closer look".
2. **Findings:** a score bar for each label, with the flag threshold at the bar's midpoint.
3. **Where to look:** a Grad-CAM heatmap per finding, computed inside the ONNX graph, plus a plain
   region name ("left lower zone").
4. **Note:** a templated note by default. Opt in to on-device AI notes (one-time ~360 MB download), and
   the app shows the AI note **only if** the deterministic verifier finds it consistent with the model
   output. Otherwise the template stays.

## Method
**Image model.** `mobilenetv3_large_100` (ImageNet-pretrained, timm), 224 px, 5 sigmoid outputs, trained
on CheXpert-v1.0-small. Uncertain labels follow the CheXpert paper's per-label policy (U-ones for
atelectasis, edema and effusion; U-zeros for consolidation), except cardiomegaly, where uncertain labels
are ignored. The experiment loop found that change (below). Studies are scored by the maximum over their
images. Export: ONNX, ~17 MB fp32, with an analytic Grad-CAM head checked against autograd
(max probability difference 4e-7).

**Self-improving experiment loop.** An Opus orchestrator runs one experiment per iteration, with Haiku
subagents: an analyst proposes hypotheses, an experimenter edits `src/` and trains, and an auditor
checks the diff for leakage or rule-breaking. `scripts/ratchet.py` evaluates on the dev split and keeps
the commit only if mean AUROC beats the best by at least 0.002. Otherwise it `git reset`s. Each
experiment uses the same frozen compute budget (`eval/budget.yaml`: 40k images, 2 epochs), so gains come
from ideas, not from training longer. The evaluation code is frozen: tag `eval-frozen`, sha256 manifest,
tamper check before every run, and deny rules that stop agents editing `eval/` or touching valid/test.

| Experiment (dev, 40k-image budget) | Mean AUROC [95% CI] | Kept |
|---|---|---|
| Baseline, MobileNetV3 224 px | 0.7956 [0.7901, 0.8011] | yes |
| Cardiomegaly uncertain: zeros → ignore | **0.7986** [0.7935, 0.8046] | yes |
| Pleural effusion uncertain: ones → ignore | 0.7961 | no |
| Image size 224 → 320 | 0.7990 | no (+0.0004, within noise) |
| BCE pos_weight | 0.7956 | no |
| Backbone → EfficientNet-B0 | 0.8003 | no (+0.0017 < margin, CI overlaps) |
| Photometric jitter | 0.7988 | no |
| Atelectasis uncertain: ones → ignore | 0.7958 | no |

Six non-improving experiments in a row triggered the loop's stop rule. The full log, including every
reverted idea, is `experiments/log.tsv` on the loop branch, where the git history alternates `exp:` and
`log:` commits. The final model is the best config retrained once at the full budget (all 212k training
images, 5 epochs, 53 min on one 8 GB GPU): dev mean AUROC **0.8168** [0.8117, 0.8223], up from 0.7986
for the same config at the loop budget.

**Note model.** Gemma 3 270M (instruction-tuned), LoRA fine-tuned to write a note from the classifier's
structured output only (scores, thresholds, refer decision, region names). It never sees the image.
- *Training data:* 3,000 template and paraphrase notes, plus 1,454 Claude (Opus) notes from 1,500
  requested. All were filtered by the same verifier the app uses (Claude's pass rate was 96.9%,
  costing $3.71).
- *SFT:* 2 epochs, loss on note tokens only. Validation loss went from 3.97 to 0.23.
- *GRPO:* on top of SFT. The reward is the verifier (+1 for pass, −0.25 per violated rule), with a
  KL penalty against the SFT model. Prompts come from a rebalanced synthetic sampler covering the cases
  SFT got wrong: 4–5 flags, and flagged but not referred.
- *Export:* ONNX, 4-bit weights (MatMulNBits, block 32) with an 8-bit output layer, split into files
  under 90 MB so GitHub Pages can serve them. It runs in transformers.js on WASM.

**Faithfulness verifier** (`app/note.js`, twin in `llm/note_rules.py`; parity-tested on 2,473 vectors,
0 mismatches). A note passes only if:
- the referral decision matches;
- every flagged finding is mentioned and no unflagged finding is asserted (negation-aware);
- regions match the heatmap regions;
- every number appears in the model output;
- it names nothing outside the five findings;
- it has at most 120 words.

On deliberately corrupted notes it catches 100% of most corruption types, 95% of wrong percentages,
and 93% of invented numbers.

## Evaluation
| Set | Labels | Size | Used for |
|---|---|---|---|
| dev (5% of CheXpert train patients, seed 1337, patient-level split) | NLP-extracted | 11,224 images / 3,227 patients | experiment loop, provisional thresholds |
| CheXpert valid | 3-radiologist majority | 234 images (200 studies) | final numbers only, **run once** (`export/final_valid.py`) |
| CheXlocalize test | radiologist consensus | 500 | not used: we did not obtain the test set in time |

- **Primary metric:** mean AUROC over the 5 findings, study level, with a bootstrap 95% CI.
- **Clinical metric:** the "any finding" referral operating point, targeting 95% sensitivity, reporting
  referral rate, specificity and **missed studies per 1,000**. The harm that matters most is a positive
  study that is not referred.
- **No peeking:** every threshold the app ships (referral and the 5 flag thresholds) was chosen on dev
  before valid was scored, then applied to valid unchanged. Valid was predicted exactly once.

**Final results, CheXpert valid** (200 studies, 128 with at least one finding; MobileNetV3, 16.9 MB ONNX):

| | Value [95% CI] |
|---|---|
| Mean AUROC, 5 findings | **0.865** [0.839, 0.889] |
| Per finding | Edema 0.940 · Pleural effusion 0.920 · Consolidation 0.886 · Atelectasis 0.812 · Cardiomegaly 0.765 |
| Referral sensitivity (shipped threshold, chosen on dev) | **92.2%** [86.2, 95.7]: 118 of 128 positive studies referred |
| Specificity | 65.3% (47 of 72 negative studies not referred) |
| Referral rate | 71.5% |
| Missed studies per 1,000 | **50** (10 of 200) |
| For comparison only: threshold chosen on valid itself | 95.3% sensitivity, 55.6% specificity, 77% referred, 30 missed / 1,000 (in-sample, optimistic) |

The dev-chosen threshold fell short of its 95% target on radiologist labels (92.2%, though the CI
includes 95%). Dev labels come from report text and valid labels from radiologists, and the two disagree.
We report the shortfall rather than retune on valid. A real deployment should recalibrate on a small
labeled local sample.

Per-finding flags (shipped thresholds, targeting 80% sensitivity on dev) on valid: sensitivity
consolidation 94%, edema 88%, atelectasis 87%, pleural effusion 72%, **cardiomegaly 48%**. The
cardiomegaly flag misses half of the cardiomegaly cases at its threshold. Most of those studies are
still referred through the overall score, but the "where to look" pointer for the heart is unreliable.
Full numbers: `experiments_results/final_valid.json`.

Reference point, not a target: the CheXpert paper (Irvin et al. 2019) reports three radiologists' operating
points on these same valid labels.

**Note model faithfulness** (share of notes the app would actually show; Wilson 95% CI). Real
classifier outputs come from frontal dev images the classifier never trained on.

| Note writer | Set | Verifier pass |
|---|---|---|
| Template | any | 100% by construction |
| Gemma 270M, no fine-tune (rules in the prompt) | real outputs, 400 | 0.0% [0, 0.95] |
| Gemma 270M, SFT | real outputs, 400 (inspected during development) | 92.0% [88.9, 94.3] |
| Gemma 270M, SFT | **held-out** 400 (different dev images, never inspected) | 90.0% [86.7, 92.6] |
| Gemma 270M, SFT + GRPO | real outputs, 400 (inspected) | 100% [99.0, 100] |
| Gemma 270M, SFT + GRPO | **held-out** 400 | **100% [99.0, 100]** |
| Shipped ONNX: 4-bit body, 4-bit lm_head (274 MB) | held-out, first 60 | 73.3% |
| Shipped ONNX: 4-bit body, 8-bit lm_head (357 MB) | held-out, 400 | _TBD_ |

How we got here, honestly: the SFT model's failures on the first real set were concentrated in cases the
synthetic sampler rarely produced (4–5 flags: 25% of real outputs vs 5% of synthetic; flagged but not
referred). We changed only the GRPO prompt sampler in response, then measured on a second set of 400 dev
images that had not been looked at. GRPO's cost is style diversity: notes are faithful but formulaic
(3 distinct opening sentences across 40 sampled notes). Quantizing the 262k-vocabulary output layer to
4 bits broke faithfulness (73%), so the shipped model keeps it at 8 bits. Asymmetric 4-bit was worse (68%).

## Who benefits, and who could be harmed
- **Benefits:** a clinician without radiology support gets a consistent second look, a pointer to where
  to look, and a referral prompt tuned to miss few positives.
- **False reassurance** is the main harm. On valid, 10 of 128 positive studies (about 1 in 13) were
  not referred at the shipped threshold. The app says so, and it says it screens for five findings only. It
  is silent on pneumothorax, nodules, TB, fractures, lines and tubes, and everything else.
- **Over-referral:** a high-sensitivity operating point means many referrals, which costs radiologist time.
- **Automation bias:** the heatmap shows where the model looked, not proof of a finding. The note is
  generated from the model's own output, so it can never be more correct than the classifier.
- **Population shift:** CheXpert comes from one US academic hospital (Stanford). Film quality, patient
  mix and disease prevalence (e.g. TB) in the target clinics differ. Performance there is unknown until
  measured locally.

## Limitations
- Training labels are NLP-extracted from reports and noisy. Only the 200-study valid set has radiologist
  labels, so the final CIs are wide.
- The final numbers are on CheXpert valid (200 studies), run once after all development. There is no
  external test set, and the CIs are wide.
- The cardiomegaly flag is weak (48% sensitivity at its threshold on valid).
- The LLM faithfulness sets were built from the classifier's earlier (loop-budget) checkpoint. The
  note model only sees structured output, so this measures the same task, but the score distribution
  of the final classifier differs slightly.
- Heatmaps are coarse (7×7 Grad-CAM) and not validated against radiologist segmentations; that would
  need CheXlocalize.
- Phone photos of films (CheXphoto) are not evaluated yet.
- The LLM note adds no diagnostic information. It only rephrases the classifier output, and the
  verifier checks consistency, not clinical correctness.
- The note model is a ~360 MB one-time download, too large for some connections, which is why
  it is opt-in.

## Reproduce
```bash
pip install -r requirements.txt
# Download CheXpert-v1.0-small (Stanford AIMI) into data/chexpert/
python eval/split.py                          # patient-level split + eval/MANIFEST.json
python src/train.py --full                    # final model
python export/export_onnx.py && python export/calibrate.py --split valid
python eval/evaluate.py --split valid
# Note model
python llm/make_pairs.py && python llm/finetune.py && python llm/grpo.py
python llm/real_facts.py --n 400 --seed 1 --exclude llm/data/real_facts.jsonl --out llm/data/real_facts_heldout.jsonl
python llm/eval_llm.py --model llm/runs/gemma270m-grpo --facts llm/data/real_facts_heldout.jsonl
PYTHONNOUSERSITE=1 .venv-export/Scripts/python llm/export_browser.py --model llm/runs/gemma270m-grpo
# App: serve app/ over http(s) and open index.html
```

## Another month
CheXlocalize test (localization quality of the heatmaps and the final held-out numbers), CheXphoto
robustness, int8 image model, a reader study with clinicians, and local recalibration on a small labeled
sample from a target clinic.

## Attribution
- CheXpert: Irvin et al., AAAI 2019, Stanford AIMI (research use agreement). No CheXpert images or labels
  are in this repository; trained weights are published for the demo.
- CheXlocalize: Saporta et al., Nature Machine Intelligence 2022. CheXphoto: Phillips et al., 2020.
- Gemma 3 270M: Google, used under the Gemma Terms of Use.
- Synthetic training notes partly written by Claude (Anthropic) and filtered by our verifier.
- timm, PyTorch, ONNX Runtime and ONNX Runtime Web (MIT), transformers.js (Apache-2.0); see `app/vendor/NOTICE.md`.
