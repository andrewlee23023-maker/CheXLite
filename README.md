# ChexLite

An offline chest X-ray second look for clinics without a radiologist. A phone-sized model
(MobileNet-class, ~5–15 MB) flags five findings, highlights regions to check, and recommends
whether to refer the study to a radiologist. It runs in the browser on a phone with no server.

> Status: work in progress for the AI for Medicine Global Student Hackathon (due 2026-12-04).

## Problem and who has it
_TODO_

## Method
_TODO: model, uncertain-label policy, training budget, self-improving experiment loop (see below)._

## Evaluation
| Set | Labels | Studies | Used for |
|---|---|---|---|
| dev (5% of CheXpert train patients, seed 1337) | NLP-extracted | ~3k | experiment loop only |
| CheXpert valid | 3-radiologist majority | 200 | thresholds, model selection by human |
| CheXlocalize test | radiologist consensus | 500 | final numbers, run once (`experiments/test_runs.log`) |
| CheXphoto | phone photos of films | _TODO_ | robustness under clinic conditions |

Primary metric: mean AUROC over 5 findings with a bootstrap 95% CI.
Clinical metric: referral operating point at 95% sensitivity for "any finding"
(threshold set on valid, applied unchanged to test). Reports referral rate, specificity,
and **missed studies per 1,000**.
Reference: published CheXpert radiologist operating points. Baseline: DenseNet-121, same recipe.

_TODO results table_

## Who benefits, and who could be harmed
_TODO_

## Limitations
_TODO_

## Reproduce
```bash
pip install -r requirements.txt
# 1. Download CheXpert-v1.0-small (Stanford AIMI) and set CHEXPERT_ROOT to its parent folder
python eval/split.py                     # writes eval/splits + eval/MANIFEST.json
python src/train.py --full               # final model
python eval/evaluate.py --split valid
CHEX_ALLOW_TEST=1 python eval/evaluate.py --split test
```

## Experiment loop
`experiments/log.tsv` records every experiment, including reverted ones. The git history alternates
`exp:` and `log:` commits. See `CLAUDE.md` for the protocol and `.claude/agents/` for the subagents.

## Another month
_TODO_

## Attribution
- CheXpert: Irvin et al., AAAI 2019, Stanford AIMI (research use agreement)
- CheXlocalize: Saporta et al., Nature Machine Intelligence 2022
- CheXphoto: Phillips et al., 2020
- timm (Ross Wightman), PyTorch, torchvision; ImageNet-pretrained weights via timm
