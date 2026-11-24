# Loop notes (maintained by the orchestrator)

## Status (after 5 log rows, 2026-11-24)
Best is 0.7986 (CI 0.7935-0.8046): mobilenetv3_large_100 at 224px with Cardiomegaly U-ignore.
Every result so far sits inside the baseline CI of about ±0.005, so dev differences this small are
mostly noise. Prefer ideas that could plausibly move the score by more than 0.005.
Counters: 2 non-improving iterations in a row, 1 kept improvement that fell inside the previous CI.

## What helps
- Cardiomegaly uncertain zeros -> ignore: 0.7956 -> 0.7986, KEPT, but inside the baseline CI.

## What hurts / no effect
- Pleural Effusion uncertain ones -> ignore: 0.7961. U-ones looks right for Effusion.
- img_size 224 -> 320: 0.7990, essentially flat, not kept. With the fixed budget, the larger input does not pay off.
- BCE pos_weight clip(neg/pos, 5): 0.7956, flat to worse. Atelectasis AUROC is still the weakest label at 0.686.

## Untried ideas
1. `frontal_only: true` (existing config flag). The worst false alarms are all-negative studies with
   Effusion/Edema p of about 0.9 or higher, possibly from lateral views. Dev still includes laterals.
   Check with the auditor that filtering *training* views is not "dropping samples".
2. Consolidation uncertain zeros -> ignore, mirroring the kept Cardiomegaly change. Expect a gain inside the CI.
3. Atelectasis is the bottleneck label at 0.686. Try label-specific ideas there, e.g. U-ignore or U-zeros for Atelectasis.
4. Bigger levers inside the budget: LR schedule or warmup, stronger augmentation, or a different small
   backbone (fp32 checkpoint < 25 MB; the current one is 17 MB).

## Infra (resolved)
Iterations 7-9 of an earlier run were blocked by Bash permission denials for the venv python and git.
This is now fixed: training, eval and ratchet all run headless. The old STOP file is gone.
