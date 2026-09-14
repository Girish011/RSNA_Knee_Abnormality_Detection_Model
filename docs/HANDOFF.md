# Handoff — greenfield campaign (2026-09-15)

## Read first
1. `docs/STATUS.md`
2. `docs/SERIES.md`
3. `configs/gf_baseline_v0.yaml`
4. Tail of `docs/experiments.md`

## Where we are
- Volume iterates v1/v2 **KILLED**; floor recipe = gf_v0 3×12×224
- **5-fold OOF running:** `girishbose/gf-baseline-v0-5fold`
- Goal: true OOF gold-58 (honest floor). Prior 0.7281 was fold0→all58 (slightly optimistic)

## Immediate next (when user says the kernel is done)
1. Download outputs from `girishbose/gf-baseline-v0-5fold`
2. Read `oof_gold58_metrics.json` + `oof_weak_metrics.json`
3. Update STATUS / experiments; no LB yet

## Working rule (locked)
- **Do not poll or watch Kaggle jobs.** Push/launch, log the slug in STATUS/HANDOFF, wait for the user to report done or paste logs.
- Same rule lives in `.cursor/rules/rsna-knee.mdc`.

## New-chat opener
> Read STATUS/HANDOFF/SERIES. gf_v0 5-fold OOF is the active job (`gf-baseline-v0-5fold`). I will tell you when it finishes — do not poll Kaggle.
