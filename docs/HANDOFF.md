# Handoff — greenfield campaign (2026-09-14)

Paste into a **new chat**. Source of truth: this file + `docs/STATUS.md` + `docs/SERIES.md`
+ latest entries in `docs/experiments.md`.

## Read first
1. `docs/STATUS.md`
2. `docs/SERIES.md`
3. `configs/gf_baseline_v0.yaml` (still the floor)
4. Tail of `docs/experiments.md`

## Where we are
- **gf_v0 floor:** gold-58 ≈ **0.728**
- **gf_v1 (24 slices) KILLED:** gold 0.709
- **gf_v2 (336px) KILLED:** gold **0.693** (Δ -0.036 vs v0); MCL 0.431
- xFormers warnings on Kaggle are harmless
- Stop pure volume A/B on the 3-series recipe

## Immediate next task (pick one)
1. **Multi-fold OOF on gf_v0** (same 12×224 cache) → stabler gold read before any LB
2. **Teacher/labels** (Post 05 track): better train-time report labels; never in submit

Keep threshold vs v0 remains gold ≥ **0.7331** for recipe changes.

## Do not
- Resume old v6*; KneeCoT; reports at inference
- Adopt 24-slice or 336 caches; LB before OOF win
- Blind “more volume” without a new hypothesis

## New-chat opener
> Read `docs/STATUS.md`, `docs/HANDOFF.md`, `docs/SERIES.md`.
> gf_v1 and gf_v2 volume KILLED; floor is gf_v0 gold 0.728. Next = multi-fold OOF on v0 OR labels track. Post 04 later.
