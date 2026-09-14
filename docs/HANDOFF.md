# Handoff — greenfield campaign (2026-09-14)

Paste into a **new chat**. Source of truth: this file + `docs/STATUS.md` + `docs/SERIES.md`
+ latest entries in `docs/experiments.md`.

## Read first
1. `docs/STATUS.md`
2. `docs/SERIES.md`
3. `configs/gf_baseline_v0.yaml` (still the floor)
4. Tail of `docs/experiments.md`

## Where we are
- **gf_v0 floor:** gold-58 ≈ **0.728** (Post 03 drafted).
- User chose volume iterate. **gf_v1 (24 slices) KILLED:** gold **0.7089** (Δ -0.019 vs v0).
- Weak-val on v1 looked OK (~0.723) while gold fell — another reminder weak is smoke only.
- Active cache remains `cache_gf_v0` (3×12×224). Do not ship 24-slice.

## Immediate next task
**Volume iterate #2: resolution 224→336**, same 12 slices + same v0 picks, frozen-S, weak_v1, fold0.
Keep if gold-58 ≥ **0.7331**.

## Key paths
| Item | Location |
|---|---|
| Active floor config | `configs/gf_baseline_v0.yaml` |
| Killed v1 config | `configs/gf_baseline_v1.yaml` |
| v0 cache | `/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0` |
| v1 local results | `outputs/kaggle_download/gf-baseline-v1-fold0/gf_baseline_v1/` |

## Do not
- Resume old v6c / cache_v1; KneeCoT; reports at inference
- Treat 24-slice as a win; unfreeze yet; LB before OOF win

## New-chat opener (copy/paste)
> Read `docs/STATUS.md`, `docs/HANDOFF.md`, and `docs/SERIES.md`.
> gf_v1 24-slice KILLED (gold 0.709 < v0 0.728). Next = resolution 224→336 with 12 slices / same picks. Keep threshold 0.7331. Post 04 later.
