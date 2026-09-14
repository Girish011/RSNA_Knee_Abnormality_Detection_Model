# STATUS

Last updated: 2026-09-14

## Phase
**Greenfield Imaging Playbook**. Blog Posts 01–03 done.
Coding: **gf_v1 + gf_v2 volume iterates both KILLED**. Active floor = **gf_v0 gold-58 = 0.7281**.

## Done
- Post 01–03 (`site/`)
- gf_baseline_v0 fold0 + full-58 gold **0.7281**
- Volume #1 (24 slices): gold **0.7089** → KILL
- Volume #2 (336px): gold **0.6925** → KILL

## Active design (still gf_baseline_v0)
- 1 Sag + 1 Cor + 1 Ax; cache `cache_gf_v0` (3×12×224)
- Temporary teacher: `weak_labels_v1.csv`
- Frozen DINOv2-S; fold 0; no unfreeze
- Ruler: full-58 gold macro AUC, margin 0.005

## Gold-58 comparison
| Run | Weak-val best | Gold-58 macro | Verdict |
|---|---|---|---|
| gf_v0 (12×224) | 0.718 | **0.7281** | floor / KEEP |
| gf_v1 (24×224) | 0.723 | **0.7089** | KILL (Δ -0.019) |
| gf_v2 (12×336) | 0.704 | **0.6925** | KILL (Δ -0.036) |

## Next 3 actions
1. **Stop pure volume A/B** on this thin 3-series recipe (slices and res both lost).
2. Prefer next: **5-fold OOF on gf_v0** (stabilize gold floor) **or** Post 05-style teacher/labels (not more series yet).
3. No LB until an OOF win over v0 gold floor.

## Kaggle artifacts (slugs)
- Active floor cache: `girishbose/rsna-knee-cache-gf-v0`
- Active floor fold0: `girishbose/rsna-knee-gf-v0-fold0`
- Killed v1: `girishbose/gf-cache-v1`, `girishbose/gf-baseline-v1-fold0`
- Killed v2: `girishbose/gf-cache-v2`, `girishbose/gf-baseline-v2-fold0`
- Competition: `rsna-knee-abnormality-detection`

## Do not
- Reports at test time; KneeCoT; LLM APIs in submit
- Treat weak-val as the ship ruler
- Adopt killed 24-slice or 336 caches
- Burn more GPU on “just add volume” without a new hypothesis
- Resume old v6c / cache_v1 as the active path
- Push GPU scripts without `machine_shape: NvidiaTeslaT4`
