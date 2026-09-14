# STATUS

Last updated: 2026-09-14

## Phase
**Greenfield Imaging Playbook**. Blog Posts 01–03 done.
Coding: **gf_baseline_v1 (24 slices) KILLED**. Active floor remains **gf_v0 gold-58 = 0.7281**.

## Done
- Post 01–03 (`site/`)
- gf_baseline_v0 fold0 + full-58 gold **0.7281** (weak-val smoke 0.718)
- Volume iterate #1: `gf_baseline_v1` 12→24 slices → gold **0.7089** → **KILL**

## Active design (still gf_baseline_v0)
- 1 Sag + 1 Cor + 1 Ax; cache `cache_gf_v0` (3×12×224)
- Temporary teacher: `weak_labels_v1.csv`
- Frozen DINOv2-S; fold 0; no unfreeze
- Ruler: full-58 gold macro AUC, margin 0.005

## Gold-58 comparison
| Run | Weak-val best | Gold-58 macro | Verdict |
|---|---|---|---|
| gf_v0 (12 slices) | 0.718 | **0.7281** | floor / KEEP |
| gf_v1 (24 slices) | 0.723 | **0.7089** | KILL (Δ -0.019) |

## Next 3 actions
1. Next volume lever: **resolution 224→336**, keep **12 slices** + same v0 series picks (single-axis).
2. Build `cache_gf_v2` + fold0 train+gold; keep if gold ≥ 0.7331.
3. Do not default to 24-slice cache; no LB until an OOF win over v0.

## Kaggle artifacts (slugs)
- v0 cache (active): `girishbose/rsna-knee-cache-gf-v0`
- v0 meta: `girishbose/rsna-knee-gf-v0-meta`
- v0 fold0: `girishbose/rsna-knee-gf-v0-fold0`
- v1 meta: `girishbose/rsna-knee-gf-v1-meta`
- v1 cache kernel (killed recipe): `girishbose/gf-cache-v1`
- v1 train kernel (killed): `girishbose/gf-baseline-v1-fold0`
- Competition: `rsna-knee-abnormality-detection`

## Do not
- Reports at test time; KneeCoT; LLM APIs in submit
- Treat weak-val as the ship ruler (v1 weak looked fine while gold fell)
- Adopt 24-slice as default after this kill
- Resume old v6c / cache_v1 as the active path
- Unfreeze backbone yet
- Push GPU scripts without `machine_shape: NvidiaTeslaT4`
