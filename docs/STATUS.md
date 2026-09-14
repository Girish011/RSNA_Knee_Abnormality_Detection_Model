# STATUS

Last updated: 2026-09-15

## Phase
**Greenfield Imaging Playbook**. Blog Posts 01–03 done.
Coding: **gf_v0 5-fold OOF running** (true OOF gold-58). Floor recipe unchanged: 3×12×224.

## Done
- Post 01–03 (`site/`)
- gf_v0 fold0 all-58 gold **0.7281** (not true OOF; model saw some gold in train)
- Volume #1/#2 both KILLED
- Launched `girishbose/gf-baseline-v0-5fold` (T4; seed 42; folds 0–4 sequential)

## Active design (gf_baseline_v0)
- Cache `cache_gf_v0` (3×12×224); weak_v1; frozen DINOv2-S; seed 42
- Ruler for this run: **true OOF full-58 gold macro AUC** (each gold study scored by its holdout fold)
- Gold per fold: 13 / 11 / 10 / 12 / 12

## Gold-58 comparison (so far)
| Run | Weak | Gold-58 | Notes |
|---|---|---|---|
| gf_v0 fold0 model→all58 | 0.718 | **0.7281** | optimistic (train leak on some gold) |
| gf_v0 5-fold OOF | — | — | **running** |
| gf_v1 24-slice | 0.723 | 0.7089 | KILL |
| gf_v2 336px | 0.704 | 0.6925 | KILL |

## Next 3 actions
1. User watches `girishbose/gf-baseline-v0-5fold`; when done, agent pulls true OOF gold + weak OOF.
2. Treat that OOF gold as the honest v0 floor for future keep/kill.
3. No LB until a later recipe beats this OOF gold by ≥0.005.

## Kaggle artifacts
- Cache: `girishbose/rsna-knee-cache-gf-v0`
- Meta (seed trainer): `girishbose/rsna-knee-gf-v0-meta`
- **5-fold kernel:** `girishbose/gf-baseline-v0-5fold`
- Competition: `rsna-knee-abnormality-detection`

## Do not
- **Poll / background-watch Kaggle kernels** (user reports when jobs finish)
- Reports at test time; KneeCoT; LLM APIs in submit
- Adopt killed volume caches; LB before OOF win
- Confuse fold0-all58 gold with true OOF gold
