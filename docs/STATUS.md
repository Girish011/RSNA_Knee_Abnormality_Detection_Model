# STATUS

Last updated: 2026-09-21

## Phase
**Greenfield Imaging Playbook**. Structural encoder stage 1 live.
Coding: **gf-mri-core-seed42-5fold RUNNING** (v2 after train.csv path fix).

## Active experiment
- Kernel: `girishbose/gf-mri-core-seed42-5fold` **v2** (T4)
- Meta: `girishbose/rsna-knee-gf-mri-core-meta` (now includes `data/raw/train.csv`)
- v1 ERROR: competition `train.csv` missing at `/kaggle/input/competitions/...` (assert at startup). Fixed via meta-bundled train + path resolver.
- Weights: `girishbose/mri-core-vitb-rsna-knee` (Apache-2.0)
- Cache: existing `rsna-knee-cache-gf-v0` (3×12×224; upsample → 384 at encode)
- Recipe: frozen MRI-CORE ViT-B, 768-d pool (SAM neck skipped), batch 1, chunk 8, seed 42, 5-fold true OOF
- Gate vs v0 seed42 **0.6144**: KILL <0.5944; PROMISING ≥0.6344; else INCONCLUSIVE → kill

## Closed this cycle
| Lever | Verdict |
|---|---|
| train-longer / lig3 / unfreeze / 24-slice | all KILL |

## Next 3 actions
1. User reports when MRI-CORE kernel finishes (do not poll).
2. Read `oof_gold58_metrics.json` verdict.
3. PROMISING → seeds 1337+2024; else DINOv2-B / aggregator redesign.

## Do not
- Poll Kaggle; reopen thin-recipe levers; treat 0.6144/0.728 as floors
- Reports at test time; KneeCoT
