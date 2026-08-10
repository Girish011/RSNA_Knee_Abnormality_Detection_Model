# STATUS

Last updated: 2026-08-10

## Phase
**Baseline prep** — cache/train scripts + DINOv2-S weights downloaded. Awaiting Kaggle Dataset uploads + cache build.

## Dataset facts
- Studies: **4407** | Expert-labeled: **58** | Folds: `data/folds/folds_v1.csv`
- Weak labels v1 audited (macro F1 ~0.44)
- DINOv2-S weights local: `data/external/dinov2/dinov2_vits14_pretrain.pth` (84MB, gitignored)
- Upload packs: `outputs/kaggle_code_upload/rsna-knee-code.zip`, `dinov2-vits14-rsna-knee.zip`

## Best scores
| Split | Macro AUC | Notes |
|---|---|---|
| OOF | — | waiting on Kaggle cache |
| Public LB | — | no submission yet |

## Blockers
1. Need you to upload the two zips as Kaggle Datasets (browser)
2. Run `02_build_cache.ipynb` on Kaggle (smoke LIMIT=50 then full)
3. Then `10_train_baseline.ipynb` GPU

## Next 3 actions
1. Upload `rsna-knee-code` + `dinov2-vits14-rsna-knee` datasets on Kaggle
2. Build `rsna-knee-cache-v1` via notebook 02
3. Train fold-0 smoke baseline; log OOF in experiments.md

## Repo
- https://github.com/Girish011/RSNA_Knee_Abnormality_Detection_Model

## Session log
- 2026-08-08/09: Scaffold, metadata, folds, weak labels EN+ES.
- 2026-08-10: Commit/push weak labels; added cache builder, cached dataset, train_baseline_fold, Kaggle notebooks/runbook; fetched DINOv2-S; packaged upload zips.
