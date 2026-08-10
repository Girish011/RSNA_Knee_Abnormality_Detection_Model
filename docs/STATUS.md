# STATUS

Last updated: 2026-08-09

## Phase
**Weak supervision v1** — EN+ES report extractor audited on 58 expert studies. Next: Kaggle DICOM cache + DINOv2-S baseline.

## Dataset facts
- Studies: **4407** | Series: **24371** | Expert-labeled: **58**
- Weak labels v1: `data/processed/weak_labels_v1.csv` (local, gitignored)
- Audit: `docs/audit/weak_label_vs_expert.csv` — macro F1 **~0.44**, precision **~0.69**, recall **~0.34** (conservative by design)
- Folds: `data/folds/folds_v1.csv`

## Best scores
| Split | Macro AUC | Notes |
|---|---|---|
| OOF | — | no image training yet |
| Public LB | — | no submission yet |

## Blockers
1. Full DICOMs on Kaggle only — need cache notebook
2. Prefer revoke/regenerate Kaggle token if still the exposed one
3. Weak labels still weak on PF OA / Fracture / Medial OA recall — iterate after baseline

## Next 3 actions
1. Bundle DINOv2-S weights as Kaggle Model/Dataset
2. Build resized series cache on Kaggle (`notebooks/02_build_cache`)
3. Train/submit DINOv2-S baseline (`configs/baseline_dinov2_s.yaml`)

## Repo
- Private GitHub backup: https://github.com/Girish011/RSNA_Knee_Abnormality_Detection_Model
- Active machine: this Mac

## Session log
- 2026-08-08: Scaffold, metadata, folds_v1, private GitHub.
- 2026-08-09: Continue on this Mac; keep transfer artifacts (useful). Weak labels EN+ES; macro F1 0.33→0.44 vs expert.
