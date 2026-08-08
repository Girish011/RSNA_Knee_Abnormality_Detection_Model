# STATUS

Last updated: 2026-08-08

## Phase
**Data audit (metadata)** — CSV metadata on disk; `folds_v1` frozen. Next: weak-label prototype + Kaggle DICOM cache/baseline.

## Dataset facts (from local CSVs)
- Studies: **4407**
- Series rows: **24371**
- Expert-labeled studies: **only ~58** (all 12 labels present together) → reports are the main supervision source
- Test example CSV: 3 studies (full test ~1300 on submit)
- Competition slug: `rsna-knee-abnormality-detection` (entered=True)

## Best scores
| Split | Macro AUC | Notes |
|---|---|---|
| OOF | — | no training yet |
| Public LB | — | no submission yet |
| Efficiency (est.) | — | no runtime yet |

## Active configs
- Main candidate: `configs/baseline_dinov2_s.yaml`
- Efficiency candidate: `configs/efficiency_student.yaml`
- Folds: `data/folds/folds_v1.csv` (882/882/881/881/881)

## Blockers
1. ~~Identity verification~~ done
2. ~~API auth~~ done via `~/.kaggle/access_token`
3. ~~Metadata download~~ done (~9 MB, not 569 GB)
4. Full DICOMs remain on Kaggle — need cache notebook next
5. **Security:** API token was pasted into terminal/chat earlier — **revoke & regenerate** on Kaggle settings

## Next 3 actions
1. Revoke exposed Kaggle token and write a new one to `~/.kaggle/access_token`
2. Prototype weak labels on all 4407 reports; audit accuracy on the 58 expert studies
3. On Kaggle: run DICOM spot-check + start resized cache build for baseline training

## Session log
- 2026-08-08: Foundation scaffold.
- 2026-08-08: Fixed auth script for `access_token`; downloaded CSVs; froze folds_v1; local audit shows 58/4407 expert labels.
