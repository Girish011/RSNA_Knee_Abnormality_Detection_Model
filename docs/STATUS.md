# STATUS

Last updated: 2026-09-10

## Phase
**Greenfield Imaging Playbook** (learn + implement from scratch).
Blog Posts 01–02 done. Coding Step B in progress.
Do not resume the old label/cache campaign as the active plan.

## Done
- Post 01 playbook + Post 02 EDA (site/)
- Step A: metadata audit (`notebooks/01_data_audits.py`)
- B.1–B.2: plane-covered series picker → `outputs/eda/gf_v0_series_picks.csv` (4407 studies × 3)
- B.3: `configs/gf_baseline_v0.yaml`

## Active design (gf_baseline_v0)
- Series: 1 Sagittal + 1 Coronal + 1 Axial (prefer fluid-sensitive)
- Cache plan: 3 × 12 slices × 224 (`cache_gf_v0`, not old cache_v1)
- Labels for v0: `data/processed/weak_labels_v1.csv` (temporary teacher)
- Model: frozen DINOv2-S, fold 0 first
- Ruler: full-58 gold macro AUC, margin 0.005; weak-val is smoke only

## Next 3 actions
1. B.4.1 local cache dry-run (`notebooks/03_cache_dry_run.py`)
2. B.4.2 build `cache_gf_v0` on Kaggle (DICOMs)
3. B.5 fold-0 train + log OOF / full-58 gold in `docs/experiments.md`

## Blockers
- Full DICOMs only on Kaggle (not on laptop)
- Multi-machine: git push/pull + this STATUS/HANDOFF before switching

## Do not
- Use reports at test time
- KneeCoT or other gated hospital-agreement datasets
- Call commercial LLMs from the submit notebook
- Treat old v6* / cache_v1 scores as the thing to continue