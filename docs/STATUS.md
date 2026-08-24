# STATUS

Last updated: 2026-08-24

## Phase
**Noisy-teacher lever, measured with discipline.** Competition is live (RSNA 2026 Knee; deadline 2026-10-22). Hidden test is expert-radiologist-graded on images while train labels are noisy report-derived — so OOF ~0.76 vs public ~0.94 is partly a *label/measurement* gap, not just capacity. External top-15 signal: bigger backbone / more ensemble / TTA / extra pretraining ≈ 0; LB noise-limited in 3rd decimal. Focus: (1) v3 NLI labels, (2) robust losses, (3) pre-registered multi-fold OOF. Do not submit until the full 5-fold OOF clears the rule.

## Best scores
| Setup | Fold0 best |
|---|---|
| **S + weak_v1 frozen, cache_v1** | **0.764** (ckpt missing) |
| **B + weak_v1 fully frozen, cache_v1** | **0.759** ← keep |
| B + expert head-FT | 0.760 then 0.745 (kill) |
| S + cache_v2 | 0.738 |
| Public LB top | ~0.942 |

## New tooling (2026-08-24, code-only, unit-tested; no scores yet)
- Robust losses: `rsna_knee.training.loss.masked_multilabel_loss` (bce|gce|sce, label smoothing, per-label pos_weight); trainer reads `loss.mode` (defaults = BCE parity).
- OOF discipline: `scripts/oof_report.py` + `rsna_knee.evaluation` (full-OOF macro AUC, per-label, bootstrap CI, keep/kill rule margin 0.005).
- Labels: ensembled NLI hypotheses + tested `merge_pseudo_labels`; notebook 14 now imports the tested package.
- Config: `configs/labels_v3_robust.yaml` (frozen-B + weak_v3 + GCE + smoothing).

## Next 3 actions
1. Kaggle: run `notebooks/14_pseudo_label_reports.py` (GPU T4x2, **Internet ON**) → `weak_labels_v3.csv`; check expert audit vs v1 (~F1 0.44).
2. Kaggle: 5-fold frozen-B with `configs/labels_v3_robust.yaml` (weak_v3 + GCE). Save per-fold `fold*_oof.csv`.
3. Locally: `python scripts/oof_report.py --oof <v3 folds> --baseline-oof <weak_v1 B folds> --targets data/folds/folds_v1.csv` → keep only on a rule win vs **0.759**.

## Do not
- Submit before an OOF-rule win; unfreeze; cache_v2 5-fold; weak_v2; ship FT weights
- Use reports at test time; chase 3rd-decimal single-fold deltas
