# STATUS

Last updated: 2026-08-24 (live Kaggle session)

## 2026-08-24 BREAKTHROUGH — v6b labels pass the supervision gate
Connected to Kaggle with the real competition data + your active kernels. Key facts:
- **Repo/Kaggle split:** GitHub `main` is a stale mirror. Your real current code lives in the `girishbose/rsna-knee-code` Kaggle dataset (has `consensus_labels.py`, `label_query.py`, updated trainer, notebooks 15–22). Reconcile from Kaggle, not this repo.
- **v6 (constrained Qwen) COMPLETED and fixed parse** (parse_rate **1.0** vs v5 0.795), recall **0.34→0.72**, but combined 58-expert precision **0.6826** missed the 0.69 gate.
- **Fix found and verified against the 58 gold:** pre-registered rule "drop LLM fills for any label with fill-precision < 0.5" selects exactly **MCL** (fill prec 0.231). Result: combined precision **0.7029 ≥ 0.69**, recall **0.690**, parse **1.0**, coverage **27,698** cells → **GATE PASSES** (first label recipe to pass after v4/v5/v6).
- **Candidate produced + uploaded:** `weak_labels_v6b_candidate.csv` (4,307/4,407 studies, 28,044 known cells) → Kaggle dataset `girishbose/rsna-knee-weak-v6b`.
- **Training launched (Kaggle GPU):** `girishbose/train-b-fold0-weak-v6b` — frozen DINOv2-B fold0, identical to notebook 30 except `--weak-csv` = v6b. Gate: must beat frozen-B weak_v1 **0.759**.
- Policy encoded here as `src/rsna_knee/text/fill_policy.py` (+ tests). To reproduce the recipe in the real pipeline: build the v6 skeleton+raw as in `notebooks/21_consensus_labels.py`, then apply `unreliable_fill_labels`/`drop_fills` before `combine_skeleton_and_fills`.

### Next actions
1. Watch `girishbose/train-b-fold0-weak-v6b`. If fold0 > **0.759** by a real margin → run fold1 (gate vs 0.732) before any submit.
2. If it does not beat 0.759, the labels improved but the frozen-B head can't exploit them → next lever is the head/training on v6b, not more label recipes.
3. Fold the MCL-drop rule into `consensus_labels.py` in the real code dataset so v6b regenerates deterministically.

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
