# STATUS

Last updated: 2026-08-27 (live Kaggle session)

## 2026-08-27 — v6c adopted (full-58 gold OOF 0.7023 > v6b 0.6895); v6d refinement testing
- **FULL 5-fold gold OOF (all 58 experts): v6b 0.6895 → v6c 0.7023 (+0.013).** ADOPT v6c. ACL 0.501→0.604, MCL 0.601→0.667, Synovitis +0.066, Contusion +0.080, Medial OA +0.043.
- But dropping **Lateral OA** fills (borderline precision 0.545) backfired: Lateral OA 0.714→0.609 (-0.104); also Lat Meniscus -0.067, Effusion -0.055 (representation shift).
- **v6d** = drop ONLY {MCL, ACL} coin-flip fills, KEEP Lateral OA. Gate passes (prec 0.712, rec 0.680). Uploaded `girishbose/rsna-knee-weak-v6d`; testing fold0+1 (`train-b-fold{0,1}-weak-v6d`).
- Honest state: gold OOF ~0.70, approaching but not clearing the 0.72 submit bar; far from ~0.94 top. Label lever giving diminishing per-iteration gains (+0.013). Still NO submit.
- Method that works: audit each label's LLM-fill precision vs 58 gold; drop only the truly coin-flip (<0.51) fills; keep the rest.


## 2026-08-24 BREAKTHROUGH — v6b labels pass the supervision gate
Connected to Kaggle with the real competition data + your active kernels. Key facts:
- **Repo/Kaggle split:** GitHub `main` is a stale mirror. Your real current code lives in the `girishbose/rsna-knee-code` Kaggle dataset (has `consensus_labels.py`, `label_query.py`, updated trainer, notebooks 15–22). Reconcile from Kaggle, not this repo.
- **v6 (constrained Qwen) COMPLETED and fixed parse** (parse_rate **1.0** vs v5 0.795), recall **0.34→0.72**, but combined 58-expert precision **0.6826** missed the 0.69 gate.
- **Fix found and verified against the 58 gold:** pre-registered rule "drop LLM fills for any label with fill-precision < 0.5" selects exactly **MCL** (fill prec 0.231). Result: combined precision **0.7029 ≥ 0.69**, recall **0.690**, parse **1.0**, coverage **27,698** cells → **GATE PASSES** (first label recipe to pass after v4/v5/v6).
- **Candidate produced + uploaded:** `weak_labels_v6b_candidate.csv` (4,307/4,407 studies, 28,044 known cells) → Kaggle dataset `girishbose/rsna-knee-weak-v6b`.
- **Training launched (Kaggle GPU):** `girishbose/train-b-fold0-weak-v6b` — frozen DINOv2-B fold0, identical to notebook 30 except `--weak-csv` = v6b. Gate: must beat frozen-B weak_v1 **0.759**.
- Policy encoded here as `src/rsna_knee/text/fill_policy.py` (+ tests). To reproduce the recipe in the real pipeline: build the v6 skeleton+raw as in `notebooks/21_consensus_labels.py`, then apply `unreliable_fill_labels`/`drop_fills` before `combine_skeleton_and_fills`.

### Result so far — v6b beats BOTH folds (win transfers)
- fold0 v6b **0.7700** > 0.759 (+0.011); fold1 v6b **0.7387** > 0.732 (+0.007). Monotonic, no collapse.
- Caveat: val scored vs each run's own labels (v6b covers 4307 vs 2749), so favorable but not a fully clean A/B; per-fold gold cross-check is tiny. Vetted signal = passed label gate (prec 0.703/rec 0.690).

### DECISIVE full 5-fold OOF (all 58 experts) — 2026-08-26
- **OOF vs expert gold = 0.6895** (all 58). vs v6b weak labels 0.7508 (confounded; overstates by ~0.06).
- Fold scores (vs v6b labels): f0 0.770, f1 0.739, f2 0.763, f3 0.766, f4 0.736.
- Per-label vs gold: **ACL 0.501 (CHANCE), Fracture 0.556, MCL 0.601, Contusion 0.611**, Synovitis 0.700, Med Men 0.709, PF OA 0.712, Lat OA 0.714, Baker's 0.726, Lat Men 0.733, **Effusion 0.850, Medial OA 0.860**.

### Verdict: v6b labels KEPT; DO NOT SUBMIT (gold-OOF 0.69 < 0.72 floor, vs ~0.94 top)
v6b was a legitimate, fully-gated supervision win (beat label gate + all 5 folds vs weak labels), but the true gold-OOF is **0.69** — the supervision lever alone did not close the gap. The bottleneck is now clearly **image signal for specific labels**, above all **ACL at chance (0.50)** while Medial OA/Effusion work — so the model is blind to ACL despite clean labels. This is an image-pipeline problem, not labels.

### ANSWER: ACL was label noise, and v6c fixes it (2026-08-26)
- v6c fold0+1 vs v6b fold0+1 on 24 gold experts: macro **0.6894 → 0.7358**. **ACL 0.452 → 0.770 (+0.319)**, MCL 0.381→0.698, Med Men +0.121, Medial OA +0.118. (Effusion dipped 0.90→0.67 — likely n=24 variance; confirm on full 5-fold.)
- So ACL/MCL were **poisoned by coin-flip LLM fills**, NOT image-blind. Cleaning them (keyword-only) recovers detection. This is the strongest lever found.
- v6c fold scores vs its own labels: f0 0.768, f1 0.749. Rolling to full 5-fold (`train-b-fold{2,3,4}-weak-v6c`) to confirm full-58 gold OOF beats v6b's 0.6895.

### (resolved) ACL diagnostic — label vs image
- Ruled OUT missing data: **100% of studies have a sagittal series** (train_series.csv). So ACL=0.50 is not missing-plane.
- Found: v6 **ACL LLM-fill precision was exactly 0.50** (coin-flip) → ~half the added ACL positives are wrong. v6b kept them (rule was strictly <0.5).
- **v6c** = drop LLM fills with precision <0.55 → {MCL, ACL, Lateral OA}, keep keyword skeleton. Gate PASSES (prec 0.724, rec 0.634, coverage 23,143). ACL positives 657→291 (cleaner). Uploaded `girishbose/rsna-knee-weak-v6c`.
- **Running:** `train-b-fold{0,1}-weak-v6c`. Decides: if ACL gold-AUC rises off 0.50 → it was label noise; if it stays ~0.50 → frozen DINOv2 can't see ACL (image lever needed).

### Next lever (image side; only if v6c does NOT fix ACL)
1. Higher-res / more-slice cache with guaranteed sagittal ACL mid-slices; per-label plane routing.
2. Then a scoped cache/plane experiment targeting ACL+Fracture+Contusion (sagittal coverage, higher res, more slices) — the only labels with real macro headroom.
3. Fold the MCL-drop rule into `consensus_labels.py` so v6b regenerates deterministically.
4. No submit/final until gold-OOF is well above 0.72.

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
