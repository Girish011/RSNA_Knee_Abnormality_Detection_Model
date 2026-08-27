# STATUS

Last updated: 2026-08-27 (v8 KILLED — full-58 gold 0.6935 < v6c 0.7023)

## 2026-08-27 — v8 KILL; next lever: GCE on v6c (RUNNING)
- **yunus gap-fill screened out:** strict intersect adds 0 cells; naive yunus gap-fill adds +24,628 cells (mostly negatives) — do not train without a gate.
- **GCE A/B launched:** `train-b-fold{0,1}-gce-v6c` with dataset `girishbose/rsna-knee-loss-gce` (shadows BCE-only loss on stale code dataset). Same v6c labels, frozen-B 8ep. v2 pushed with patch attached (v1 may fail without dataset).
- Keep rule unchanged: full-58 gold vs v6c BCE **0.7023**, margin **0.005**.

## 2026-08-27 — v8 KILL (full-58 gold OOF)
- All 5 folds COMPLETE. Weak-val looked better (confounded): f0–4 = 0.785/0.783/0.781/0.767/0.742 vs v6c 0.768/0.749/0.762/0.764/0.721.
- **Full-58 gold: v6c 0.7023 → v8 0.6935 (Δ −0.0088) → KILL** (rule margin 0.005).
- Matched 4-fold had already warned (−0.024). Additive TR/EL gap-fill of v7∩Qwen onto dropped ACL/MCL/LatOA cells re-poisoned training.
- **Adopted labels remain v6c.** Calibrated public LB still **0.682** (fold0 probe). Projected LB ≈ gold-OOF − 0.02.
- Do not LB-probe v8. Do not continue TR/EL consensus gap-fills on coin-flip columns.

## 2026-08-27 — Auth restored; cross-check DONE; v8 candidate on Kaggle
- Kaggle auth working as `girishbose`. Submission API confirms probe **publicScore 0.682** (ref 55818692).
- **In-domain label cross-check** (soft→hard lo=0.2/hi=0.7):
  - vs **yunus**: **91.2%** agree where both commit (best external teacher signal)
  - vs **dread**: 80.9%; MCL agree only 38%
  - vs **barun** `pseudo_*`: useless (mass at 0.5)
- v8 candidate was uploaded + trained; **killed** (see above).
- **SECURITY:** token was pasted in chat — **rotate**.

## 2026-08-27 — LB PROBE RESULT (calibrated)
- Notebook `girishbose/rsna-knee-submit-v6c` **Version 1** Succeeded → **public LB = 0.682**.
- Recipe: images-only, frozen DINOv2-B **v6c fold0 only** (not 5-fold blend), cache_v1 3×12×224.
- **Calibration vs internal rulers:**
  | Ruler | Score | vs LB |
  |---|---|---|
  | Public LB (this probe) | **0.682** | — |
  | Full-58 gold OOF (5-fold v6c) | 0.7023 | +0.020 optimistic |
  | Weak-label OOF (~v6c) | ~0.75 | +0.07 overstates badly |
- **Verdict:** gold-OOF is the usable internal ruler (±~0.02 to public). Weak-label OOF is not. **0.682 ≪ 0.94 top** and below our old 0.72 “think about finals” floor → **do not select as final; do not burn more probes** until a full-58 gold OOF clearly beats 0.7023 by ≥0.005 *and* a projected LB (OOF−0.02) clears a deliberate bar.
- Caveat: single-fold submit may slightly understate a 5-fold blend; gap to top is still ~0.26 — label micro-tweaks will not close it alone.
- Remaining blocker for next work: **rotated `KAGGLE_API_TOKEN`** in this env (still missing).

## 2026-08-27 — Session: v8 consensus tooling + cross-check scripts (code-only)
- **Shipped (GitHub):** `rsna_knee.text.label_consensus` + `scripts/build_weak_labels_v8.py` + `scripts/crosscheck_labels.py`. Unit tests + CLI smoke passed.
- Still true: GitHub is a stale mirror vs `girishbose/rsna-knee-code`; fold new modules into that dataset before Kaggle train.

## 2026-08-27 — v6c adopted (full-58 gold OOF 0.7023 > v6b 0.6895); v6d refinement testing
- **FULL 5-fold gold OOF (all 58 experts): v6b 0.6895 → v6c 0.7023 (+0.013).** ADOPT v6c. ACL 0.501→0.604, MCL 0.601→0.667, Synovitis +0.066, Contusion +0.080, Medial OA +0.043.
- But dropping **Lateral OA** fills (borderline precision 0.545) backfired: Lateral OA 0.714→0.609 (-0.104); also Lat Meniscus -0.067, Effusion -0.055 (representation shift).
- **v6d** = drop ONLY {MCL, ACL} coin-flip fills, KEEP Lateral OA. Gate passes (prec 0.712, rec 0.680). Uploaded `girishbose/rsna-knee-weak-v6d`; testing fold0+1 (`train-b-fold{0,1}-weak-v6d`).
- Honest state: **public LB 0.682** (v6c fold0 probe); gold OOF ~0.70 (+0.02 vs LB); far from ~0.94 top. Label lever giving diminishing per-iteration gains (+0.013). Do not final the probe.
- Method that works: audit each label's LLM-fill precision vs 58 gold; drop only the truly coin-flip (<0.51) fills; keep the rest.

## 2026-08-27 (cont.) — Option #1: LB probe — DONE (public 0.682)
- Built + submitted `girishbose/rsna-knee-submit-v6c` Version 1: images-only, frozen DINOv2-B v6c fold0, cache_v1. **Public LB 0.682.** See top section for calibration.

## 2026-08-27 (cont.) — Option #2 external ruler = NEGATIVE (domain shift)
- KneeMRI (Croatia, 736 sag volumes, ACL 0/1/2, prevalence 24.8%) is accessible on Kaggle; MRNet is gated.
- Ran v6c fold0 on it (`knee-acl-ruler-v6c`): external ACL AUC **0.53** even after fixing the adapter (fluid/fat metadata, spread slices). ≈ chance.
- Verdict: external-image datasets are NOT a viable ruler (or training source) here — domain shift + our multi-series model structure. Matches "extra corpora ≈ 0".
- Real ruler options now: (a) ONE calibrated LB probe of v6c (direct competition-distribution signal), or (b) in-domain cross-check of our labels vs competitors' public RSNA-knee label sets (barun2104/dreaddevelopment/yunusgmsoy) — zero domain shift.

## 2026-08-27 (cont.) — STOP label micro-tuning: it's below the noise floor
- v6d (keep Lateral OA) fold0+1 gold = **0.7378** vs v6c **0.7358** — a tie.
- **Proof the per-label gold reads are noise:** v6c and v6d have IDENTICAL ACL+MCL labels (both drop those fills; only Lateral OA differs), yet ACL gold swings 0.770→0.596 and MCL 0.698→0.508 between them on the same 24 experts. Changing only Lateral OA moved ACL by 0.17. So the earlier "ACL fixed +0.32" was mostly noise.
- **Only stable signal = full-58 macro: v6b 0.6895 → v6c 0.7023 (+0.013).** Per-label attributions and sub-0.02 deltas at n≤58 are not resolvable.
- **Decision: ADOPT v6c. STOP further label micro-tuning** (v6d≈v6c within noise; not worth a full 5-fold / more GPU). Do NOT keep reading 2-fold gold — it drives false reversals (the documented top-team failure mode).
- Remaining gap to ~0.94 is NOT a label problem now; it needs the image/backbone lever (frozen DINOv2 + 3×12×224 cache is the likely ceiling). That is a large scoped effort; several image directions (MRI-CORE, 384 rank, unfreeze) were already killed. Needs a deliberate plan + user steer, not more autonomous label runs.


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
| Setup | Score |
|---|---|
| **Public LB (v6c fold0 probe)** | **0.682** ← calibrated |
| Full-58 gold OOF **v6c** (5-fold) | **0.7023** ← adopted |
| Full-58 gold OOF v8 (5-fold) | 0.6935 ← **KILL** |
| Full-58 gold OOF v6b (5-fold) | 0.6895 |
| Public LB top | ~0.942 |

## New tooling (2026-08-24, code-only, unit-tested; no scores yet)
- Robust losses: `rsna_knee.training.loss.masked_multilabel_loss` (bce|gce|sce, label smoothing, per-label pos_weight); trainer reads `loss.mode` (defaults = BCE parity).
- OOF discipline: `scripts/oof_report.py` + `rsna_knee.evaluation` (full-OOF macro AUC, per-label, bootstrap CI, keep/kill rule margin 0.005).
- Labels: ensembled NLI hypotheses + tested `merge_pseudo_labels`; notebook 14 now imports the tested package.
- Config: `configs/labels_v3_robust.yaml` (frozen-B + weak_v3 + GCE + smoothing).

## Next 3 actions
1. Wait for `train-b-fold{0,1}-gce-v6c` (retry v2 if v1 failed without loss patch) → folds 2–4 if fold0+1 promising.
2. Full-58 gold OOF vs v6c BCE **0.7023** — keep GCE only if Δ ≥ 0.005.
3. **Rotate** pasted Kaggle token; fold `fill_policy` + GCE loss into `rsna-knee-code` dataset (hygiene).

## Do not
- Select v6c fold0 (0.682) as a final; burn LB probes without a gold-OOF rule win
- Retry v8-style TR/EL gap-fills on ACL/MCL/LatOA; trust per-label / 2-fold gold at n≤58
- Reopen killed paths (unfreeze, expert head-FT, cache_v2, 384, MRI-CORE, external-image train, **v8**)
- Use reports at test time; commit secrets / weights / DICOMs
