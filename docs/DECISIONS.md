# DECISIONS

Format: date | decision | why | rejected

## 2026-08-08 — Backbone: Meta DINOv2
- **Decision:** Use public DINOv2 (`vits14` baseline/efficiency, `vitb14` main).
- **Why:** Already leading early Model Hub usage; competition allows public pretrained models; we compete on wrapper/data/labels not novel ViTs.
- **Rejected:** Train ViT from scratch; primary 3D ConvNet path (too slow/data-hungry for first iterations).

## 2026-08-08 — Reports train-only
- **Decision:** Reports used for weak labels / pretraining only; never at inference.
- **Why:** `test.csv` has no reports; leakage if we design multimodal test-time fusion.
- **Rejected:** Late-fusion multimodal inference depending on report text.

## 2026-08-08 — Dual finals
- **Decision:** Final #1 max-AUC ensemble; Final #2 distilled DINOv2-S efficiency specialist.
- **Why:** Efficiency prize requires eligible accurate+fast submission selected as a final.
- **Rejected:** Single submission hoping to win both; two near-identical slow ensembles.

## 2026-08-08 — Compute/storage
- **Decision:** Kaggle notebooks/datasets as primary data+train environment; local Mac for code/docs/tests only.
- **Why:** Full dataset 569 GB; local free disk ~129 GB.
- **Rejected:** Download full raw DICOM tree to laptop as training store.

## 2026-08-08 — Continuity via repo, not chat
- **Decision:** STATUS / DECISIONS / experiments / WEEKLY + Cursor rule are source of truth.
- **Why:** 2-month campaign; chat context rot is guaranteed.
- **Rejected:** Relying on long agent threads as project memory.

## 2026-08-08 — Kaggle-first data layout
- **Decision:** Keep full DICOM tree on Kaggle; locally download **CSV metadata only**; train/cache/submit on Kaggle notebooks.
- **Why:** Dataset is 569 GB; Mac has ~129 GB free.
- **Rejected:** Full local `train_series/` download.

## 2026-08-08 — Auth via access_token
- **Decision:** Use modern `~/.kaggle/access_token` (KGAT), not legacy `kaggle.json`.
- **Why:** Matches current Kaggle API settings UI; CLI 1.7+/2.x supports it.
- **Rejected:** Requiring only legacy kaggle.json in scripts.

## 2026-08-08 — Expert labels are tiny
- **Decision:** Treat report weak supervision as first-class (not optional); expert 58 studies are fine-tune / audit set.
- **Why:** Only **58 / 4407** studies have expert labels; all reports non-empty; multilingual (e.g. Spanish in sample).
- **Rejected:** Expert-only training as the main path.

## 2026-08-09 — Weak labels EN+ES v1
- **Decision:** Ship keyword weak labels with Spanish synonyms (LCA, Derrame, etc.); keep high precision / lower recall; confidence threshold 0.5.
- **Why:** English-only missed common ES reports (macro F1 ~0.33 → ~0.44 on expert audit).
- **Rejected:** Jumping straight to paid LLM labeling before a measurable keyword baseline.

## 2026-08-11 — Kill weak_labels_v2 for main train
- **Decision:** Train on `weak_labels_v1` until a new label recipe beats fold0 ablate.
- **Why:** Matched fold0 ablate (`pos_weight=1`): v1 **0.725** > v2 **0.718**; earlier v2+pw1.5 was 0.707.
- **Rejected:** Shipping multilingual v2 as default; full 5-fold on v2.

## 2026-08-15 — Report labels v3 via multilingual zero-shot NLI (train-only)
- **Decision:** Generate `weak_labels_v3.csv` with public mDeBERTa-XNLI + v1 fallback + expert override; never use text at inference.
- **Why:** Keyword v2 and expert head-FT failed; FR-heavy reports need a multilingual reader, not more image tweaks.
- **Rejected:** Paid LLM as the first v3 path; training a text model on 58 experts only.

## 2026-08-15 — Kill expert-only head fine-tune as default
- **Decision:** Do not ship expert head-FT; keep frozen-B (or S) weak_v1 checkpoints.
- **Why:** B FT: 0.7591 → 0.7601 then drop to 0.745 on 45 gold studies.
- **Rejected:** Longer expert FT / unfreezing last blocks on this tiny set.

## 2026-08-15 — Keep cache_v1 (3×12); kill cache_v2 as default
- **Decision:** Train on `cache_v1` until a new cache recipe beats fold0 0.764.
- **Why:** Frozen S on 4×16 cache_v2 peaked at **0.738** (below 0.764 / 0.759).
- **Rejected:** 5-fold or B-first on cache_v2 after a losing S smoke.

## 2026-08-24 — Strategy reframe: noisy teacher + evaluation discipline
- **Decision:** Treat the campaign as *learning from a noisy teacher graded by a different exam*, not an architecture race. Prioritize (1) label quality (v3 NLI), (2) robust/noisy losses, (3) pre-registered multi-fold OOF decisions. De-prioritize bigger backbones / more ensemble members / TTA / extra pretraining.
- **Why:** Competition confirmed live (2026 RSNA Knee Abnormality Detection AI Challenge; deadline 2026-10-22, winners in Nov — no writeups yet). Public signal from a top-15 team (LB 0.937): the hidden test is **graded by expert radiologists reading the images** while our train labels are noisy report-derived; "you are never optimising the thing you are scored on." They report bigger backbones / more ensemble members / TTA / extra pretraining "all measured, all worth roughly zero," the LB is noise-limited in the 3rd decimal, and the discipline that mattered was pre-registering the decision rule and reading multiple folds. Remaining open lever: "how you learn from a noisy teacher."
- **Rejected:** Spending GPU budget on architecture/TTA/ensemble scaling before exhausting label-quality and noisy-loss levers; trusting single-fold deltas.

## 2026-08-27 — v8 overlay is additive gap-fill (not overwrite)
- **Decision:** When applying v7∩Qwen consensus onto adopted v6c, **only fill cells where v6c is NaN** (`only_where_base_isna=True`). Do not overwrite trusted v6c commits. In practice this re-supplies ACL/MCL/Lateral OA on TR/EL studies where both extractors agree — the columns whose LLM fills v6c dropped.
- **Why:** Matches the “high-precision multilingual supervision” intent; measured delta was +495 cells with only ~7 accidental overwrites in the overwrite variant. Safer A/B vs v6c.
- **Rejected:** Blind overwrite of all consensus cells onto v6c.

## 2026-08-27 — LB calibration: gold-OOF ≈ public + 0.02 (v6c fold0 probe)
- **Decision:** Treat **full-58 gold OOF** as the primary internal ruler. Translate to expected public LB with **≈ −0.02** (measured: gold 0.7023 vs public **0.682** on fold0-only v6c probe). Ignore weak-label OOF for keep/kill (~0.75 overstated by ~0.07). Do not select the 0.682 probe as a final; do not spend more public submits until gold OOF beats 0.7023 by the pre-registered 0.005 margin *and* projected LB (OOF−0.02) clears an explicit bar.
- **Why:** First real competition-distribution signal. Confirms the 58-gold macro is slightly optimistic but directionally honest; weak-label val is not.
- **Rejected:** Calibrating on weak-label OOF; assuming fold0 LB equals 5-fold blend LB; chasing more probes to climb from 0.68 without a gold win.

## 2026-08-27 — v8 label recipe = v7∩Qwen consensus overlay (not yet measured)
- **Decision:** Next label candidate after adopted v6c is **v8**: keep v6c as base; replace cells with **v7 ∩ constrained-Qwen** where both commit and agree (else abstain). Prefer restricting the overlay to Turkish/Greek studies (`detect_language`), since that is where v7 adds independent signal (88%/77% agree with Qwen). Keep/kill only on **full-58 gold OOF** vs v6c **0.7023** (margin 0.005 rule); do not trust per-label or 2-fold gold.
- **Why:** v7 alone is not clearly better than v6c (Qwen already covers TR/EL); agreement cells are the high-precision subset. Encoded in `label_consensus.py` + `scripts/build_weak_labels_v8.py`.
- **Rejected:** Retraining on v7 alone; further v6d-style micro-tuning on the 58; treating competitor-label cross-check as a substitute for LB calibration (it is complementary).

## 2026-08-24 — v6b: per-label LLM-fill reliability gate (drop MCL)
- **Decision:** When combining constrained-Qwen fills with the keyword skeleton, drop LLM fills for any label whose measured 58-expert fill-precision is < 0.5 (pre-registered), keeping the keyword skeleton for that label. On v6 this is exactly MCL and lifts combined precision 0.6826 → 0.7029 (passes the 0.69 gate) with recall 0.690 and parse 1.0.
- **Why:** MCL LLM fills are consistently poisonous (v5 prec 0.27, v6 prec 0.231); reports describe MCL inconsistently. This is the minimal principled intervention, not a scan for the best label. First supervision recipe to pass the gate after v4/v5/v6.
- **Rejected:** Relaxing the 0.69/0.98 gates; hand-picking labels; dropping additional labels (Contusion/Lateral OA) — unnecessary and costs recall/coverage.
- **Note:** GitHub repo is a stale mirror; the authoritative code is the `girishbose/rsna-knee-code` Kaggle dataset. Fold this rule into its `consensus_labels.py`.

## 2026-08-24 — Robust losses + pre-registered OOF rule (tooling)
- **Decision:** Add opt-in robust multilabel losses (GCE, Symmetric CE, two-sided label smoothing, per-label pos_weight) defaulting to exact BCE parity; add `rsna_knee.evaluation` + `scripts/oof_report.py` for full 5-fold OOF macro AUC with study-level bootstrap CIs and a keep/kill/inconclusive rule (default margin 0.005). Extract the v3 label layering into unit-tested `merge_pseudo_labels` and ensemble multiple NLI hypotheses per label.
- **Why:** Directly operationalizes the two levers above; all logic unit-tested (BCE-mode == legacy loss; end-to-end CPU model step).
- **Rejected:** Rewriting the validated frozen trainer; changing defaults (frozen backbone, weak_v1) without an OOF-rule win.

## 2026-08-12 — No backbone unfreeze until frozen-B wins
- **Decision:** Default train recipe keeps DINOv2 backbone frozen for all epochs; unfreeze only as a deliberate later experiment with tiny LR / last-block-only.
- **Why:** B fold0 freeze→×0.05 LR unfreeze collapsed 0.729→0.615 (same as early S).
- **Rejected:** Staged unfreeze as the default main-track schedule.
