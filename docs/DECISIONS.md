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

## 2026-09-13 — Greenfield volume iterate #1 = more slices (not more series / not unfreeze)
- **Decision:** Next greenfield experiment after gf_v0 is `gf_baseline_v1`: **same series picks**, `n_slices` 12→**24**, image_size 224, frozen DINOv2-S, weak_v1, fold0. Keep if full-58 gold macro ≥ v0 0.7281 + 0.005.
- **Why:** v0 weak spots are ACL/MCL/Contusion; thin depth along each plane is the clearest single volume lever. Old cache_v2 (4×16) already failed on the prior stack, so do not jump to more series first. Unfreeze remains blocked by the 2026-08-12 decision.
- **Rejected:** Multi-fold-first before any volume change; 224→336 as the first volume lever; adding a 4th series; unfreeze-on-v1.

## 2026-09-14 — Kill gf_v1 24-slice; next try resolution 336
- **Decision:** **KILL** `gf_baseline_v1` / `cache_gf_v1` as default. Keep `gf_v0` (3×12×224) as the gold floor. Next single-axis volume experiment = **image_size 224→336**, keep **12 slices** and the same v0 series picks.
- **Why:** fold0 gold-58 fell 0.7281 → **0.7089** (Δ -0.019); MCL 0.451 (below chance). Weak-val (0.723) was misleading. Doubling slices without other changes hurt the ruler.
- **Rejected:** Adopting 24-slice despite weak-val; jumping to more series next; unfreeze.

## 2026-09-14 — Launch gf_v2 = 336px (same 12 slices / picks)
- **Decision:** Run `gf_baseline_v2` as the next greenfield volume A/B: same v0 picks, n_slices=12, image_size=**336**, frozen-S, weak_v1, fold0, seed 42. Keep if gold-58 ≥ 0.7331.
- **Why:** v1 kill leaves resolution as the unused single-axis volume lever; 336 is divisible by DINOv2 patch 14.
- **Rejected:** Retrying 24 slices; adding series; unfreeze.

## 2026-09-14 — Kill gf_v2 336px; stop pure volume A/B
- **Decision:** **KILL** `gf_baseline_v2` / `cache_gf_v2`. Keep `gf_v0` (3×12×224) as the only greenfield floor. **Do not** spend the next iterate on more slices, higher res, or more series on this same thin recipe without a new hypothesis.
- **Why:** gold-58 fell 0.7281 → **0.6925** (Δ -0.036); MCL 0.431. Combined with v1 kill, both volume axes failed.
- **Rejected:** Adopting 336 as default; trying 448 next; adding a 4th series as the immediate follow-up.
- **Next preferred axes:** 5-fold OOF on v0 (measurement) or fairer train-time teacher/labels (Post 05 track).

## 2026-09-15 — True OOF gold is the ship ruler (supersedes fold0→all58)
- **Decision:** Going forward, keep/kill uses **5-fold OOF full-58 gold macro AUC**. The gf_v0 OOF gold floor is **0.6144**. The earlier fold0-checkpoint-on-all-58 score (**0.7281**) is retired as optimistic (gold studies in that fold’s train set were not held out).
- **Why:** OOF gold 0.614 vs fold0→all58 0.728 (Δ −0.114) proved the optimistic ruler. Weak OOF 0.685 also sits well above true gold.
- **Rejected:** Shipping or ranking recipes against fold0→all58; public LB before beating OOF gold 0.6144 + 0.005.
- **Next:** Teacher/labels (Post 05), not more volume on 3×12×224.

## 2026-09-15 — Labels iterate #1 = ligament gap-fill on weak_v1 (not full v2 redo)
- **Decision:** Build `weak_labels_gf_lig1` by **only filling NaN** ACL/MCL/Medial Meniscus cells on top of weak_v1 (v7 TR/EL + EN injury/grade patches). Same image recipe; measure with true OOF gold vs 0.6144+0.005.
- **Why:** OOF gold blindness is concentrated in those labels; weak_v1 coverage there is thin (ACL 429 / MCL 262). Full weak_v2 was already killed for noise; gap-fill avoids rewriting committed v1 cells.
- **Rejected:** Retraining on full weak_v2; jumping straight to paid LLM fills for this first greenfield label A/B.

## 2026-09-15 — Kill gf_labels_lig1; next tighten fills (drop MCL)
- **Decision:** **KILL** `gf_labels_lig1` as training teacher. Keep weak_v1 + gf_v0 OOF gold floor **0.6144**. Next label A/B should **not** include the current MCL gap-fills (gold fill-prec ~0.53); prefer Med Men-only or Med Men+ACL with higher precision gate.
- **Why:** OOF gold 0.6144 → **0.6019** despite Med Men +0.128 and small ACL/MCL lifts; Effusion/Lat OA/Synovitis regressions dominated macro.
- **Rejected:** Shipping lig1 because weak OOF rose to 0.707; adding more aggressive fills without a precision gate.

## 2026-09-15 — Labels iterate #2 = Med-Men-only gap-fill (drop MCL and ACL)
- **Decision:** Build `weak_labels_gf_lig2` by filling NaN **Medial Meniscus** cells only on weak_v1. Do not fill MCL (gold fill-prec ~0.53) or ACL on this A/B. Same image recipe; keep if true OOF gold ≥ 0.6194.
- **Why:** lig1’s only large gold win was Med Men (+0.128); MCL fills were the noisiest; ACL fill was smaller and may have contributed to the Effusion/OA macro collapse. One-axis ablate.
- **Rejected:** Shipping lig1; jumping to Med Men+ACL in the same run; raising min_conf before measuring Med-Men-only.
