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

## 2026-09-16 — Kill gf_labels_lig2; freeze teacher A/Bs until the ruler is usable
- **Decision:** **KILL** `gf_labels_lig2` as training teacher (OOF gold 0.6103 < 0.6194). Keep weak_v1 + gf_v0. **Additionally: stop running new teacher/label A/Bs judged by a single 5-fold OOF gold-58 macro number.** A paired study-level bootstrap puts the lig2−v0 delta at −0.004 with 95% CI [−0.049, +0.041] and delta sd ≈ 0.022, versus a keep margin of 0.005 — the ruler is 4–5× coarser than the effect. Both the lig1 and lig2 kills are inside noise.
- **Why:** lig2 changed only the Medial Meniscus column, yet labels with identical teacher cells moved ±0.11–0.16 (ACL +0.110, Effusion −0.155; sd 0.079 over 11 unchanged labels). Continuing to spend ~4.4 h GPU per A/B to read sub-0.02 deltas cannot produce a trustworthy decision. This is the macro-level restatement of the 2026-08-27 "per-label gold reads are NOISE" finding.
- **Also recorded:** (a) gap-fill A/Bs were never single-factor — filling NaN cells also admits new studies into training (2449 → 3023 for lig2); (b) the recipe is undertrained (weak-val still rising at the final epoch in 8/10 folds) and the checkpoint is selected on weak-val, which disagrees with gold — both inflate run-to-run variance.
- **Kept as the one reproducible signal:** Med Men gap-fill gains +0.129 (lig1) and +0.137 (lig2) on gold Med Men across two independent runs. Do not discard the Med Men fills; re-test them once the ruler is quieter, or adopt them on per-label evidence.
- **Rejected:** Declaring lig2 a real regression; launching a third teacher recipe (tighter min_conf, Med Men + ACL) against the same noisy ruler; relaxing the 0.005 margin to manufacture a keep.

## 2026-09-16 — Derive the keep margin from measured seed noise (pre-registered)
- **Decision:** Before any further recipe A/B, run gf_v0 5-fold at two extra seeds (**1337**, **2024**) with everything else byte-identical, and adopt this rule, **fixed before the results are seen**: the greenfield baseline becomes the **seed-averaged OOF gold macro** over seeds {42, 1337, 2024}, and the keep margin becomes **max(0.005, 2 × measured seed sd)**. Candidates are scored as seed-averaged predictions against that threshold.
- **Why:** The 0.005 margin was a guess, and the 2026-09-16 noise audit shows the paired-delta sd is ~0.022. Deciding recipes with an unquantified instrument is how lig1 and lig2 got "killed" on coin flips. Seed replicates cost ~4.4 h GPU each and make every future decision interpretable; seed-averaging also attacks training stochasticity, which the identical-label swings (ACL +0.110, Effusion −0.155) show is a dominant noise source.
- **Implementation note:** `train_baseline_fold.py` takes the seed from the config with no CLI override, so the kernels patch the single `seed:` line of the unmodified `gf_baseline_v0.yaml` at runtime and assert that exactly one line changed. The `rsna-knee-gf-v0-meta` dataset is reused untouched so code cannot drift from the seed-42 run.
- **Rejected:** Guessing a wider margin without measuring; widening the gold set with pseudo-labels (reintroduces teacher bias); a public LB probe to calibrate (burns a submit on a 0.61-macro model); continuing teacher A/Bs while the ruler is unquantified.

## 2026-09-16 — Retire the 0.6144 "floor"; gf_v0 is 0.592 ± 0.021; single-seed A/Bs are banned
- **Decision:** The gf_v0 OOF gold floor of **0.6144 is retired.** Three runs of the identical recipe (seeds 42 / 1337 / 2024) scored **0.6144 / 0.5720 / 0.5889** — mean **0.5918**, sd **0.0214**, range **0.0424**. gf_v0 is now recorded as **0.592 ± 0.021**, and 0.6144 is understood as its luckiest draw (+1.06 sd). Going forward: **no recipe may be kept or killed on a single-seed run.** Every A/B runs ≥3 seeds and is judged on the **seed-averaged** OOF gold macro; the current baseline to beat is the seed-averaged **0.6173**.
- **Why:** With a measured seed sd of 0.0214 and a 0.005 margin, our decision procedure was reading noise. Concretely, `gf_labels_lig1` (0.6019) and `gf_labels_lig2` (0.6103) both sit *above* the v0 seed mean yet were killed for failing to beat the lucky draw; and the Effusion "collapse" that drove those kills reflects Effusion's own seed sd of **0.133** (0.889 was the outlier, not the norm). The weak ruler's sd is only 0.0085 on ~2449 studies versus 0.0214 on 58 gold, so the amplifier is small-n gold, which no amount of training discipline removes.
- **Consequences:** (a) v1/v2/lig1/lig2 verdicts are all void — they were decided inside the noise band, and none may be cited as evidence; (b) seed-ensembling is adopted as a genuine component of the model, not just a measurement trick, since the 3-seed average (0.6173) beats every single seed; (c) an A/B now costs ~13 h GPU (3 × 4.4 h), so at a 30 h weekly quota we get roughly two honest A/Bs per week and must spend them on large-effect levers.
- **Rejected:** Keeping 0.6144 as the bar; reinstating lig1/lig2 as wins (they are single-seed draws too — indistinguishable, not proven); using the guessed 0.005 margin for single-seed runs; deciding recipes on the quieter weak ruler alone (it measures agreement with the noisy teacher, not truth).

## 2026-09-16 — Next lever = convergence + checkpoint selection, measured within-run (paired)
- **Decision:** Spend the next A/B on `gf_v0c`: the gf_v0 recipe trained **10 epochs instead of 5**, backbone frozen throughout, saving val predictions every epoch so that checkpoint-selection policies (best-on-weak-val / final epoch / last-3 average / last-5 average) are compared **offline from the same trained models**. Stage 1 is one seed (~8 h); seeds 1337 + 2024 (~18 h) follow only if stage 1 shows a paired gain.
- **Why:** Chosen over label recipes, unfreeze, and volume because it is the one lever that should both raise the mean *and* shrink the seed sd, which makes every later A/B cheaper — the binding constraint is now measurement cost (~13 h GPU per honest 3-seed A/B against a 30 h weekly quota). The evidence: weak-val was still rising at the final epoch in 8/10 folds, so the recipe never converged, and the shipped checkpoint was the weak-val peak even though weak-val disagrees with gold.
- **Why one seed is legitimate for stage 1:** with a constant LR and no unfreeze, epochs 0–4 of the 10-epoch run retrace the 5-epoch run at the same seed, so "5 vs 10 epochs" and "policy A vs policy B" are paired comparisons within a single trajectory and carry none of the cross-run seed noise that voided the earlier verdicts. No keep/kill will be declared from stage 1 alone.
- **Rejected:** Changing epochs and selection policy as separate GPU runs (the per-epoch OOF makes selection free); adding an LR schedule or unfreeze in the same run (would break the pairing property and reopen a settled decision); jumping straight to 3 seeds before knowing whether the curve even rises.

## 2026-09-20 — Kill "train longer" on frozen-S; do not confirm at 3 seeds
- **Decision:** **KILL** the gf_v0c longer-schedule hypothesis (epochs 5→10) as a path to higher gold. Do **not** launch seeds 1337/2024 at 10 epochs. Gold OOF peaks at epoch **4 (0.6317)** then declines while weak keeps rising (teacher overfit). Keep the recipe at **5 epochs**. Secondary finding: best-on-weak-val checkpointing costs ~0.017 gold vs a uniform epoch-4 pick on this seed; prefer fixed-epoch / last-k averaging over weak-val peak-picking as a default, but treat that as hygiene, not a keep.
- **Why:** Stage-1 pairing worked (epochs 0–4 weak-val drift 0.000 vs recorded seed-42). The pre-registered gate was "only spend ~18 h on more seeds if the gold curve rises past epoch 4." It did not. Burning 18 h to confirm `avg_last5` would validate a selection trick on a schedule that is already past the gold peak.
- **Rejected:** Declaring avg_last5 (0.6265) a keep against the 0.6173 seed-averaged baseline (single seed; margin 0.0427); extending to 15–20 epochs; using weak-val rise as evidence of undertraining.

## 2026-09-20 — Next lever = clean Med Men fills (lig3), staged per-label gate
- **Decision:** Run `gf_labels_lig3`: Med Men gap-fill on weak_v1 **only for studies that already have ≥1 weak_v1 label** (ANY stays 2449). Stage 1 = seed 42 with a **pre-registered per-label gate**: Med Men gold OOF ≥ 0.566. Only if that passes, spend ~9 h on seeds 1337+2024 for a seed-averaged macro keep vs 0.6173 + 0.0427.
- **Why:** Med Men +0.13 reproduced on lig1 and lig2, but both admitted hundreds of new studies into training (ANY 2449→3023). lig3 is the first single-factor test of that signal. Empirically lig3 only adds **+20** Med Men cells inside the existing pool — so if the gate fails, the earlier “Med Men win” was mostly the study-set confound.
- **Rejected:** Jumping to unfreeze (still high risk on noisy teacher); volume redesign (cache rebuild; prior directional failures); burning 13 h on 3 seeds before knowing whether the clean fills move Med Men at all; treating stage-1 macro as a keep/kill.

## 2026-09-20 — Kill clean Med Men (lig3); reopen unfreeze under paired gates
- **Decision:** **KILL** `gf_labels_lig3` (Med Men gold 0.4868, gate 0.566 failed). Close further Med Men gap-fill A/Bs with this extractor. Next: `gf_v0u` — freeze 5 epochs then unfreeze backbone at lr×0.05 for 3 epochs, with per-epoch gold OOF and pre-registered COLLAPSE / PROMISING / INCONCLUSIVE gates. Multi-seed only if PROMISING.
- **Why:** Clean fills (+20 cells, ANY fixed at 2449) produced zero Med Men lift → the lig1/lig2 +0.13 was confounded by admitting new studies. Remaining large-effect levers on the frozen thin recipe are adaptation (unfreeze) or volume/encoder redesign. Unfreeze is the cheapest to measure honestly because epochs 0–4 retrace v0 (paired).
- **Rejected:** Spending ~9 h on lig3 seeds 1337/2024; more aggressive Med Men fills that re-admit studies; treating lig2 as a win.

## 2026-09-21 — Kill unfreeze (paired collapse −0.114); honest volume retest next
- **Decision:** **KILL** backbone unfreeze on gf_v0 / weak_v1 (frozen max gold 0.6317 → unfrozen max 0.5177, Δ −0.114). Keep DECISIONS 2026-08-12. Next: honest **true-OOF** retest of 24-slice `cache_gf_v1` (prior kill void), stage-1 seed 42, gates ±0.02 vs v0 seed42 0.6144.
- **Why:** Unfreeze destroys gold immediately on the first unfrozen epoch; no ambiguity. Remaining cheap lever with existing artifacts is volume (cache already built). Structural / MRI-CORE waits until volume is honestly settled.
- **Rejected:** Multi-seed unfreeze; last-block-only micro-variants before a volume read; rebuilding cache.

## 2026-09-21 — Kill 24-slice volume (honest true-OOF); next = MRI-CORE encoder
- **Decision:** **KILL** `cache_gf_v1` / 24-slice under true 5-fold OOF (gold **0.6080** vs v0 seed42 **0.6144**, Δ −0.006 → INCONCLUSIVE gate → no multi-seed). Close further pure volume A/Bs on the frozen DINOv2-S 3-series recipe (12↔24 slices already settled; 336px prior kill stands as directional). **Next lever: MRI-CORE ViT-B** as frozen slice encoder (Apache-2.0; already vendored under `third_party/mri_foundation`), stage-1 seed 42 true-OOF, same gates ±0.02 vs 0.6144.
- **Why:** Labels, train-longer, unfreeze, and volume all failed or collapsed on the thin DINOv2-S recipe. Remaining large-effect bet in EXTERNAL_ASSETS / greenfield thesis is a domain-matched MRI foundation encoder (or later DINOv2-B / aggregator). MRI-CORE is public, offline-bundleable, and already used in the old stack’s feature path — greenfield `src/rsna_knee` still only loads DINOv2.
- **Rejected:** Seeds 1337/2024 on v1; more slice/resolution micro-sweeps; another weak-label teacher pass.

## 2026-09-21 — Launch MRI-CORE ViT-B stage 1 (honest true-OOF)
- **Decision:** Run `gf_mri_core_v0`: frozen MRI-CORE ViT-B on `cache_gf_v0`, upsample 224→384, seed 42, 5-fold true OOF. Gate ±0.02 vs v0 seed42 0.6144. Multi-seed only if PROMISING.
- **Loader note:** Official `build_sam` mis-assigns teacher `pos_embed` and the checkpoint has no SAM neck. We map `teacher.backbone.*` ourselves, interpolate pos embeds, and pool **768-d** pre-neck tokens.
- **Why:** Thin DINOv2-S levers closed; MRI-CORE is the pre-registered domain encoder in EXTERNAL_ASSETS; weights + source already offline-bundled.
- **Rejected:** Rebuilding a 384 cache first; feature-cache-only head (want end-to-end parity with gf_v0); DINOv2-B before MRI-CORE.

## 2026-09-23 — Pivot: fork the public 0.936 floor; autonomous search loop; 6 Oct gate
- **Decision (user-approved plan):** Stop greenfield thin-cache A/Bs. Day 1: fork the public rsna-base recipe (Max-Span Dense Corpus + twelve-findings weights + DINOv2, public LB 0.936, ~3 min on T4x2) and submit once to confirm the floor. The separating lever is our own multilingual report-distilled soft labels trained on dense geometry. Training moves off Kaggle GPU to a rented RunPod RTX 4090 (~$0.34/hr); total cap $300 (~$230 GPU, ~$50 LLM API). Search runs through `src/rsna_knee/agent/` (MLEvolve-style MCGS journal + retrospective memory + ban list, Dream-RSI replay over stored logits and the experiment tree). Pre-registered gate on **2026-10-06**: public LB ≥ 0.950 keeps main-board search; below that, spend the rest on seed/fold averaging and the efficiency final.
- **Why:** gf_v0 is 0.592 ± 0.021 gold vs a public band of ~0.95–0.96 with 10+ teams at 0.96; rebuilding the public recipe from scratch costs a third of the remaining time. Public measurements show geometry and labels, not backbones or ensembles, move the board.
- **Supersedes:** "no public LB until OOF gold win" for the single floor-confirmation submit (a public, known-score notebook); later submits are proxy calibrations, max ~2/day. The MRI-CORE thin-cache run is logged when it finishes but no longer blocks anything.
- **Open risk:** the Max-Span Dense Corpus and twelve-findings weights show no explicit license on their Kaggle pages (only the notebooks are Apache-2.0). Must be confirmed before either is part of a final, because winners must open-source all assets.
- **Rejected:** Top-5 main board as the success criterion (stretch only); rebuilding a dense cache before forking the public one; training on Kaggle GPU (weekly quota stalls an autonomous loop); running MLEvolve unmodified on raw MRI (install/ban-list waste on a $300 cap).

## 2026-09-24 — First-train supervision = public LLM soft labels; keyword teachers retired
- **Decision:** The first RunPod train uses the mean of the `pilkwang/rsna-knee-llm-labels` and `dreaddevelopment/rsna-knee-labels` soft probabilities (both CC0; plain mean, not rank-mean, so per-label prevalence survives as a BCE target), with expert labels overriding the 58. File: `data/processed/labels_llm_blend_v1.csv` from `scripts/build_llm_blend_labels.py`. Under the pre-registered per-label gate (precision ≥ 0.5 on the experts), **Contusion** from pilk fails (0.44): it stays in the loss as a soft target at half weight rather than dropped, because it is a scored label. Keyword `weak_v1` / `v7` are no longer used as teachers for any new train.
- **Why:** On the 57 experts pilk scores macro AUC 0.870 vs keyword 0.63–0.65; the paired gap is +0.24 [0.19, 0.29], ~6x the gold-58 noise band. dread cannot be graded on the experts (it omits them), but it agrees with pilk at Spearman 0.81 on the other ~4300 studies, so it is a second LLM read, not an independent weaker one.
- **Consequence for the plan:** "Our own soft labels beat the public ones" is now a measurable bar: a candidate label file must beat pilk on the experts with a paired bootstrap CI excluding 0, with the caveat that pilk was prompt-checked on the annotated studies. Spend on our own LLM pass (DeepSeek) only for the weak columns (Synovitis, Contusion, Lat OA) and only after the first train exists.
- **Rejected:** Building a full own-LLM label set before the first train; dropping Contusion from the loss; averaging keyword labels into the teacher.
