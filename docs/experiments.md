# Experiments registry

Append one row (or block) per run. Never delete history.

| ID | Date | Config | Fold | OOF macro | Public LB | Runtime s | Notes / conclusion |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | No runs yet |

### 2026-08-09 — weak_labels_v1 (EN+ES keywords)
- config: `src/rsna_knee/text/weak_labels.py`
- audit set: 58 expert-labeled studies
- macro F1 ≈ 0.44, macro precision ≈ 0.69, macro recall ≈ 0.34
- artifact: `data/processed/weak_labels_v1.csv`, `docs/audit/weak_label_vs_expert.csv`
- conclusion: keep as noisy pretrain signal; still need more languages + better OA phrases; do not trust as sole supervision

### 2026-08-11 — baseline_dinov2_s fold0 smoke (Kaggle T4)
- config: `configs/baseline_dinov2_s.yaml`
- data: cache_v1 (3×12×224), weak_labels_v1, folds_v1
- train/val: 1953 / 496 (2449 studies with any label)
- epochs: 3; freeze_backbone_epochs=1 then unfreeze
- val macro_auc: **0.685 (ep0)** → 0.614 (ep1) → 0.554 (ep2)
- conclusion: pipeline works; unfreeze+LR too aggressive / weak-label noise. Next: keep backbone frozen longer, lower LR, optional expert fine-tune. Do not submit yet vs LB ~0.94.

### 2026-08-11 — baseline_dinov2_s fold0 frozen 5ep
- same data/cache; backbone **frozen all epochs**; lr 3e-4 head-only
- val macro_auc by epoch: 0.701 → 0.727 → 0.739 → 0.748 → **0.764**
- artifacts: `/kaggle/working/baseline_dinov2_s/fold0_best.pt` (+ oof csv/npy)
- conclusion: **keep** — clear gain over 0.685. Next: folds 1–4 same recipe; still no LB probe until multi-fold OOF.

### 2026-08-11 — baseline_dinov2_s folds 1–4 frozen 5ep
- same recipe as fold0 (frozen backbone, weak+expert, cache_v1)
- best val: f1 **0.732**, f2 **0.725**, f3 **0.697**, f4 **0.675**
- mean folds1–4: **0.707**; mean folds0–4 with prior fold0 0.764: **~0.719**
- conclusion: reproducible ~0.72 ceiling for this setup. Next lever is labels + bigger backbone, not more identical folds. No submit.

### 2026-08-11 — weak_labels_v2 (code + audit)
- config: `src/rsna_knee/text/weak_labels.py` (FR/DE/PT/NL + EN/ES)
- expert audit (58): macro F1 **0.428**, prec **0.722**, rec **0.342** (similar to v1 ~0.44; expert set not FR-heavy)
- coverage: studies with any label **2449 → 2749**; positives +252 net (Effusion +212, Baker's +77, ACL +34; OA slightly fewer)
- artifacts: `data/processed/weak_labels_v2.csv`, `docs/audit/weak_label_v2_vs_expert.csv`, `docs/RANK1_ROADMAP.md`
- train path: `configs/main_dinov2_b.yaml`, trainer `pos_weight` + `unfreeze_lr_mult`, notebooks 10/30
- conclusion: ship for Kaggle A/B vs frozen-S mean **0.719**; no LB until OOF lift.

### 2026-08-11 — baseline_dinov2_s + weak_v2 fold0 (partial 5-fold)
- same cache/backbone/freeze-5ep; **weak_labels_v2**; confounds: `pos_weight=1.5` (new vs v1 run)
- studies with any label: 2749; fold0 train/val 2192/557
- fold0 val: 0.649 → 0.652 → 0.695 → 0.698 → **0.707** best
- fold1 (partial): 0.676 → 0.685 → 0.687 @ ep2 (still running)
- vs v1 fold0 **0.764** → clear fold0 regression
- conclusion: **provisional kill on v2-as-shipped** pending full mean; next isolate `pos_weight=1` vs label noise. No submit.

### 2026-08-11 — baseline_dinov2_s folds1–4 re-run (weak_v1, confirm)
- **Not weak_v2**: `studies with any label 2449`; out `baseline_dinov2_s/fold*`
- best: f1 **0.732**, f2 **0.725**, f3 **0.697**, f4 **0.675**; mean 1–4 **0.707**
- matches prior folds1–4 within noise; with fold0 0.764 → still **~0.719**
- note: log lines duplicated (likely 2 GPUs / double cell) — same scores twice
- conclusion: **keep** as confirmed floor; do not re-run again. Pivot to label/pos_weight ablation or B.

### 2026-08-11 — fold0 ablate v1 vs v2 (`pos_weight=1.0`)
- frozen S 5ep, cache_v1, local `third_party/dinov2`
- `v1_pw1`: 2449 labels → best **0.725** (ep: 0.682→0.700→0.704→0.704→0.725)
- `v2_pw1`: 2749 labels → best **0.718** (ep: 0.657→0.698→0.700→**0.718**→0.713)
- vs historical v1 fold0 **0.764** (ablate v1 under-shot; still beats v2)
- conclusion: **kill weak_v2** for training; noise > coverage. Next = DINOv2-B or richer cache on **v1**.

### 2026-08-12 — main_dinov2_b fold0 weak_v1 (freeze 4 → unfreeze)
- cache_v1, weak_v1, pos_weight=1.0, lr 1.5e-4, unfreeze_lr_mult=0.05
- val: 0.667 → 0.708 → 0.723 → **0.729** (ep3) → unfreeze → **0.615** (ep4)
- conclusion: **keep ep3 checkpoint only**; unfreeze still destroys signal. Next: full-freeze B 8ep. No submit.

### 2026-08-12 — main_dinov2_b fold0 weak_v1 (fully frozen 8ep)
- cache_v1, weak_v1, pos_weight=1.0, backbone frozen all 8 epochs
- val: 0.693 → 0.722 → 0.727 → 0.744 → 0.741 → 0.746 → 0.749 → **0.759**
- runtime: ~12,156 s on Kaggle T4x2 notebook (single process on one GPU)
- conclusion: **best B result so far**, but still below frozen-S fold0 **0.764**. Bigger backbone alone is not enough on `cache_v1`; next try richer cache, not more unfreeze.

### 2026-08-15 — baseline_dinov2_s fold0 cache_v2 (4×16, frozen, weak_v1)
- cache_v2 used (path True); 2449 labels; freeze 5ep; pos_weight=1.0
- val: 0.687 → 0.709 → 0.712 → **0.738** → 0.727
- vs cache_v1 S fold0 **0.764** and frozen B **0.759**
- conclusion: **kill cache_v2 as default**. Extra series/slices did not help. No 5-fold on v2. Next: expert fine-tune or better labels on cache_v1.

### 2026-08-15 — expert head-FT frozen B fold0
- init: `main_dinov2_b_v1_frozen/fold0/fold0_best.pt` (ckpt_auc **0.7591**)
- train: 45 expert studies (fold≠0); val: full fold0 496; lr 3e-5; backbone frozen
- before FT **0.7591**; ep0 **0.7601**; then 0.760→…→**0.745**
- conclusion: **kill**. +0.001 is noise; later epochs overfit 45 studies. Keep original B ckpt. Next: better weak labels (LLM), not more head-FT.


### 2026-08-24 — Noisy-teacher tooling (code-only; no training run yet)
- code: robust losses (`masked_multilabel_loss`: bce|gce|sce, label smoothing, per-label pos_weight), `rsna_knee.evaluation` + `scripts/oof_report.py` (full-OOF macro AUC + study-level bootstrap CI + keep/kill rule), ensembled NLI hypotheses + tested `merge_pseudo_labels`, `configs/labels_v3_robust.yaml`, notebook 14 now imports the package.
- validation: 26 unit tests pass locally (CPU torch). BCE mode is numerically identical to the legacy loss; GCE is provably less perturbed by a flipped label; a tiny end-to-end model forward/backward runs on CPU. OOF A/B CLI verified on synthetic overlapping data (baseline 0.711 vs candidate 0.869 → KEEP with separated CIs).
- **no competition scores claimed** — training needs Kaggle GPU + the 569 GB DICOM data.
- planned A/B (pre-registered): frozen-B weak_v3 + GCE + smoothing vs frozen-B weak_v1 0.759; keep only if full 5-fold OOF delta ≥ 0.005 and candidate bootstrap CI clears the baseline mean.
- conclusion: iterate — ship tooling, then measure on Kaggle before any submit.

### 2026-08-24 — v6b constrained-Qwen labels PASS the gate (verified vs 58 gold)
- source: reconstructed from Kaggle `girishbose/weak-labels-v6-constrained-qwen` outputs (skeleton + raw fills) + real `train.csv`.
- v6 baseline (reproduced exactly): combined 58-expert prec **0.6826**, rec 0.7178, parse **1.0**, coverage 30,216 → FAIL (prec < 0.69).
- per-label drag: MCL combined prec **0.381** (LLM-fill MCL prec **0.231**); Contusion 0.591, Lateral OA 0.60; rest strong.
- rule (pre-registered): drop LLM fills where fill-precision < 0.5 → **{MCL}**.
- v6b result: combined prec **0.7029**, rec **0.690**, parse **1.0**, coverage **27,698**, fill_pos 110, fill_prec 0.715 → **PASS**.
- candidate: `weak_labels_v6b_candidate.csv` — 4,307/4,407 studies, 11,539 positives, 28,044 known cells; uploaded as `girishbose/rsna-knee-weak-v6b`.
- code: `src/rsna_knee/text/fill_policy.py` (+ `tests/test_fill_policy.py`), 28 tests pass.
- training: `girishbose/train-b-fold0-weak-v6b` COMPLETE (frozen-B fold0; first launch died on a Kaggle GPU/env mismatch — fixed by pinning `machine_shape=NvidiaTeslaT4` + the known-good `docker_image`).
- result: fold0 val macro_auc by epoch 0.726→0.733→0.749→0.751→0.751→0.757→**0.770**(ep6)→0.763. BEST **0.7700 > frozen-B weak_v1 0.759** (+0.011). Clean monotonic train, no collapse.
- caveat (honest): val AUC is scored against each run's own weak labels, and v6b labels differ from v1 (v6b covers 4307 vs 2749 studies), so 0.770-vs-0.759 is favorable but not a fully clean A/B. Per-fold gold cross-check is tiny (only 13 experts in fold0 val → macro 0.672, high variance) so not decisive on its own. The vetted gold signal is the passed label gate.
- fold1 A/B (`girishbose/train-b-fold1-weak-v6b`) COMPLETE: epochs 0.718→0.726→0.728→0.734→0.734→0.735→0.732→**0.739**. BEST **0.7387 > untouched fold1 0.732** (+0.007). Win TRANSFERS.
- 5-fold: folds 2,3,4 launched to complete the OOF (`train-b-fold{2,3,4}-weak-v6b`). Kaggle caps 2 concurrent GPU sessions → f2+f3 run first, f4 after a slot frees.
- next decisive eval: aggregate all 5 folds' OOF and score vs ALL 58 expert gold (unconfounded, n=58 not 13) + full weak-label OOF, then decide submit.
- folds 2,3 COMPLETE: fold2 **0.7626**, fold3 **0.7661** (vs v6b labels). fold4 running.
- **DECISIVE gold read (4-fold OOF, 46/58 experts):** macro AUC **vs expert gold = 0.7124**; vs v6b weak labels = 0.7543 (confounded). This is the first meaningful gold-OOF estimate in the project.
- per-label vs gold: strong Medial OA 0.893, Effusion 0.854, Baker's 0.785, Lat Meniscus 0.764; **weak ACL 0.552, Contusion 0.559, Fracture 0.597, MCL 0.663, PF OA 0.696**.
- FULL 5-fold complete: f2 0.763, f3 0.766, f4 0.736.
- **FINAL full 5-fold OOF (all 58 experts): vs gold = 0.6895; vs v6b weak labels = 0.7508.**
- per-label vs gold: **ACL 0.501 (chance), Fracture 0.556, MCL 0.601, Contusion 0.611**, Synovitis 0.700, MedMen 0.709, PFOA 0.712, LatOA 0.714, Baker's 0.726, LatMen 0.733, Effusion 0.850, MedialOA 0.860.
- conclusion: KEEP v6b labels (legit gated win), **DO NOT submit** — true gold-OOF 0.69 < 0.72 floor and far from ~0.94. Bottleneck is now image signal, esp. **ACL at chance (0.50)** while OA/Effusion work → the model is blind to ACL despite clean labels. Next lever = image pipeline (sagittal coverage / cache / plane routing) targeting ACL/Fracture/Contusion, NOT more label recipes.

### 2026-08-26 — v6c (drop ACL/MCL/LatOA coin-flip LLM fills) FIXES ACL
- rule: drop LLM fills with 58-expert fill-precision < 0.55 → {MCL, ACL, Lateral OA}; keep keyword skeleton. Gate PASSES (prec 0.724, rec 0.634, coverage 23,143). Uploaded `girishbose/rsna-knee-weak-v6c`.
- v6c fold0 0.768 / fold1 0.749 (vs its own labels). Kernels `train-b-fold{0,1}-weak-v6c`.
- **gold read (fold0+1, 24 experts): v6c 0.7358 vs v6b 0.6894 (+0.046)**. ACL **0.452→0.770 (+0.319)**, MCL 0.381→0.698, Med Men +0.121, Medial OA +0.118, Synovitis +0.107, Baker's +0.159; regressions Effusion -0.230, Lateral OA -0.159 (likely n=24 variance).
- v6c folds 2,3 done: 0.7624 / 0.7636 (vs own labels).
- **CONFIRMED 4-fold gold OOF (46 experts): v6c 0.7365 vs v6b 0.7124 (+0.024).** ACL +0.163 (0.552→0.715), MCL +0.099, Synovitis +0.103, Contusion +0.091, Medial OA +0.044; regressions Lat Meniscus -0.117, Lat OA -0.093, Effusion -0.073 (representation shift from dropped label cols; fills for those weren't changed). Net clearly positive.
- conclusion: ADOPT v6c over v6b — confirmed gold win on 46 experts, driven by fixing coin-flip ACL/MCL fills. fold4 running for full-58. Watch Lateral Meniscus/Lateral OA regressions (Lat OA fill 0.545 was borderline; may relax threshold to <0.5+ACL later). Method generalizes: audit per-label fill precision, drop coin-flips.

### 2026-08-27 — v6d (keep Lateral OA) ties v6c; per-label gold reads are NOISE
- v6d = drop only {MCL, ACL} fills, keep Lateral OA. Gate passes (prec 0.712, rec 0.680). `girishbose/rsna-knee-weak-v6d`, kernels `train-b-fold{0,1}-weak-v6d` (0.761/0.765 vs own labels).
- fold0+1 gold (n=24): v6b 0.6894, v6c 0.7358, v6d **0.7378** — v6c/v6d tied.
- **Noise proof:** v6c & v6d share IDENTICAL ACL+MCL labels, yet ACL gold 0.770(v6c) vs 0.596(v6d), MCL 0.698 vs 0.508 — driven only by the Lateral OA column change → per-label gold AUC at n≤58 is training-noise-dominated.
- conclusion: ADOPT v6c (full-58 0.7023 > v6b 0.6895 is the only stable signal). STOP label micro-tuning — v6d≈v6c within noise; do not burn GPU resolving sub-0.02 deltas or trust per-label gold stories at this n. Next real lever is image/backbone (needs a plan + user steer), not more label recipes.

### 2026-08-27 — v7 multilingual extractor (adds Turkish + Greek)
- code: `src/rsna_knee/text/weak_labels_v7.py` (+ `tests/test_weak_labels_v7.py`, 7 tests). Language detect → Turkish/Greek handled with correct negation direction + normalcy + borderline("minimal/mild"→abstain, matching host "on the fence = negative"); other langs delegate to v2. Fixed the Turkish dotted-i (İ/ı) re.IGNORECASE fold bug that misdetects English.
- coverage: Turkish 0.0→6.6 known cells/study, Greek 0.1→4.5 vs the v2 keyword extractor (recovers ~860 previously-unsupervised studies). BUT v6c already labels TR/EL via Qwen (5.2/2.9), so v7 alone is not a clear win over the current best.
- 58-gold audit (only 6 TR + 3 EL → unreliable): TR positive-precision ~0.47, TR negatives NPV 0.80; EL pos 0.67.
- **v7 vs Qwen cross-check: 88.2% agreement on Turkish (2107 cells), 76.8% on Greek (482).** Two independent methods concur → both capture real signal; the agreement cells are high-precision.
- conclusion: v7 is a validated independent multilingual labeler. Its highest-value use is a **v7∩Qwen consensus** (label where both agree, else abstain) for high-precision TR/EL supervision — not v7 alone. Blocker remains measurement: 6+3 TR/EL gold can't validate model impact → need an external ruler (MRNet/KneeMRI) before trusting a retrain delta.

### 2026-08-29 — Rank1 fold0+1 ep10/9; fold2+3 pushed
- Still RUNNING. Best: fold0 **0.7558** (ep10), fold1 **0.7640** (ep6, ep9 **0.7636**). Fold1 **+0.015** vs v6c 0.7493 → smoke pass.
- Fold1 **+0.015** vs v6c → smoke pass. Folds 2–3 push **blocked** (GPU full); retry on COMPLETE.
- conclusion: iterate — provisional KEEP; wait COMPLETE + folds 2–4.

### 2026-08-29 — Rank1 fold0+1 late-train: fold1 beats v6c
- Still RUNNING. Best: fold0 **0.7545** (ep7, −0.014 vs 0.7683), fold1 **0.7640** (ep6, **+0.015** vs 0.7493).
- Fold1 smoke-bar pass. Finish remaining epochs then push folds 2–4 if fold0 closes or holds; decide keep on full-58 gold ≥ 0.7073.
- S01b deferred (do not steal GPU from a live win).
- conclusion: iterate — provisional KEEP lean on rank1 image lever.

### 2026-08-29 — Rank1 fold0+1 mid-train ep2–3
- Still RUNNING. Best so far: fold0 **0.7416** (ep2), fold1 **0.7215** (ep2) vs v6c 0.7683 / 0.7493 (Δ −0.027 / −0.028).
- Climbing but trailing; ~8–9 epochs left. S01b still GPU-slot blocked.
- conclusion: iterate — wait for finish; provisional lean if final best stays < v6c − 0.005.

### 2026-08-29 — Rank1 fold0+1 TRAINING (ep0 interim weak-val)
- Live: both RUNNING; `label_plane_routing=True`; folds/code/cache/weights resolved correctly.
- epoch 0 weak-val: fold0 **0.6847**, fold1 **0.6822** (vs v6c 0.7683 / 0.7493). Too early to keep/kill.
- S01b GPU submit still waiting on free slot. S01 timeout unchanged (no score).
- conclusion: iterate — wait for ≥epoch 6–12; then compare weak-val and launch folds 2–4 only if promising.

### 2026-08-29 — S01 TIMED OUT (no score); rank1 GPU launched; S01b decode-once queued
- S01 ref **55851760** COMPLETE with empty publicScore — runtime exceeded on hidden test (CPU ~126s/study × 5 ckpts). Ensemble hypothesis **not killed**.
- Fix: `run_model_submission` decode-once; full `src/rsna_knee` + folds in `rsna-knee-rank1-patch`; GPU submit kernel; `push_kaggle_kernels.py`.
- Rank1: fold0+1 pushed/running after import+folds path fixes. Gate: full-58 gold ≥ 0.7073.
- S02 still queued pending a scored S01b ≥ 0.690.
- conclusion: iterate — wait rank1 weak-val; free a GPU slot for S01b submit then competition re-submit.

### 2026-08-29 — GPU launch queue: rank1 train + S02 submit kernel
- code: `run_model_submission` in `infer.py` (uniform + per-label AUC blend from OOF); S02 kernel `submit-v6c-5fold-s02`; rank1 fold2–4 kernels + metadata; `scripts/push_kaggle_kernels.py`, `scripts/package_rank1_patch.py`.
- tests: `tests/test_infer.py` (OOF weight normalization + submission schema); 36 passed locally.
- blocked: Kaggle token not in cloud env — user must `export KAGGLE_API_TOKEN=...` then push/launch.
- conclusion: iterate — launch rank1 fold0+1 on GPU; queue S02 when S01 LB ≥ 0.690.

### 2026-08-28 — S01 SUBMITTED: 5-fold v6c uniform blend (LB pending)
- experiment **S01**: hypothesis fold0-only LB 0.682 understates 5-fold; expect **~0.70–0.72**.
- kernel `girishbose/submit-v6c-5fold` v1 COMPLETE (5 checkpoints, uniform mean, 379s on 3-study dry run).
- **competition submit:** ref **55851760**, message "S01: 5-fold v6c uniform blend", status **PENDING** (2026-08-28).
- baseline: fold0-only **0.682** (ref 55818692). Kill if LB < **0.690**.
- note: v1 ran on CPU (GPU quota); re-push GPU metadata for faster/full test if needed.
- conclusion: iterate — record public score; queue S02 if ≥ 0.690.

### 2026-08-28 — Rank-1 image stack (plane routing + cache_v3 + ensemble submit)
- code: label plane routing (`LABEL_PLANE_PRIOR` → per-label series attention), ACL sagittal slice bias, `rank1_v6c.yaml`, `infer_ensemble.py`, Kaggle patch `rsna-knee-rank1-patch`.
- train kernels prepared; **GPU quota blocked push**.
- keep rule unchanged: full-58 gold vs v6c **0.7023**, margin **0.005**.
- conclusion: iterate when quota resets — rank1 is the registered hardest in-repo path toward 0.95; success not guaranteed.

### 2026-08-28 — GCE v6c KILL (fold0+1 identical to BCE; GCE never applied)
- fold0+1 v4 COMPLETE. Weak-val GCE = BCE (0.7683/0.7493). fold0+1 gold OOF **0.7358** vs BCE **0.7358** (Δ 0.0); OOF max abs diff **0.0**.
- Root cause: Kaggle `rsna-knee-code` trainer uses `masked_bce_with_logits`; loss-gce patch only shadowed `loss.py`.
- folds 2–4 not launched. Audit: `docs/audit/gce_v6c_fold01_keepkill.json`.
- conclusion: **KILL GCE** as implemented. Label lever exhausted with v8. Next real lever = image/backbone (user steer).

### 2026-08-27 — GCE v6c fold0+1 v4 RUNNING (v1–v3 empty-script failures fixed)
- kernels `train-b-fold{0,1}-gce-v6c` v1–v3: **0-byte scripts** → instant COMPLETE, no outputs. Real scripts in `kernels/train-b-fold*-gce-v6c.py`.
- fold0 **v4 RUNNING**; fold1 **v4 pushed + RUNNING** (22:15 UTC). fold2–4 dirs prepared, not launched.
- yunus screened: strict intersect +0; naive gap +24,628 mostly neg → no train.
- conclusion: iterate — compare weak-val vs v6c BCE (0.7683/0.7493); launch folds 2–4 if promising; full-58 gold vs 0.7023 when 5 folds done.

### 2026-08-27 — GCE loss A/B on v6c labels (fold0+1 RUNNING)
- code: `configs/v6c_gce.yaml` (GCE q=0.7, smoothing 0.05); Kaggle dataset `girishbose/rsna-knee-loss-gce` shadows stale code `loss.py`.
- kernels: `train-b-fold{0,1}-gce-v6c` (v2 with patch dataset). v1 may fail — missing patch at launch.
- yunus screened: strict intersect +0 cells; naive gap +24,628 (mostly neg) → no train.
- conclusion: iterate — measure GCE vs v6c BCE on full-58 gold; do not yunus gap-fill.

### 2026-08-27 — v8 KILL (full-58 gold 0.6935 < v6c 0.7023)
- fold4 COMPLETE (weak-val 0.7420). Full 5-fold OOF vs 58 gold: **v8 0.6935 vs v6c 0.7023 (Δ −0.0088) → KILL** (margin 0.005).
- Per-label vs v6c: Effusion +0.043, MedMen +0.020; losses MCL −0.051, LatOA −0.040, Fracture −0.032, ACL −0.020.
- Matched 4-fold already −0.024; weak-val lift was confounded by +495 TR/EL cells on dropped coin-flip cols.
- conclusion: **KILL v8**. Keep adopted **v6c** + LB calibration 0.682. Do not probe v8. Next levers: yunus-consensus teacher or image plan — not more consensus gap-fills on ACL/MCL/LatOA.

### 2026-08-27 — v8 folds 0–3 COMPLETE; matched 4-fold gold LOSES to v6c
- weak-val: f0 0.785 / f1 0.783 / f2 0.781 / f3 0.767 (all ≥ v6c counterparts). fold4 QUEUED.
- **Matched 4-fold gold (n=46): v6c 0.7365 → v8 0.7127 (−0.024).** Regressions: MCL −0.08, Fracture −0.07, Baker's −0.06, MedOA −0.04; small gains Effusion/PF OA.
- Earlier fold0+1 gold +0.01 was noise (same failure mode as v6d). Weak-val lift confounded by +495 TR/EL cells on ACL/MCL/LatOA.
- conclusion: provisional **KILL lean** on v8; confirm on full-58 after fold4. Do not LB-probe v8. Additive gap-fill of consensus onto dropped coin-flip labels likely re-poisoned training.

### 2026-08-27 — v8 train fold0+1 COMPLETE (interim; full-58 pending)
- kernels: `train-b-fold{0,1}-weak-v8` COMPLETE; fold2+3 RUNNING.
- weak-val (confounded): f0 **0.7853** / f1 **0.7827** vs v6c 0.7683 / 0.7493 (+0.017 / +0.033).
- gold fold0+1 (n=24, noisy): v6c 0.7358 → v8 **0.7459** (+0.010). Per-label swings large (MCL −0.17, LatOA +0.14) — expected noise at this n.
- conclusion: iterate — **do not keep/kill**; wait for folds 2–4 + full-58 gold vs 0.7023 (±0.005 rule). Projected LB ≈ OOF − 0.02.

### 2026-08-27 — In-domain cross-check + v8 candidate (additive TR/EL consensus)
- LB API confirm: submission 55818692 publicScore **0.682**.
- Cross-check (`scripts/crosscheck_labels.py`, soft lo/hi 0.2/0.7): yunus **91.2%** agree, dread 80.9% (MCL 38%), barun pseudo unusable (≈0.5 mass).
- v8: `build_weak_labels_v8.py` default = **additive gap-fill** of v7∩Qwen onto v6c for TR/EL only. Agree 85.1% (2593/3048). +495 cells (ACL+108, MCL+120, LatOA+267). Uploaded `girishbose/rsna-knee-weak-v8`.
- Gold n on TR/EL = 9 → cannot keep/kill from label audit alone.
- conclusion: iterate — **retrain frozen-B 5-fold on v8**; keep iff full-58 gold OOF ≥ 0.7023+0.005. No new LB probe yet. Rotate pasted token.

### 2026-08-27 — LB probe v6c fold0 = public **0.682** (CALIBRATION)
- submit: `girishbose/rsna-knee-submit-v6c` Version 1 Succeeded; images-only frozen DINOv2-B **v6c fold0**, cache_v1 3×12×224.
- **public LB: 0.682**
- vs full-58 gold OOF v6c **0.7023** → gold overstates public by **~0.020**
- vs weak-label OOF ~0.75 → weak overstates by **~0.07** (do not use for decisions)
- vs public top ~0.94 → gap **~0.26**; single fold0 probe may slightly understate a 5-fold blend, but not enough to change strategy
- conclusion: **KEEP gold-OOF as internal ruler** (apply ≈−0.02 for LB projection). **DO NOT final** this submission. Next: v8 consensus / in-domain competitor-label cross-check; retrain only with full-58 keep rule vs 0.7023; no further LB until projected LB clears a deliberate bar. Still need Kaggle token in cloud env to execute.

### 2026-08-27 — v8 consensus tooling + in-domain cross-check CLI (code-only)
- code: `src/rsna_knee/text/label_consensus.py` (`intersect_labels`, `agreement_stats`, `overlay_consensus`); `scripts/build_weak_labels_v8.py` (v6c base ⊕ v7∩Qwen, optional TR/EL lang filter via `detect_language`); `scripts/crosscheck_labels.py` (ours vs named competitor CSVs + optional gold audit).
- tests: `tests/test_label_consensus.py` (4) — agreement-only keep, outer unilateral→NaN, stats rates, overlay overwrite vs gap-fill. Related suite 13 passed.
- CLI smoke (synthetic): intersect agree 5/7; crosscheck found 1 disagreement cell as expected.
- **no competition scores** — blocked on rotated Kaggle token + human LB Submit on `rsna-knee-submit-v6c`. Next measure: download real v7/Qwen/v6c + competitor sets, build v8, full-58 gold OOF vs 0.7023.
- conclusion: iterate — tooling ready; do not retrain until auth + (preferably) LB calibration.

### 2026-08-27 — External ACL ruler (KneeMRI) — NEGATIVE (domain shift)
- Option #2: use external expert-labeled knee MRI as an unbiased ACL ruler. MRNet is gated (Stanford DUA; only a 22-byte Kaggle stub). KneeMRI (Croatia) IS on Kaggle (`sohaibanwaar1203/kneemridataset`): 736 sagittal volumes accessible, ACL 0/1/2, binary-positive prevalence 24.8% (representative, unlike the 58's 41%).
- Built `girishbose/knee-acl-ruler-v6c`: run our v6c fold0 model on KneeMRI as a 1-series sagittal study.
- Result: external ACL AUC **0.507** (fluid=0/center slices) → **0.530** (fixed: fluid=1/fat=1, spread slices). Both ≈ chance.
- conclusion: **external-image ruler is not viable for our model.** Severe domain shift + structural mismatch (our model expects the competition's 3-series plane/fluid/fat attention; KneeMRI is a single sagittal series) drive it to chance. Corroborates the top team's "extra corpora ≈ 0" — external knee MRI won't help THIS competition as ruler OR training data. Also a red flag that our ACL detector leans on competition-specific cues, not robust anatomy.
- pivot: the only reliable read of true performance on the competition distribution is (a) a single calibrated **LB probe** of v6c, or (b) an **in-domain** label cross-check vs competitors' public RSNA-knee label sets (barun2104, dreaddevelopment, yunusgmsoy) — no domain shift.

## Template
```text
### exp-XXX — YYYY-MM-DD
- config: configs/...
- code: <git sha>
- folds: data/folds/folds_v1.csv
- OOF macro_auc: 
- per-label highlights: 
- public LB: 
- runtime_s (cold): 
- conclusion: keep / kill / iterate because ...
```
