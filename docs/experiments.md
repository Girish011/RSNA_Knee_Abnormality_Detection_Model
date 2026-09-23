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

### 2026-09-08 — Step A metadata EDA (local CSVs)
- code: `notebooks/01_data_audits.py`
- data: local `train.csv` + `train_series.csv` (Downloads); no DICOM decode
- findings: 4407 studies, 24371 series, series/study 3–14 (mean 5.53); planes Sag 9864 / Cor 8609 / Ax 5898; all studies have all 3 planes; Fluid_Sensitive paired with Fat_Suppression; gold-58 = 1.32%, ACL+ rate 0.414 in gold
- artifacts: `outputs/eda/step_a_summary.txt`, `site/assets/eda_planes.png`, `site/assets/eda_gold_rates.png`
- conclusion: keep as Post 02 evidence; next = greenfield baseline planning (Post 03 / Step B), not resume old thin-cache stack

### 2026-09-13 — gf_baseline_v0 fold0 train (Kaggle GPU, interactive)
- config: `configs/gf_baseline_v0.yaml`
- cache: `girishbose/rsna-knee-cache-gf-v0` (4407 npz, 3×12×224, plane-covered)
- meta: `girishbose/rsna-knee-gf-v0-meta` (weak_v1 + folds_v1 + src)
- fold 0; epochs 5; freeze_epochs 5; dinov2_vits14; lr 3e-4; pos_weight 1.0
- studies with any weak label: 2449; train/val 1953/496 (all cached)
- weak-val macro_auc by epoch: 0.670 → 0.701 → 0.732 → **0.742** → 0.741 (best ep3)
- artifacts: **not persisted** (interactive `/kaggle/working` only; later gold cell FileNotFound)
- conclusion: pipeline works; weak-val smoke only; need Save Version / Dataset for ckpt.

### 2026-09-13 — gf_baseline_v0 fold0 retrain + gold-58 (API kernel)
- kernel: `girishbose/gf-baseline-v0-fold0` v2 (`machine_shape: NvidiaTeslaT4`; v1 P100 failed CUDA)
- same config/cache/meta/fold/epochs as above; no fixed seed (weak-val ≠ prior 0.742)
- weak-val by epoch: 0.657 → 0.677 → 0.710 → 0.713 → **0.718** (best ep4)
- **gold-58 macro_auc: 0.7281** (n=58, all 12 labels defined)
- per-label gold: Effusion 0.903, Medial OA 0.870, Synovitis 0.792, PF OA 0.786, Lateral OA 0.758, Fracture 0.731, Baker's 0.696, Med Men 0.695, Lat Men 0.665, ACL 0.654, Contusion 0.628, MCL 0.560
- artifacts Dataset: `girishbose/rsna-knee-gf-v0-fold0` (`fold0_best.pt`, OOF, `fold0_gold58_*.{csv,json}`)
- local copy: `outputs/kaggle_download/gf-baseline-v0-fold0/gf_baseline_v0/`
- conclusion: **first greenfield gold ruler = 0.728**. Weak spots MCL/ACL/Contusion. Next = Post 03 + choose volume iterate vs multi-fold; no LB until OOF win.

### 2026-09-13 — gf_baseline_v1 volume iterate launched (12→24 slices)
- decision: volume axis first (Post 04 writeup later); single change vs v0
- config: `configs/gf_baseline_v1.yaml` (same 1 sag+cor+ax picks as v0, **n_slices=24**, 224, frozen DINOv2-S, weak_v1, fold0, seed=42)
- keep/kill: gold-58 macro ≥ **0.7281 + 0.005 = 0.7331**
- not changing: series count, resolution, teacher, unfreeze (DECISIONS: no unfreeze yet)
- code: `scripts/build_cache.py` gains `--picks-csv` for exact v0 series UIDs
- meta Dataset: `girishbose/rsna-knee-gf-v1-meta`
- cache kernel: `girishbose/gf-cache-v1` (CPU; writes `/kaggle/working/cache_gf_v1`)
- train+gold kernel (queued after cache): `girishbose/gf-baseline-v1-fold0` (T4; kernel_sources cache output)
- conclusion: **running** — await cache then fold0 gold vs v0 floor.

### 2026-09-14 — gf_baseline_v1 fold0 + gold-58 → **KILL**
- kernel: `girishbose/gf-baseline-v1-fold0` COMPLETE (T4; cache from `girishbose/gf-cache-v1`)
- cache: 4407 npz, shape (3, 24, 224, 224); same picks as v0
- seed 42; frozen DINOv2-S 5ep; weak_v1
- weak-val by epoch: 0.683 → 0.716 → 0.723 → 0.722 → **0.723** (best ep4)
- **gold-58 macro_auc: 0.7089** (n=58) vs v0 **0.7281** (Δ **-0.019**); keep threshold was 0.7331
- per-label gold: Effusion 0.847, PF OA 0.802, Synovitis 0.793, Lat OA 0.778, Contusion 0.753, Med OA 0.750, Baker's 0.717, Lat Men 0.691, Fracture 0.660, Med Men 0.649, ACL 0.615, **MCL 0.451**
- vs v0: Contusion up; OA/Effusion/ACL/MCL down; MCL below chance
- artifacts local: `outputs/kaggle_download/gf-baseline-v1-fold0/gf_baseline_v1/`
- conclusion: **KILL 24-slice volume**. Do not make cache_gf_v1 the default. Next volume lever = **resolution 224→336** (same 12 slices / same picks), not more series.

### 2026-09-14 — gf_baseline_v2 volume iterate launched (224→336)
- config: `configs/gf_baseline_v2.yaml` (same picks as v0, **12 slices**, **image_size=336**, frozen DINOv2-S, weak_v1, fold0, seed=42)
- keep/kill: gold-58 ≥ **0.7331**
- meta: `girishbose/rsna-knee-gf-v2-meta`
- cache kernel: `girishbose/gf-cache-v2`
- train kernel (after cache): `girishbose/gf-baseline-v2-fold0` (T4)
- conclusion: **running**

### 2026-09-14 — gf_baseline_v2 fold0 + gold-58 → **KILL**
- kernel: `girishbose/gf-baseline-v2-fold0` COMPLETE (T4; cache from `gf-cache-v2`)
- cache: 4407 npz, shape (3, 12, 336, 336); same picks as v0
- xFormers missing warnings: ignore (expected without xformers)
- weak-val by epoch: 0.685 → 0.673 → 0.703 → 0.698 → **0.704** (best ep4)
- **gold-58 macro_auc: 0.6925** vs v0 **0.7281** (Δ **-0.036**); keep threshold 0.7331
- per-label gold: Med OA 0.829, Lat OA 0.764, Effusion 0.753, Baker's 0.748, Lat Men 0.743, PF OA 0.718, Synovitis 0.698, Contusion 0.676, Med Men 0.657, ACL 0.657, Fracture 0.636, **MCL 0.431**
- artifacts local: `outputs/kaggle_download/gf-baseline-v2-fold0/gf_baseline_v2/`
- conclusion: **KILL 336px volume**. Both single-axis volume levers (slices, res) lost vs v0. Stop pure volume on 3-series recipe; next = multi-fold OOF on v0 or better teacher/labels.

### 2026-09-15 — gf_v0 5-fold OOF launched
- recipe: same `cache_gf_v0` / weak_v1 / frozen DINOv2-S / 5ep / seed **42**
- kernel: `girishbose/gf-baseline-v0-5fold` (T4; trains folds 0–4 then aggregates)
- meta updated with seed-enabled `train_baseline_fold.py`
- metric of interest: **true OOF gold-58** (holdout-fold preds only); also weak OOF macro
- note: prior 0.7281 was fold0 model on all 58 (some gold were in fold0 train)
- conclusion: **running**

### 2026-09-15 — gf_v0 5-fold OOF COMPLETE (honest floor)
- kernel: `girishbose/gf-baseline-v0-5fold` COMPLETE (~3.8h wall)
- weak OOF macro AUC: **0.6855**
- **true OOF gold-58 macro AUC: 0.6144** (n=58, all 12 labels)
- per-fold weak-val best: f0 0.725 / f1 0.688 / f2 0.743 / f3 0.724 / f4 0.690
- per-label gold OOF: Effusion 0.889, Baker's 0.723, Synovitis 0.677, Lat Men 0.647, PF OA 0.624, Med OA 0.620, Contusion 0.607, Fracture 0.594, Lat OA 0.592, Med Men **0.486**, ACL **0.473**, MCL **0.440**
- vs prior fold0→all58 gold 0.7281: Δ **-0.114** (that number was optimistic)
- artifacts: `outputs/kaggle_download/gf-baseline-v0-5fold/gf_baseline_v0_5fold/`
- conclusion: **adopt 0.614 as honest v0 floor**. No LB. Next lever = labels/teacher (volume already failed); ACL/MCL/meniscus near chance on gold.

### 2026-09-15 — gf_labels_lig1 teacher built + 5-fold OOF launched
- teacher: `weak_labels_gf_lig1.csv` = weak_v1 + gap-fill only NaN cells for **ACL / MCL / Medial Meniscus**
- fills: `weak_labels_v7` (TR/EL) + patched EN injury/grade patterns; expert override on gold-58
- coverage: ACL 429→1254 (+825), MCL 262→700 (+438), Med Men 866→1460 (+594); any-label 2449→3235
- pure extractor on gold (conf≥0.5): ACL prec/rec 0.77/0.89; MCL 0.53/1.00 (watch noise); Med Men 0.71/0.94
- train: same cache_gf_v0, frozen-S, seed 42, 5-fold OOF
- kernel: `girishbose/gf-labels-lig1-5fold`; meta `girishbose/rsna-knee-gf-lig1-meta`
- keep if OOF gold ≥ **0.6194**
- conclusion: **running** — user will report when done (no poll).

### 2026-09-15 — gf_labels_lig1 5-fold OOF COMPLETE → **KILL**
- kernel: `girishbose/gf-labels-lig1-5fold` COMPLETE
- weak OOF macro: **0.7073** (vs v0 0.685)
- **true OOF gold-58: 0.6019** vs floor **0.6144** (Δ **−0.0125**); keep thr 0.6194 → KILL
- focus labels vs v0 OOF gold: Med Men 0.486→**0.614** (+0.128), ACL 0.473→0.516 (+0.043), MCL 0.440→0.476 (+0.036)
- regressions: Effusion 0.889→0.697, Lat OA 0.592→0.484, Synovitis 0.677→0.607, PF OA 0.624→0.582
- artifacts: `outputs/kaggle_download/gf-labels-lig1-5fold/` (download pending if not local yet)
- conclusion: **KILL lig1-as-shipped**. Gap-fill helped target labels but hurt macro via label noise / distribution shift. Next = drop MCL fills (and maybe tighten ACL) or Med-Men-only gap-fill A/B.

### 2026-09-15 — gf_labels_lig2 teacher built + 5-fold OOF launched
- teacher: `weak_labels_gf_lig2.csv` = weak_v1 + gap-fill only NaN cells for **Medial Meniscus** (no ACL, no MCL)
- fills: same v7+patches extractor as lig1; expert override on gold-58
- coverage: Med Men 866→1460 (+594, same as lig1); ANY 2449→3023 (lig1 was 3235); non-gold ACL/MCL identical to weak_v1
- gold audit (conf≥0.5): Med Men prec/rec 0.71/0.94
- train: same cache_gf_v0, frozen-S, seed 42, 5-fold OOF
- kernel: `girishbose/gf-labels-lig2-5fold`; meta `girishbose/rsna-knee-gf-lig2-meta`
- keep if OOF gold ≥ **0.6194**
- conclusion: launched; result below.

### 2026-09-16 — gf_labels_lig2 5-fold OOF COMPLETE → **KILL** (but inside ruler noise)
- kernel: `girishbose/gf-labels-lig2-5fold` COMPLETE (~15,990 s wall)
- weak OOF macro: **0.6774** (v0 0.6855, lig1 0.7073)
- **true OOF gold-58: 0.6103** vs floor **0.6144** (Δ **−0.0042**); keep thr 0.6194 → **KILL** by the rule
- per-label gold OOF: Effusion 0.734, Med OA 0.671, Baker's 0.674, Lat OA 0.644, **Med Men 0.623**, Lat Men 0.619, Contusion 0.611, Fracture 0.590, ACL 0.583, Synovitis 0.570, PF OA 0.537, MCL 0.467
- Med Men reproduces the lig1 gain: v0 0.486 → **0.623 (+0.137)**; lig1 was +0.129
- artifacts: `outputs/kaggle_download/gf-labels-lig2-5fold/gf_labels_lig2_5fold/`
- conclusion: **KILL lig2-as-shipped**, but see the ruler-noise entry below — this Δ is not distinguishable from noise.

### 2026-09-16 — Ruler noise audit: the 58-gold OOF ruler cannot resolve 0.005
- code: `scripts/gold_oof_ab.py` (paired study-level bootstrap on the shared 58 gold studies, 5000 draws)
- **paired bootstrap deltas (candidate − baseline):**
  | A/B | Δ macro | 95% CI | P(cand > base) |
  |---|---|---|---|
  | lig2 vs v0 | −0.0042 | [−0.049, +0.041] | 0.435 |
  | lig1 vs v0 | −0.0126 | [−0.055, +0.030] | 0.276 |
  | lig2 vs lig1 | +0.0084 | [−0.043, +0.058] | 0.623 |
- delta sd ≈ **0.022**; single-run bootstrap sd ≈ **0.027**. Our keep margin is **0.005** → the instrument is ~4–5× coarser than the effect we are trying to read. **Both lig1 and lig2 "kills" are coin flips.**
- **direct noise proof (macro-level version of the 2026-08-27 finding):** lig2 changed *only* the Medial Meniscus teacher column, yet labels with **byte-identical** teacher cells moved wildly — ACL **0.473 → 0.583 (+0.110)**, Effusion **0.889 → 0.734 (−0.155)**, Synovitis −0.108, PF OA −0.088. The 11 unchanged labels have delta sd **0.079**, max |delta| 0.155.
- **only reproducible effect in the campaign so far:** Med Men gap-fill, +0.129 (lig1) and +0.137 (lig2) across two independent runs.
- **confound found in both gap-fill A/Bs:** filling NaN cells also *adds studies to training* (any-label 2449 → 3235 lig1 / 3023 lig2), so neither run was a single-factor label change; the new studies carry one supervised label each and shift the shared trunk.
- **variance source found:** weak-val is still **rising at the final epoch** in 8/10 folds (5 epochs = undertrained), and the shipped checkpoint is chosen by best weak-val, a ruler that disagrees with gold. Adjacent-epoch weak-val gaps are 0.005–0.041 while gold swings ±0.1.
- conclusion: **stop A/B-ing teachers against this ruler.** Next step must either quieten the ruler (seed replicates / seed-averaged OOF, fix undertraining + checkpoint selection) or accept per-label evidence for the one reproducible effect. Do not read further sub-0.02 macro deltas as signal.

### 2026-09-16 — gf_v0 seed replicates launched (measure the ruler, not a new recipe)
- goal: measure the **seed-to-seed sd** of true OOF gold-58 macro AUC, then derive the keep margin from it and build a variance-reduced (seed-averaged) baseline.
- recipe: identical to the seed-42 v0 5-fold in every respect — same `girishbose/rsna-knee-gf-v0-meta` code/labels, same `cache_gf_v0`, 5 epochs, frozen DINOv2-S, weak_v1. **Only `train.seed` differs.**
- mechanism: the trainer reads `seed` from the config and has no `--seed` flag, so each kernel copies `configs/gf_baseline_v0.yaml` at runtime, rewrites the single `seed:` line, and **asserts exactly one line changed** (verified locally: `'  seed: 42' -> '  seed: 1337'`, parsed configs equal apart from seed). No dataset re-upload, so the code cannot drift from the original run.
- kernels: `girishbose/gf-v0-seed1337-5fold`, `girishbose/gf-v0-seed2024-5fold` (both T4, RUNNING, ~4.4 h each, 2 concurrent GPU sessions)
- analysis tool (written before results, rule pre-registered): `scripts/seed_variance_report.py`
- **pre-registered rule:** new baseline = seed-averaged OOF gold macro over seeds {42, 1337, 2024}; new keep margin = **max(0.005, 2 × measured seed sd)**; future candidates judged as seed-averaged predictions against that threshold.
- **expected outcome either way is informative:** small sd ⇒ the lig1/lig2 kills were real after all and 0.005 was defensible; large sd (≥0.01) ⇒ every greenfield verdict to date was noise and the margin must widen.
- conclusion: launched; result below.

### 2026-09-16 — gf_v0 seed replicates COMPLETE → **0.6144 was never a floor; it was the luckiest of three draws**
- kernels: `gf-v0-seed1337-5fold`, `gf-v0-seed2024-5fold` both COMPLETE. Identical recipe, only `train.seed` differs.
- **true OOF gold-58 macro by seed:**
  | seed | gold OOF | weak OOF |
  |---|---|---|
  | 42 (original "floor") | **0.6144** | 0.6855 |
  | 1337 | **0.5720** | 0.6716 |
  | 2024 | **0.5889** | 0.6870 |
- **mean 0.5918, sd 0.0214, range 0.0424.** Same code, same labels, same cache, same folds — the entire 0.042 spread is noise.
- **the floor was a fluke:** the 0.6144 we have been ranking every greenfield experiment against is +1.06 sd above the recipe's own mean. The honest description of gf_v0 is **0.592 ± 0.021**.
- **the lig1/lig2 kills were artifacts:** lig1 0.6019 and lig2 0.6103 both sit **above** the v0 seed mean 0.5918. They were killed for failing to beat a lucky draw. Neither is a proven win either (both are single-seed draws) — they are simply *indistinguishable* from v0.
- **the "Effusion collapse" was regression to the mean:** Effusion has the worst per-label seed sd, **0.133** (seed42 0.889 vs seed1337 0.631 vs seed2024 0.707). The 0.889 that made lig1/lig2 look catastrophic on Effusion was itself the outlier. Other high-variance labels: Lateral OA 0.090, Baker's 0.086, MCL 0.080, Med Men 0.066, Contusion 0.064, Fracture 0.060. Stable: PF OA 0.012, Lat Meniscus 0.012, ACL 0.022, Synovitis 0.028.
- **seed-averaging works:** averaging the 3 seeds' OOF probabilities gives gold macro **0.6173**, above *every* individual seed including the lucky 0.6144. Bootstrap 95% CI [0.567, 0.667].
- **the weak ruler is 2.5× quieter than the gold ruler:** weak OOF sd **0.0085** (n≈2449) vs gold sd **0.0214** (n=58), same models. The gold ruler amplifies model-to-model differences because 58 studies give each label only ~10–25 positives. Small n, not just training stochasticity, is doing the damage.
- **pre-registered rule now instantiated:** baseline = seed-averaged OOF gold **0.6173**; margin = max(0.005, 2 × 0.0214) = **0.0427**; single-seed keep threshold **0.6600**. A single-seed A/B would need a +0.043 gain to be credible, which is not a practical bar — hence multi-seed runs become mandatory.
- artifacts: `docs/audit/gf_v0_seed_variance.json`, `outputs/kaggle_download/gf-v0-seed{1337,2024}-5fold/`
- conclusion: **every greenfield keep/kill verdict to date (v1, v2, lig1, lig2) was decided inside the noise band and none of them are trustworthy.** Stop single-seed A/Bs. Report recipes as 3-seed mean ± sd and compare seed-averaged predictions.

### 2026-09-16 — gf_v0c stage 1 launched: converge the recipe (paired, one seed)
- hypothesis: the recipe is stopped mid-climb and its checkpoint is chosen badly. Weak-val was still **rising at the final epoch in 8/10 folds**, and the shipped checkpoint is the weak-val peak — a ruler that disagrees with gold. Both cost score and add variance.
- config: `configs/gf_baseline_v0c.yaml` — identical to gf_v0 (3×12×224, frozen DINOv2-S, weak_v1, cache_gf_v0, seed 42, constant LR) with **epochs 5 → 10**; `freeze_backbone_epochs=10` so the unfreeze branch never fires (DECISIONS 2026-08-12 respected).
- **why one seed is defensible here:** there is no LR schedule and no unfreeze, so epochs 0–4 of this run retrace the 5-epoch seed-42 run exactly. "Epoch 4 vs epoch 9" and "policy A vs policy B" are therefore **paired within one training trajectory, with zero seed noise** — unlike the cross-run deltas that misled us. The kernel prints epochs 0–4 weak-val against the recorded seed-42 values as a reproduction check.
- code: `train_baseline_fold.py --save-epoch-oof` (new, defaults off so the old path is unchanged) writes val predictions every epoch; `src/rsna_knee/epoch_policies.py` (new, 7 unit tests) assembles OOF for any fold→epochs selection.
- policies compared offline from the SAME trained models, no extra GPU: `best_weakval_first5` (= old recipe), `best_weakval_all10`, `final_epoch`, `avg_last3`, `avg_last5`, plus a per-epoch gold curve for all 10 epochs.
- kernel: `girishbose/gf-v0c-conv-seed42-5fold` (T4, ~8 h est: ~9.6 min per fold-epoch × 50); meta `girishbose/rsna-knee-gf-v0c-meta`
- ruler: seed-averaged v0 baseline **0.6173**, seed sd 0.0214, margin **0.0427**. **This single seed cannot produce a keep/kill.** If a policy shows a real paired gain, seeds 1337 + 2024 follow (~18 h) for the 3-seed verdict.
- staged deliberately to protect the 30 h weekly quota: ~8 h now, and we only spend the remaining ~18 h if stage 1 looks promising.
- conclusion: **ABORTED** mid-run — user reported GPU quota exhausted and asked to stop. Kernel still showed RUNNING via API; cancel requires the Kaggle UI (API token lacks `kernelSessions.cancel`). Resume same staged job when quota resets; do not re-upload meta unless code changes.

### 2026-09-19 — gf_v0c stage 1 resumed (quota reset)
- prior: v1 CANCEL_ACKNOWLEDGED after GPU exhaustion
- action: re-pushed same kernel + same meta (no rebuild) → **v2 RUNNING**
- kernel: `girishbose/gf-v0c-conv-seed42-5fold` v2; meta `girishbose/rsna-knee-gf-v0c-meta`
- conclusion: launched; result below.

### 2026-09-20 — gf_v0c stage 1 COMPLETE → longer training FAILS gold; selection policy is the real finding
- kernel: `girishbose/gf-v0c-conv-seed42-5fold` v2 COMPLETE; artifacts `outputs/kaggle_download/gf-v0c-conv-seed42-5fold/`; audit `docs/audit/gf_v0c_convergence_seed42.json`
- **reproduction check PASSED** (pairing property holds): epochs 0–4 weak-val match the recorded seed-42 5-epoch run to 0.001 on every fold (max |drift| = 0.000 after rounding).
- **per-epoch true OOF gold-58 (all folds, same epoch):**
  | ep | gold | weak |
  |---|---|---|
  | 0 | 0.5807 | 0.6239 |
  | 1 | 0.5956 | 0.6628 |
  | 2 | 0.5951 | 0.6707 |
  | 3 | 0.6001 | 0.6825 |
  | **4** | **0.6317** | 0.6894 |
  | 5 | 0.6150 | 0.6803 |
  | 6 | 0.6136 | 0.6943 |
  | 7 | 0.6276 | 0.6943 |
  | 8 | 0.6162 | 0.7049 |
  | 9 | 0.6110 | 0.6991 |
- **Gold peaks at epoch 4 and declines.** Weak keeps rising through ep8 (0.689→0.705). Weak−gold gap grows 0.058→0.088 → teacher overfit: more epochs fit noisy weak labels harder, not gold.
- **Checkpoint policies (same trained models, paired):**
  | policy | gold | weak |
  |---|---|---|
  | **uniform ep4** (all folds @ ep4) | **0.6317** | — |
  | avg_last5 | 0.6265 | 0.7059 |
  | avg_last3 | 0.6211 | 0.7067 |
  | best_weakval_all10 | 0.6195 | 0.7075 |
  | best_weakval_first5 (= old recipe) | **0.6144** | 0.6855 |
  | final_epoch | 0.6110 | 0.6991 |
- Old recipe matches recorded seed-42 OOF **exactly** (0.6144); fold3 picked ep1, fold4 picked ep3 on weak-val, leaving ~0.017 vs uniform ep4.
- avg_last5 vs old: **+0.012** paired — a *selection* win, not a *longer-training* win (uniform ep4 still beats it).
- conclusion: **KILL "train longer"** for this frozen recipe. Do **not** spend ~18 h on seeds 1337/2024 at 10 epochs. Escalate. Fixed-epoch / no-weak-val picking is hygiene, not a path to ~0.94.

### 2026-09-20 — Chose clean Med Men (lig3); stage 1 launched
- choice vs fork: clean Med Men fills (not unfreeze / volume / re-plan) — only reproducible signal; cheapest honest label A/B; no cache rebuild.
- teacher: `weak_labels_gf_lig3.csv` = weak_v1 + Med Men gap-fill **restricted to studies with ≥1 weak_v1 label**
- coverage: Med Men 866→886 (**+20 fills**); ANY **2449→2449** (0 new studies). Contrast lig2: +594 fills, ANY→3023 — most of lig2's Med Men lift admitted new studies.
- kernel: `girishbose/gf-labels-lig3-5fold`; meta `girishbose/rsna-knee-gf-lig3-meta`
- **pre-registered stage-1 gate:** Med Men gold OOF ≥ **0.566** (not a macro keep). Fail → kill without multi-seed. Pass → seeds 1337+2024 for seed-averaged macro vs 0.6173+0.0427.
- conclusion: launched; result below.

### 2026-09-20 — gf_labels_lig3 stage 1 COMPLETE → **GATE FAIL / KILL**
- kernel: `girishbose/gf-labels-lig3-5fold` COMPLETE
- weak OOF macro 0.682; gold OOF macro **0.6018**
- **Med Men gold 0.4868** vs v0 seed42 0.4856 (Δ **+0.001**); gate ≥ 0.566 → **FAIL**
- per-label vs v0: no Med Men lift; Effusion still strong (0.901); Med OA −0.067 (noise-scale)
- artifacts: `outputs/kaggle_download/gf-labels-lig3-5fold/`; audit `docs/audit/gf_lig3_oof_gold58_metrics.json`
- conclusion: **KILL clean Med Men fills.** The lig1/lig2 Med Men +0.13 required admitting new studies (ANY 2449→3023); fills inside the existing 2449 pool (+20 cells) do nothing. Do not run seeds 1337/2024. Label gap-fill track closed for this extractor.

### 2026-09-20 — Chose careful unfreeze (gf_v0u); stage 1 launched
- next lever: reopen DECISIONS 2026-08-12 under paired measurement (past collapses were single-seed / weak-val).
- kernel: `girishbose/gf-v0u-unfreeze-seed42-5fold` RUNNING; **reuses** `rsna-knee-gf-v0c-meta` (no dataset upload)
- recipe: ep0–4 frozen (retraces v0), ep5–7 unfrozen at lr×0.05, `--save-epoch-oof`
- gates: COLLAPSE if unfrozen_max < frozen_max−0.02; PROMISING if ≥ frozen_max+0.02; else INCONCLUSIVE → kill
- conclusion: launched; result below.

### 2026-09-21 — gf_v0u unfreeze stage 1 COMPLETE → **COLLAPSE / KILL**
- kernel: `girishbose/gf-v0u-unfreeze-seed42-5fold` COMPLETE
- reproduction: epochs 0–4 weak-val match seed42 exactly (drift 0.000)
- **gold curve:** ep4 frozen **0.6317** → ep5 unfrozen **0.5094** → ep6/7 ~0.516
- frozen_max 0.6317, unfrozen_max 0.5177, Δ **−0.114** → verdict **COLLAPSE** (margin 0.02)
- artifacts: `outputs/kaggle_download/gf-v0u-unfreeze-seed42-5fold/`; audit `docs/audit/gf_v0u_unfreeze_seed42.json`
- conclusion: **KILL unfreeze** on this recipe. Confirms DECISIONS 2026-08-12 with paired true-OOF gold. No multi-seed.

### 2026-09-21 — Chose honest 24-slice volume retest (gf_v1 true OOF); stage 1 launched
- prior v1 kill (fold0→all58 0.7089) is VOID under true-OOF ruler
- reuse existing `girishbose/gf-cache-v1` (no rebuild) + `rsna-knee-gf-v1-meta`
- kernel: `girishbose/gf-v1-true-oof-seed42-5fold` RUNNING
- gate vs v0 seed42 0.6144: KILL <0.5944; PROMISING ≥0.6344; else INCONCLUSIVE → kill
- conclusion: launched; result below.

### 2026-09-21 — gf_v1 true-OOF stage 1 COMPLETE → **INCONCLUSIVE → KILL**
- kernel: `girishbose/gf-v1-true-oof-seed42-5fold` COMPLETE
- gold OOF macro **0.6080**, weak **0.6806**; Δ vs v0 seed42 0.6144 = **−0.006**
- gate: KILL <0.5944 / PROMISING ≥0.6344 → **INCONCLUSIVE** → kill (no multi-seed)
- per-label vs v0: Med OA **−0.115**, Fracture +0.068, Effusion −0.046; rest ~noise
- artifacts: `outputs/kaggle_download/gf-v1-true-oof-seed42-5fold/`; audit `docs/audit/gf_v1_true_oof_seed42.json`
- conclusion: **KILL 24-slice volume** under honest ruler. Thin-recipe volume/adaptation/labels track closed. Next = structural encoder (MRI-CORE).

### 2026-09-21 — Chose MRI-CORE ViT-B; stage 1 launched
- wire: `create_image_encoder("mri_core_vitb")` in greenfield `src/rsna_knee`; custom teacher→SAM loader (fixes upstream pos_embed bug; skips unused SAM neck → **768-d** features)
- same `cache_gf_v0`; bilinear 224→384 at encode; batch 1; encode_chunk 8; freeze 5ep
- meta: `girishbose/rsna-knee-gf-mri-core-meta`; weights dataset reused
- kernel: `girishbose/gf-mri-core-seed42-5fold` RUNNING
- gate vs v0 seed42 0.6144 ±0.02
- conclusion: launched; result / fix below.

### 2026-09-21 — gf-mri-core v1 ERROR → v2 relaunch
- v1 fail: `AssertionError: .../competitions/.../train.csv` (competition mount missing train on this worker)
- fix: bundle `data/raw/train.csv` (no Report) into meta; kernel resolves competition → meta → rglob
- kernel: `girishbose/gf-mri-core-seed42-5fold` **v2 RUNNING**
- conclusion: **running** — user will report when done (no poll).

### 2026-09-24 — Report-label audit: public LLM labels vs keyword v1/v7 on the experts (no train)
- script: `scripts/audit_public_labels.py`; outputs `docs/audit/public_labels_vs_expert.csv`, `public_labels_agreement_nonexpert.csv`, `public_labels_audit.json`
- sources: `pilkwang/rsna-knee-llm-labels` (CC0, 4406 studies, 57/58 experts), `dreaddevelopment/rsna-knee-labels` (CC0, 4349 studies, **0 experts**), our keyword v1/v7 re-run on raw reports (no expert override)
- **macro AUC vs 57 experts:** pilk **0.870** [0.831, 0.900]; kw_v7 0.651 [0.603, 0.698]; kw_v1 0.628 [0.577, 0.677]
- paired: pilk − kw_v1 **+0.242** [0.193, 0.290]; pilk − kw_v7 +0.219 [0.166, 0.272]; kw_v1 − kw_v7 −0.023 [−0.057, +0.007]
- precision ≥ 0.5 gate: pilk passes 11/12 (fails **Contusion** 0.44); kw_v1 fails PF OA; kw_v7 fails PF OA, Lat OA, Contusion
- weakest pilk labels: Synovitis 0.694, Contusion 0.768, Lat OA 0.789 (this is the report-reading ceiling vs image-read experts)
- dread agreement on non-expert studies (mean Spearman): with pilk 0.81, kw_v7 0.41, kw_v1 0.33
- caveat: pilk's labeler docstring says its prompt was checked against the annotated studies → its 0.870 is likely optimistic; the +0.24 gap is far larger than any plausible selection effect
- conclusion: keyword extractors are retired as teachers. First train supervision = public LLM soft labels (see DECISIONS 2026-09-24). Our own LLM labels must beat pilk on the experts by a paired CI before they replace it.

### 2026-09-24 — Label calibration: pilk UNK → dread (labels_llm_blend_v2)
- on 57 experts, pilk verdict UNK cells (n=177) are expert-positive **13.6%**; pilk writes a flat 0.28 there (YES cells 65.3% positive, NO 4.1%)
- dread's value in pilk-UNK cells is 0.05–0.21 for most labels (Synovitis 0.43: reports rarely mention it)
- v2 = mean(pilk, dread) where pilk is YES/NO, dread alone where pilk is UNK; expert override on the 58
- files (git-ignored): `data/processed/labels_llm_blend_v2.{csv,parquet}`, `labels_dread.parquet`; builder `scripts/build_llm_blend_labels.py --version v2`

### 2026-09-24 — RunPod labels-ab-v1 launched (first own GPU train, paired label A/B)
- pod `v4gyvfi50q6vwh` (RTX 4090 community, $0.34/hr, 71 GB RAM, 20 vCPU); launcher `scripts/runpod_launch.py --job labels-ab-v1`; on-pod `runpod/run_job.sh`
- trainer: public `dreaddevelopment/knee-mri-training-the-twelve-finding-model` vendored as `runpod/train_knee.py` (unchanged); corpus = public 44-slot `knee-raptor-corpus` + `-ext` (4407 studies)
- recipe (both arms): `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, res 384, 16 ep, bs 8, k 12 / k_eval 24, grad ckpt, imagenet norm, seed 42; validation = 58 experts (never trained on)
- arms: **dread_s42** (`labels_dread.parquet`, reproduces the public teacher) vs **blendv2_s42** (`labels_llm_blend_v2.parquet`)
- read-out rule (pre-registered): compare best-epoch, SWA and fixed last-epoch gold AUC; best-epoch is selection-biased on gold for both arms equally. Keep blend v2 only if it wins on ≥2 of the 3 with a paired bootstrap CI not centred below 0; otherwise the teacher choice is a wash and dread stays (it is the published teacher).
- caveat: the 44-slot corpus has no public test-time preprocessor, so these weights are a measurement; submit models train on our dense80 cache
- pod history: v4gyvfi50q6vwh, 3n1waju7ptv5h8, tb1g29qyhyv8me (community host CUDA faults), 8jp8lof50epyet (secure, pip hang) → running on **mw4c7rb992jk7u** (community 3090, $0.22/hr); smoke OK; dread_s42 ep0 gold 0.7172, 1056 s/epoch
- results: bundle `girishbose/rsna-knee-rp-labels-ab-v1-bundle` → pod uploads `girishbose/rsna-knee-rp-labels-ab-v1-out` and self-terminates. Est. ~7 h, ~$2.40.

### 2026-09-24 — Dense80 train cache (Kaggle CPU, 4 shards) launched
- kernels `girishbose/rsna-knee-dense80-s0..s3` (CPU, no internet); source `outputs/kernels/dense80_cache/`
- geometry: 80 slots = Sag fluid 22 / Sag non-fluid 18 / Cor fluid 15 / Cor other 10 / Ax 15; uniform over 2–98% of each series; 140 mm crop; 336 px uint8; per-slot 2–98 percentile normalisation
- picker/ordering/normalisation copied from the public widedense inference code (primary 2–98% arm) so the test-time builder is the same function with the same constants
- outputs per shard: `vols_s{k}.npy` (~1102 × 80 × 336 × 336, ~10 GB), `masks_s{k}.npy`, `ids_s{k}.npy`, `meta_s{k}.json` (errors, empty studies, minutes)
- next: when the user reports done, pull with `kaggle kernels output girishbose/rsna-knee-dense80-s{k}` onto a RunPod volume
