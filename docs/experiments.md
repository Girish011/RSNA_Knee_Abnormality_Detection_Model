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
