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
