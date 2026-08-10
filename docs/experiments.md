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
