# Experiments registry

Append one row (or block) per run. Never delete history.

| ID | Date | Config | Fold | OOF macro | Public LB | Runtime s | Notes / conclusion |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | No runs yet |

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
