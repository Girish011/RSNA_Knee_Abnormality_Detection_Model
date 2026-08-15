# STATUS

Last updated: 2026-08-15

## Phase
**Next lever: train-only report labels v3 (zero-shot multilingual NLI).** Image-side (cache/B/expert-FT) stalled at ~0.76 fold0. Do not submit.

## Best scores
| Setup | Fold0 best |
|---|---|
| **S + weak_v1 frozen, cache_v1** | **0.764** (ckpt missing) |
| **B + weak_v1 fully frozen, cache_v1** | **0.759** ← keep |
| B + expert head-FT | 0.760 then 0.745 (kill) |
| S + cache_v2 | 0.738 |
| Public LB top | ~0.942 |

## Next 3 actions
1. Kaggle: paste `notebooks/14_pseudo_label_reports.py` (GPU T4x2, **Internet ON**)
2. Download `weak_labels_v3.csv`; check expert audit vs v1 (~F1 0.44)
3. If audit prec stays high and coverage/F1 lifts: pack into `rsna-knee-code`, retrain frozen B fold0 vs **0.759**

## Do not
- Submit; unfreeze; cache_v2 5-fold; weak_v2; ship FT weights
- Use reports at test time
