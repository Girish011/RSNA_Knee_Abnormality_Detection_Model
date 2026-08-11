# STATUS

Last updated: 2026-08-11

## Phase
**Rank-1 push: weak_labels_v2 + DINOv2-B path.** Frozen S 5-fold floor ≈ **0.72** stays the baseline to beat. No LB submit until OOF clearly lifts.

## Best scores
| Fold | Best val macro AUC (frozen S + weak_v1) |
|---|---|
| 0 | **0.764** |
| 1 | 0.732 |
| 2 | 0.725 |
| 3 | 0.697 |
| 4 | 0.675 |
| **Mean (0–4)** | **~0.719** |
| Public LB top | ~0.942 |

## Rank-1 levers (see `docs/RANK1_ROADMAP.md`)
1. Multilingual weak labels **v2** (FR/DE/PT/NL + EN/ES) — largest language was under-covered in v1
2. Retrain same frozen S on v2 → isolate label lift vs 0.719
3. DINOv2-B + freeze→gentle unfreeze (`unfreeze_lr_mult=0.05`) + `pos_weight`
4. Richer cache / expert fine-tune / OOF blend — then one LB probe

## Next 3 actions
1. Local: audit → `weak_labels_v2.csv`; `bash scripts/package_kaggle_datasets.sh`; re-upload **rsna-knee-code**
2. Kaggle: paste `notebooks/10_train_baseline.py` — 5-fold frozen S on **v2** (Save Version `…-weak-v2`)
3. If mean OOF ≫ 0.72: fetch/upload Vit-B weights; fold0 `notebooks/30_train_main.py`. Else iterate FR patterns.

## Do not
- Submit ~0.72 models
- Aggressive unfreeze at full LR
- Skip packaging before Kaggle train (code dataset must include v2 CSV)

## Session log
- Fold0–4 frozen S + v1 → mean ~0.719
- Shipped v2 labels + B config + pos_weight trainer + RANK1 roadmap
- Kaggle train failed offline: `torch.hub` needed GitHub — fixed by vendoring `third_party/dinov2` + local hub load
