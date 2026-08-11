# STATUS

Last updated: 2026-08-11

## Phase
**5-fold frozen DINOv2-S baseline done** — mean val macro AUC **~0.72**. Too low to submit vs public LB ~0.94. Next: improve labels/model, not LB probes.

## Best scores
| Fold | Best val macro AUC |
|---|---|
| 0 | **0.764** |
| 1 | 0.732 |
| 2 | 0.725 |
| 3 | 0.697 |
| 4 | 0.675 |
| **Mean (0–4)** | **~0.719** |
| Public LB top | ~0.942 |

## Read of results
- Pipeline is stable (all folds train, loss falls).
- Fold spread is wide (0.675–0.764) → weak-label noise + limited expert signal.
- Frozen ViT-S + 3×12×224 cache is a floor, not a winning recipe.

## Next 3 actions
1. **Save Version** now: `baseline-dinov2s-5fold-frozen-mean072` (keep all `fold*/fold*_best.pt`)
2. Improve supervision: stronger multilingual weak labels + expert-only fine-tune stage
3. Scale model: DINOv2-B and/or more slices/series — only then consider first public submit

## Do not
- Submit this ~0.72 model yet (wastes daily quota)
- Unfreeze backbone aggressively (already shown to collapse)

## Session log
- Fold0 frozen 5ep → 0.764; folds 1–4 complete (mean ~0.707 without fold0; ~0.719 with).
