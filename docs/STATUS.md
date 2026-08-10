# STATUS

Last updated: 2026-08-11

## Phase
**Baseline improving** — fold0 frozen-backbone 5ep val macro AUC **0.764** (up from 0.685). Still below public LB ~0.94; no submit yet.

## Best scores
| Split | Macro AUC | Notes |
|---|---|---|
| Val fold0 | **0.764** | frozen DINOv2-S, 5 epochs, weak+expert labels |
| Prior fold0 | 0.685 | 3ep; unfreeze collapsed later epochs |
| Public LB | — | do not submit until multi-fold OOF is stronger |
| Public LB top | ~0.942 | reference only |

## Train curve (fold0, frozen)
0.701 → 0.727 → 0.739 → 0.748 → **0.764** (monotonic — good sign)

## Next 3 actions
1. Save Version named e.g. `baseline-dinov2s-fold0-frozen-auc0764`
2. Train folds **1–4** with the same frozen recipe (or overnight Save & Run All loop)
3. After 5-fold OOF, decide: submit probe only if OOF ≳ 0.78–0.80; else improve weak labels / DINOv2-B / resolution

## Session log
- Unfreeze hurt; freeze-only 5ep reached 0.764.
