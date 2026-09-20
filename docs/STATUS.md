# STATUS

Last updated: 2026-09-21

## Phase
**Greenfield Imaging Playbook**. Unfreeze killed; volume honest retest running.
Coding: **gf_v1-true-oof-seed42-5fold RUNNING**.

## Just decided
- **gf_v0u unfreeze → COLLAPSE / KILL**
  - frozen max gold **0.6317** → unfrozen max **0.5177** (Δ **−0.114**)
  - ep0–4 reproduction OK; collapse on first unfrozen epoch
  - no multi-seed; DECISIONS 2026-08-12 confirmed with paired gold

## Active experiment
- Kernel: `girishbose/gf-v1-true-oof-seed42-5fold` (T4)
- Cache: existing `girishbose/gf-cache-v1` (24×224, no rebuild)
- Meta: `girishbose/rsna-knee-gf-v1-meta`
- Gate vs v0 seed42 **0.6144**: KILL <0.5944; PROMISING ≥0.6344; else INCONCLUSIVE → kill

## Closed levers this cycle
| Lever | Verdict |
|---|---|
| train-longer | KILL |
| lig3 clean Med Men | GATE FAIL |
| unfreeze lr×0.05 | **COLLAPSE −0.114** |
| volume 24-slice | prior kill VOID → **retesting** |

## Next 3 actions
1. User reports when volume kernel finishes (do not poll).
2. Read `oof_gold58_metrics.json` verdict.
3. PROMISING → seeds 1337+2024; else structural / MRI-CORE re-plan.

## Do not
- Poll Kaggle; unfreeze again; Med Men fills; treat 0.6144/0.728 as floors
- Reports at test time; KneeCoT
