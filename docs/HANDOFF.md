# Handoff — greenfield campaign (2026-09-15)

Paste into a **new chat**. Source of truth: this file + `docs/STATUS.md` + `docs/SERIES.md`
+ latest entries in `docs/experiments.md` + `docs/DECISIONS.md`.

## Read first
1. `docs/STATUS.md`
2. `docs/SERIES.md`
3. `docs/DECISIONS.md` (tail from 2026-09-13)
4. Tail of `docs/experiments.md` (from 2026-09-08 / greenfield)

## Where we are (one paragraph)
Greenfield campaign on RSNA Knee. Blog Posts 01–03 done. Image recipe floor is **gf_v0**: 3 series × 12 slices × 224, frozen DINOv2-S, weak_v1, seed 42, cache `girishbose/rsna-knee-cache-gf-v0`. Honest ruler is **5-fold true OOF full-58 gold macro AUC = 0.6144** (not the old fold0→all58 0.728). Volume iterates (24 slices, 336px) both KILLED. First label iterate **gf_labels_lig1** (gap-fill ACL+MCL+Med Men) KILLED: OOF gold **0.6019**, though Med Men +0.13. Second label iterate **gf_labels_lig2** (Med-Men-only gap-fill; drop MCL and ACL fills) is **RUNNING**. Keep if OOF gold ≥ **0.6194**.

## Scoreboard
| Run | Weak OOF | Gold OOF | Verdict |
|---|---|---|---|
| gf_v0 weak_v1 | 0.685 | **0.6144** | floor |
| gf_v1 24-slice | — | (old ruler) | KILL |
| gf_v2 336px | — | (old ruler) | KILL |
| gf_labels_lig1 | 0.707 | **0.6019** | KILL |
| gf_labels_lig2 | — | — | running |

## Immediate next (when kernel done)
Download `girishbose/gf-labels-lig2-5fold` → `outputs/kaggle_download/gf-labels-lig2-5fold/`. Read `oof_gold58_metrics.json`. KEEP if gold ≥ 0.6194; else KILL. Do not poll.

## Working rules
- Do **not** poll/watch Kaggle; user reports when jobs finish
- No reports / KneeCoT / LLM APIs at submit
- No public LB until OOF gold win ≥ floor + 0.005
- Repo docs beat chat history

## Key artifacts
| Item | Slug / path |
|---|---|
| Cache | `girishbose/rsna-knee-cache-gf-v0` |
| v0 5-fold | `girishbose/gf-baseline-v0-5fold` → `outputs/kaggle_download/gf-baseline-v0-5fold/` |
| lig1 (killed) | `girishbose/gf-labels-lig1-5fold` → `outputs/kaggle_download/gf-labels-lig1-5fold/` |
| lig2 (running) | `girishbose/gf-labels-lig2-5fold`; meta `girishbose/rsna-knee-gf-lig2-meta` |
| Ligament teacher | `src/rsna_knee/text/gf_ligament_teacher.py`, `scripts/build_gf_ligament_labels.py` |
| Configs | `configs/gf_baseline_v0.yaml`, `configs/gf_labels_lig1.yaml`, `configs/gf_labels_lig2.yaml` |

## New-chat opener (copy/paste)
> Read `docs/STATUS.md`, `docs/HANDOFF.md`, `docs/SERIES.md`, and the latest `docs/experiments.md` / `docs/DECISIONS.md`.
> Continue greenfield: floor is gf_v0 true OOF gold **0.6144**. Volume KILLED. lig1 KILLED (0.602). lig2 (Med-Men-only) is the active job — I will tell you when it finishes; do not poll. Keep ≥ 0.6194. Do not reopen KneeCoT or put reports in submit. Do not treat 0.728 as the floor.
