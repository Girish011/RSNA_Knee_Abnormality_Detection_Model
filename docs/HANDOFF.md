# Handoff — greenfield campaign (2026-09-15)

Paste into a **new chat**. Source of truth: this file + `docs/STATUS.md` + `docs/SERIES.md`
+ latest entries in `docs/experiments.md` + `docs/DECISIONS.md`.

## Read first
1. `docs/STATUS.md`
2. `docs/SERIES.md`
3. `docs/DECISIONS.md` (tail from 2026-09-13)
4. Tail of `docs/experiments.md` (from 2026-09-08 / greenfield)

## Where we are (one paragraph)
Greenfield on RSNA Knee. Thin DINOv2-S recipe closed (train-longer / lig3 / unfreeze / 24-slice). Live: **MRI-CORE ViT-B** stage-1 true OOF on `cache_gf_v0` (`gf-mri-core-seed42-5fold`). Seed-avg baseline still **0.6173**.

## Active job
`girishbose/gf-mri-core-seed42-5fold` **v2 RUNNING** (v1 died: competition train.csv missing).
Meta now bundles `data/raw/train.csv`. Gate vs v0 seed42 0.6144 (±0.02).

When done:
1. Download → `oof_gold58_metrics.json` verdict.
2. PROMISING → seeds 1337+2024. Else → DINOv2-B / aggregator.
Do not poll.

## The ruler (use these numbers)
| Item | Value |
|---|---|
| Baseline: seed-averaged OOF gold (42/1337/2024) | **0.6173** |
| Measured seed sd | **0.0214** |
| Margin = max(0.005, 2×sd) | **0.0427** |
| Weak-ruler sd (n≈2449, for contrast) | 0.0085 |
| Worst per-label seed sd | Effusion 0.133, Lat OA 0.090, Baker's 0.086, MCL 0.080 |
| Best per-label seed sd | PF OA 0.012, Lat Men 0.012, ACL 0.022, Synovitis 0.028 |

Rule: **≥3 seeds per A/B, compare seed-averaged predictions.** Never decide on one seed.
Tools: `scripts/seed_variance_report.py` (seed spread + seed-averaged baseline + margin),
`scripts/gold_oof_ab.py` (paired bootstrap between two runs).
Audit: `docs/audit/gf_v0_seed_variance.json`.

## Levers still on the table
Thin DINOv2-S recipe levers are **closed**. Live / remaining:
1. **MRI-CORE ViT-B** — stage-1 RUNNING (`gf-mri-core-seed42-5fold`).
2. **DINOv2-B / study aggregator** — if MRI-CORE fails.
3. **Efficiency student** — parallel once a main encoder clears a real bar.

## Ruler noise numbers (use these, not vibes)
- paired bootstrap delta sd ≈ **0.022**; single-run bootstrap sd ≈ **0.027**; keep margin 0.005 is **unresolvable**
- 11 identical-teacher-cell labels in lig2 vs v0: delta sd **0.079**, max |delta| **0.155**
- tool: `scripts/gold_oof_ab.py --baseline <preds.csv> --candidate <preds.csv> --changed-labels ...`

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
| lig1 (verdict void) | `girishbose/gf-labels-lig1-5fold` → `outputs/kaggle_download/gf-labels-lig1-5fold/` |
| lig2 (verdict void) | `girishbose/gf-labels-lig2-5fold` → `outputs/kaggle_download/gf-labels-lig2-5fold/` |
| Seed replicates | `girishbose/gf-v0-seed{1337,2024}-5fold` → `outputs/kaggle_download/gf-v0-seed{1337,2024}-5fold/` |
| Seed-kernel template | `outputs/kernels/_template_gf_v0_seed_5fold.py` (patches one `seed:` line, asserts it) |
| Ligament teacher | `src/rsna_knee/text/gf_ligament_teacher.py`, `scripts/build_gf_ligament_labels.py` |
| Ruler tools | `scripts/gold_oof_ab.py`, `scripts/seed_variance_report.py` |
| Configs | `configs/gf_baseline_v0.yaml`, `configs/gf_labels_lig1.yaml`, `configs/gf_labels_lig2.yaml` |

## New-chat opener (copy/paste)
> Read `docs/STATUS.md`, `docs/HANDOFF.md`, `docs/SERIES.md`, and the latest `docs/experiments.md` / `docs/DECISIONS.md`.
> Continue greenfield. **Do not treat 0.6144 or 0.728 as a floor** — seed replicates proved gf_v0 is **0.592 ± 0.021** (seeds 42/1337/2024 → 0.6144/0.5720/0.5889) and 0.6144 was the lucky draw. The v1/v2/lig1/lig2 verdicts are **void** (all single-seed, inside the noise band). Baseline to beat is the **seed-averaged 0.6173**; margin **0.0427**; **every A/B needs ≥3 seeds** (~13 h GPU). We score ~0.59 vs a ~0.937 LB, so pick one large-effect lever, not tuning. One step at a time. Do not poll Kaggle. Do not reopen KneeCoT or put reports in submit.
