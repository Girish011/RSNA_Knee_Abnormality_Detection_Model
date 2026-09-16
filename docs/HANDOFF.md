# Handoff — greenfield campaign (2026-09-15)

Paste into a **new chat**. Source of truth: this file + `docs/STATUS.md` + `docs/SERIES.md`
+ latest entries in `docs/experiments.md` + `docs/DECISIONS.md`.

## Read first
1. `docs/STATUS.md`
2. `docs/SERIES.md`
3. `docs/DECISIONS.md` (tail from 2026-09-13)
4. Tail of `docs/experiments.md` (from 2026-09-08 / greenfield)

## Where we are (one paragraph)
Greenfield campaign on RSNA Knee. Blog Posts 01–03 done. Image recipe is **gf_v0**: 3 series × 12 slices × 224, frozen DINOv2-S, weak_v1, cache `girishbose/rsna-knee-cache-gf-v0`. **The defining result (2026-09-16): the 0.6144 "floor" was a fluke.** Running the identical recipe at seeds 42 / 1337 / 2024 gave 0.6144 / 0.5720 / 0.5889 — gf_v0 is really **0.592 ± 0.021**, and 0.6144 was its luckiest draw. Consequently every greenfield verdict so far (v1 24-slice, v2 336px, lig1, lig2) was decided inside the noise band and is **void**; lig1 (0.6019) and lig2 (0.6103) actually sit *above* the v0 seed mean. Single-seed A/Bs are now banned: recipes are judged as **3-seed seed-averaged OOF gold**, current baseline **0.6173**. Nothing is running.

## Scoreboard (read the caveats)
| Run | Weak OOF | Gold OOF | Status |
|---|---|---|---|
| gf_v0 seed 42 | 0.685 | 0.6144 | lucky draw, not a floor |
| gf_v0 seed 1337 | 0.672 | 0.5720 | same recipe |
| gf_v0 seed 2024 | 0.687 | 0.5889 | same recipe |
| **gf_v0 3-seed mean** | 0.681 ± 0.009 | **0.592 ± 0.021** | honest description |
| **gf_v0 seed-averaged OOF** | — | **0.6173** | **current baseline to beat** |
| gf_v1 24-slice | — | (old ruler) | verdict VOID (single seed) |
| gf_v2 336px | — | (old ruler) | verdict VOID (single seed) |
| gf_labels_lig1 | 0.707 | 0.6019 | verdict VOID; above v0 seed mean |
| gf_labels_lig2 | 0.677 | 0.6103 | verdict VOID; above v0 seed mean |

## Active job
**None.** `girishbose/gf-v0c-conv-seed42-5fold` was **ABORTED** (GPU quota exhausted).
Meta `girishbose/rsna-knee-gf-v0c-meta` is already published — keep it.

If the kernel is still RUNNING in the UI: open
https://www.kaggle.com/code/girishbose/gf-v0c-conv-seed42-5fold → **Stop Session**.
(API cannot cancel: `kernelSessions.cancel` → 403.)

When quota resets: re-push / re-run the same kernel (no meta rebuild unless code changed).
Then: download → read `convergence_report.json` → only if paired gain, run seeds 1337+2024.

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

## Levers still on the table (after gf_v0c)
We score ~0.59 against a ~0.937 public LB and the ruler resolves ~0.04, so small tuning is off
the table. Remaining large-effect candidates, in rough order:
1. **Backbone fine-tuning** — blocked by DECISIONS 2026-08-12, but every past unfreeze
   "collapse" was a single-seed read and is now suspect.
2. **Volume/pooling redesign** — the v1/v2 kills are void, so 24-slice and 336px are untested
   rather than disproven.
3. **Med Men fills, done cleanly** — single-factor version (fills restricted to the 2449
   studies already in v0 training so no new studies enter), at 3 seeds.
4. **Structural re-plan** if the above stall: the frozen-DINOv2-S 3×12×224 design may simply
   be far from competitive.

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
