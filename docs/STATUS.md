# STATUS

Last updated: 2026-09-16

## Phase
**Greenfield Imaging Playbook**. Measurement crisis resolved; convergence A/B **paused**.
Coding: **gf_v0c stage 1 ABORTED** — GPU quota exhausted; user requested stop.

## Active experiment
- None. Do not re-push GPU jobs until quota resets.
- Kernel left in place (do not delete): `girishbose/gf-v0c-conv-seed42-5fold`
- If still RUNNING in the UI: open that kernel → **Stop Session** / cancel the version.
  The Kaggle API token cannot cancel sessions (`kernelSessions.cancel` → 403).

## Headline finding (2026-09-16) — the floor was a fluke
Three runs of the **identical** gf_v0 recipe, differing only in `train.seed`:

| seed | gold OOF | weak OOF |
|---|---|---|
| 42 (the old "floor") | **0.6144** | 0.6855 |
| 1337 | **0.5720** | 0.6716 |
| 2024 | **0.5889** | 0.6870 |

**gf_v0 = 0.592 ± 0.021** (mean ± sd), range 0.042. The 0.6144 we ranked everything
against was its luckiest draw. Seed-averaged OOF = **0.6173**, better than every single seed.

## Consequences
- **v1 / v2 / lig1 / lig2 verdicts are void** — all decided inside the noise band.
  lig1 (0.6019) and lig2 (0.6103) both sit **above** the v0 seed mean 0.5918.
- The "Effusion collapse" that killed lig1/lig2 was regression to the mean:
  Effusion's own seed sd is **0.133** (0.889 → 0.631 → 0.707).
- Weak ruler sd **0.0085** (n≈2449) vs gold sd **0.0214** (n=58) → small-n gold is the
  amplifier, and no training discipline removes it.
- **Single-seed A/Bs are banned** (DECISIONS 2026-09-16). An A/B now costs ~13 h GPU
  (3 seeds × 4.4 h) → ~2 honest A/Bs per week at a 30 h quota.

## Current ruler
| Item | Value |
|---|---|
| Baseline (seed-averaged OOF gold, seeds 42/1337/2024) | **0.6173** |
| Measured seed sd | 0.0214 |
| Margin = max(0.005, 2×sd) | 0.0427 |
| Single-seed keep threshold (impractical) | 0.6600 |
| Per-label sd: worst | Effusion 0.133, Lat OA 0.090, Baker's 0.086, MCL 0.080 |
| Per-label sd: best | PF OA 0.012, Lat Men 0.012, ACL 0.022, Synovitis 0.028 |

## Next 3 actions
1. **Stop the session in the Kaggle UI** if it is still RUNNING (API cannot cancel).
2. Wait for GPU quota to reset; do not launch new GPU kernels until then.
3. Resume `gf_v0c` stage 1 (same kernel / meta already staged) when quota is back.

## Budget note
An honest 3-seed A/B costs ~13 h GPU against a ~30 h weekly quota, so ~2 per week.
Staging (1 seed, then 2 more only if promising) is deliberate. Quota exhausted mid-stage-1.

## Do not
- Poll Kaggle; cite v1/v2/lig1/lig2 verdicts as evidence; treat 0.6144 or 0.728 as a floor
- Decide any recipe on a single seed; reports at test time; KneeCoT
- Read sub-0.04 single-seed macro deltas as signal
