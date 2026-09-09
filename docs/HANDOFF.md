# Handoff — greenfield campaign

Paste into a new chat. Source of truth: this file + `docs/STATUS.md` + `docs/SERIES.md`
+ latest greenfield entries in `docs/experiments.md`.

## Read first
1. `docs/STATUS.md`
2. `docs/SERIES.md`
3. `configs/gf_baseline_v0.yaml`
4. Tail of `docs/experiments.md` (entries from 2026-09-08 onward)

## Current task
**B.4.1** — cache dry-run (CSV only, no DICOMs):
- Create/run `notebooks/03_cache_dry_run.py`
- Output: `outputs/eda/gf_v0_cache_plan.csv`

## Artifacts that matter now
- `outputs/eda/gf_v0_series_picks.csv`
- `outputs/eda/step_a_summary.txt`
- `site/assets/eda_planes.png`, `site/assets/eda_gold_rates.png`
- `notebooks/01_data_audits.py`, `notebooks/02_series_picker.py`
- `configs/gf_baseline_v0.yaml`

## How we work
- One step at a time when learning to code
- Repo docs beat chat history (anti-context-rot)
- Commit/push before switching machines

## Do not
- Resume old v6c / cache_v1 / submit-v6c as the active path
- Open long historical STATUS/HANDOFF threads unless auditing archive
- KneeCoT; reports at inference; commit secrets

## New-chat opener
> Read `docs/STATUS.md`, `docs/HANDOFF.md`, and `docs/SERIES.md`.
> Continue greenfield Step B from STATUS Next 3. One step at a time unless I say Agent mode.