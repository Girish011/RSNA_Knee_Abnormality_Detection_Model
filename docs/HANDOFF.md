# Handoff — public-floor pivot + autonomous loop (2026-09-23)

Paste the opener at the bottom into a **new chat**. Source of truth: this file, `docs/STATUS.md`,
`docs/AGENT_PLAN.md` (approved plan), `docs/DECISIONS.md` (tail from 2026-09-21).

## Where we are
- Strategy pivot approved (DECISIONS 2026-09-23): fork the public floor, train on RunPod RTX 4090,
  $300 cap, pre-registered gate **2026-10-06: public LB ≥ 0.950** or switch to robustness + efficiency.
- Top 5 main board is a stretch; ~10 teams show 0.96 public. Realistic: medal range + efficiency final.
- Built: `src/rsna_knee/agent/` (journal, search, memory + ban list, dream replay, activity log),
  `agent/memory/cold_start.md`, `scripts/agent_dashboard.py`. 11 tests in `tests/test_agent.py` pass.
- Branch: **`agent/autorun`**. Audit trail: `agent/runs/activity.jsonl`.

## Blocked / pending
1. Kaggle weekly GPU quota exhausted. Fork ready at `outputs/kernels/public_floor_rsna_base/`
   (`girishbose/rsna-knee-public-floor`, private). On reset: `kaggle kernels push -p <dir>`, then submit.
2. RunPod: key OK, $45 credit, nothing running. Next: pod pulls `dreaddevelopment/knee-raptor-corpus`
   (+`-ext`) and both public soft-label sets.
3. Dense cache builder (64-80 slots, 6-94% span) as a Kaggle **CPU** notebook — not started.
4. DeepSeek was blocked by the old laptop's corporate proxy; verify from the new machine.

## Public assets
See the license table in `docs/STATUS.md`. All CC0 except `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`
(license "other", public 0.941 bundle; floor reference only, never in a final).

## New-machine setup
```bash
git clone https://github.com/Girish011/RSNA_Knee_Abnormality_Detection_Model.git
cd RSNA_Knee_Abnormality_Detection_Model
git checkout agent/autorun            # or: git fetch <bundle> agent/autorun:agent/autorun
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,kaggle]" scipy
pytest -q tests/test_agent.py
python scripts/agent_dashboard.py --serve           # http://localhost:8765/dashboard.html
```
Secrets (never in the repo), with **freshly rotated** keys:
- `~/.kaggle/access_token` — Kaggle API token
- `~/.rsna_agent/secrets.env` — `RUNPOD_API_KEY=...` and `DEEPSEEK_API_KEY=...`

## New-chat opener (copy/paste)
> Read `docs/HANDOFF.md`, `docs/STATUS.md`, `docs/AGENT_PLAN.md`, and the tail of `docs/DECISIONS.md`.
> We are on branch `agent/autorun`. Continue the approved plan from "Next 3 actions" in STATUS.
> Secrets are in `~/.kaggle/access_token` and `~/.rsna_agent/secrets.env`; never print or commit them.
> Log every action with `rsna_knee.agent.activity.log` and keep the dashboard running.
> Do not poll Kaggle; do not reopen thin-recipe levers; reports are train-only.
