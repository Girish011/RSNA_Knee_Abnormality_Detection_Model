# Handoff — plan to the top (2026-10-02)

Source of truth: `docs/STATUS.md` (wins on conflicts), `docs/PLAN_TO_TOP.md` (strategy, results log, next build),
`docs/DECISIONS.md` (2026-10-01 entries), `agent/runs/submits.json`. Repo beats chat.

## Where we are
- **Best LB 0.945** (56754655 `coat-up15-b4`: public D4 stack + soupbestof4 in v5 slot + CoAt-branch weight +0.15).
  Pending: 56760305 `coat15-radfused` (+ our fused-label RadImageNet heads); `coat30-radfused` probe auto-submits ~05:30 IST.
- Strategy (PLAN_TO_TOP.md): cross-architecture distillation — retrain/fine-tune the weak public families (RadImageNet,
  DINO, A5) on our fused labels (expert agreement 0.946 vs report 0.883); rebalance toward CoAt; grand soup; finals Oct 16.
- Proven so far: fused-label RadImageNet heads beat upstream (E13 +0.017, main +0.005 gold OOF); CoAt weight +0.15 → +0.001 LB.
- Next build: DINO warm-start fine-tune kernel (exact design in PLAN_TO_TOP.md).
- Money: RunPod $4.10 (user tops up); Kaggle GPU 30 h/week, 2 concurrent sessions (forks take ~1.5 h).

## Harness (how it works now)
- launchd `com.rsna.agentloop` runs `rsna_knee.agent.loop`; idle/busy ticks call the DeepSeek strategist
  (V4.1-Flash, thinking ≤ $3/day, relabels ≤ $15/day). Fold-0 planner disabled.
- Member jobs: `python -m rsna_knee.agent.members queue|eval|status --node 5bc0a0f45d75 --labels … --init v5 --epochs 6 [--k 16]`.
  Up to **2 pods** at once (user-approved). Phantom pods retried ×6. Pods seen alive are never relaunched as phantoms.
- **Edit member state only via ops**: `members.request_op({"op": "drop_running"|"drop_queued"|"requeue", ...})`
  (`agent/runs/members_ops.jsonl`, applied at next tick) — direct edits get overwritten by a running tick.
- Soups: `scripts/make_soup.py --name X --members a,b,... --upload`; eval: `members eval --targets soupX`.
- Kernels: `scripts/build_ensemble_kernel.py --name … --w 0 --swap-v5 <member|soup>`; verify the loaded checkpoint
  in the run's `btkd_v559_complete.json` before submitting. LB pushes/submits only with user approval.
- **After editing agent code, restart the loop** (`launchctl kickstart -k gui/$(id -u)/com.rsna.agentloop`) — the
  running process keeps old code (2026-10-01: stale code launched the weak5 job without `--label_weights`).
- Per-finding tails: `build_ensemble_kernel.py --swap-v5 soupbestof4 --tail "PF OA|soupk16|0.4"`; check
  `dataset_sources` in kernel-metadata.json before pushing (soups must be registered in members.json first).
- I (assistant) cannot terminate RunPod pods (permission system) — ask the user.
- Tests: `.venv/bin/python -m pytest -q tests/ --ignore=tests/test_rank_architecture.py` (104 pass).

## New-chat opener (copy/paste)

```
Read docs/HANDOFF.md, docs/STATUS.md, docs/PLAN_TO_TOP.md and the 2026-10-01 entries of docs/DECISIONS.md. Repo files beat chat; STATUS wins on conflicts. Branch agent/autorun.

Goal: climb the Kaggle LB toward the top (best 0.945 now; aim 0.950+, then the 0.955-0.961 band); lock finals by 2026-10-16. You (Claude) own architecture decisions. Strategy = cross-architecture distillation: fine-tune the weak public stack families (RadImageNet done, DINO next, then A5) on our fused labels, rebalance toward CoAt, grand soup.

First: record the scores of 56760305 (coat15-radfused) and the coat30-radfused probe (submit_gate.py ledger / record), keep the better as base. Then build and run the DINO warm-start fine-tune kernel exactly as designed in PLAN_TO_TOP.md "Next build"; gate on gold OOF > 0.8425 before wiring.

Rules: submit only candidates expected to beat the current best (learning probes OK); gate ≤5/day via scripts/submit_gate.py. Strategist is propose-only for GPU. RunPod balance ~$4 — ask before spending beyond it; you cannot terminate pods. Edit member state only via members.request_op. After editing agent code restart the loop (launchctl kickstart -k gui/$(id -u)/com.rsna.agentloop). Give me a handoff prompt at ~80% context.
```
