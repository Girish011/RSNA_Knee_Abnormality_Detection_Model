# Handoff — clean pseudo-label rounds (2026-10-03 evening)

Source of truth: `docs/STATUS.md` (wins), `docs/PLAN_TO_TOP.md` (last section "2026-10-03 evening — PIVOT").

## Where we are
- Best LB 0.945 (coat-up15-b4; coat15-radexfx also 0.945). Own CoAt alone 0.920. Board #1 0.961.
- **Lever = labels.** Forum: single 224 px CoAtNet/ResNet reach 0.950–0.954 via OOF pseudo-labels. Ours on holdout-29:
  report labels 0.883, clean OOF ensemble 0.929, blended round target 0.945.
- **Leak found:** `f-7cdb6e8357` (old CoAt baseline) and every `init:v5` run saw all train studies. Clean baseline =
  `n-5bc0a0f45d75`. `scripts/family_a_rulers.py` defaults to it; `scripts/make_round_labels.py` refuses leaky teachers.
- Native-res specialists (`runpod/region_spec.py`, dense512 cache `rsna-knee-dense512-s0..7`, geometric localiser + side
  model) work but only equal clean CoAt → kept as a diverse family for later. Option 2: bestof < fused for Family A.
- **Running:** round-0 teacher folds `r0-coat-f1..4` (dense80, ~6.5 h) and `r0-fa-f1..4` (~3.7 h); pod ids in
  agent/runs/activity.jsonl; results `girishbose/rsna-knee-rp-<job>-out` (fetch files with `-f` or from the pod's
  http server `<monitor>/<tag>/raptor_*.npz` before it self-deletes). `rs-f0-v2` (pf-fused, tf-rep) ends ~17:15 IST.
- **Budget: $250 RunPod approved; spend ≈ $13 + round 0 ≈ $23 → ≈ $36. Check in at $50.** User terminates pods.

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
Read docs/HANDOFF.md, docs/STATUS.md and docs/PLAN_TO_TOP.md (last section "2026-10-03 evening — PIVOT: clean pseudo-label rounds"). Repo files beat chat; STATUS wins on conflicts. Branch agent/autorun.

Goal: #1 on the Kaggle LB (board 0.961; us 0.945; own CoAt alone 0.920). The lever is labels: clean OOF pseudo-label rounds (teachers must be ImageNet-init, never init:v5 — those leak). Claude owns architecture and decisions.

First: collect round-0 teacher folds (jobs r0-coat-f1..4, r0-fa-f1..4 → outputs/runpod_results/<job>/<tag>/), run scripts/make_round_labels.py --out labels_r1.parquet (families in its docstring), check the holdout-29 gate, then launch round-1 students (CoAt 5bc0 recipe + family_a.py, 5 folds, ImageNet init, on labels_r1) on parallel pods. After round 1: own-alone LB probe of the 5 CoAt students through the dense80 leg (baseline 0.920), then round 2, then final-level blend with the stack.

Rules: budget up to $250 RunPod (running spend line in STATUS; check in at each $50); user terminates pods. LB submits for candidates expected to improve and own-alone probes, gate <=5/day via scripts/submit_gate.py. Offline baseline = n-5bc0a0f45d75 (clean). Edit member state only via members.request_op; restart the loop after agent-code edits. Give a handoff prompt at ~80% context.
```
