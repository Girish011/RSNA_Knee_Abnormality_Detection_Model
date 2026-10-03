# Handoff — native-res region specialists (2026-10-03)

Source of truth: `docs/STATUS.md` (wins), `docs/PLAN_TO_TOP.md` (esp. "2026-10-03 — NEW PLAN"), `docs/EXTERNAL_DATA.md`.

## Where we are
- Best LB 0.945 (coat-up15-b4; 56779149 coat15-radexfx also 0.945, recorded). Stack tweaks exhausted.
- Family A (dense80 RegionMIL) and Family B (fastMRI+ init) lost to CoAt on report-label OOF → DICOMs skipped.
- **Running:** option 2 `fa-f0-rep` (Family A on `labels_bestof_v1`, 16 ep, fold 0) on pod `prno0rfoyp7hh9`,
  result `girishbose/rsna-knee-rp-fa-f0-rep-out` (if whole-dataset download 404s, fetch files with `-f`).
  Judge: `scripts/family_a_rulers.py --fa 'outputs/runpod_results/fa-f0-rep/fa-f0-rep'` — mainly pilk + holdout-29
  (dread ruler is partly circular: bestof/fused are dread-based). Compare fa-f0-v2 (fused): pilk weak5 0.842.
- **Budget: up to $250 RunPod approved (2026-10-03); spend ≈ $6–8; running line in STATUS; check in at each $50.**
- Next build (Claude owns architecture): dense512 Kaggle cache → stage-1 localiser → region specialists (plan).
- Kaggle CLI works (`.venv/bin/kaggle`); launcher needs `PATH=.venv/bin:$PATH`. fastMRI links: env only, never print.

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
Read docs/HANDOFF.md, docs/STATUS.md, docs/PLAN_TO_TOP.md (esp. "2026-10-02 night — PIVOT: own-model track"), docs/EXTERNAL_DATA.md and the 2026-10-02 entries of docs/DECISIONS.md. Repo files beat chat; STATUS wins on conflicts. Branch agent/autorun.

Goal: #1 on the Kaggle LB (board: #1 0.961, #10 0.958; us 0.945, best public notebook 0.946). We go all in on OUR OWN models for the weak five findings (Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion): Family A = region-aware 2.5D specialists on our dense80 cache (RunPod), blended at the FINAL level with real weight (0.3-0.45); Family B = fastMRI+ pretraining once the user's NYU links are in ~/.rsna_ext/fastmri_links.txt (never print/commit them). You (Claude) own architecture. Public-stack inner transplants are closed (invisible at their effective weight).

First: record the score of 56779149 (coat15-radexfx probe) via scripts/submit_gate.py record. Then build Family A: data module over dense80 (runpod/fetch_dense80.py), model, 1-fold smoke on RunPod, rulers (holdout-29 pooled AUC + 5-fold OOF vs report labels + rank-corr vs stack).

Rules: LB submits only for candidates expected to beat the best (learning probes OK), gate <=5/day via scripts/submit_gate.py. RunPod: ~$20 at Claude's discretion for plan training; ask before more; user terminates pods. Strategist paused (policy.json). Edit member state only via members.request_op; restart the loop after agent-code edits. Give a handoff prompt at ~80% context.
```
