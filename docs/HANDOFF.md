# Handoff — own-model track (2026-10-03)

Source of truth: `docs/STATUS.md` (wins on conflicts), `docs/PLAN_TO_TOP.md` (results log; "Family A — build log"),
`docs/EXTERNAL_DATA.md`, `agent/runs/submits.json`. Repo beats chat.

## Where we are (2026-10-03 ~00:45 UTC)
- Best LB 0.945 (56754655 coat-up15-b4). **56779149 (coat15-radexfx) score NOT recorded** — Kaggle submissions CLI
  was blocked in-session; user to run `submit_gate.py record --ref 56779149 --score <lb>`.
- **Family A** (`runpod/family_a.py`, RegionMIL: ConvNeXt-S tiles, coronal halves, routed finding queries) fold 0
  `fa-f0-v2` **FAILED rulers** (`scripts/family_a_rulers.py`): weak5 vs pilk 0.842 vs CoAt 0.884, dread 0.875 vs
  0.923, holdout-29 0.892 vs 0.912; blend lowers CoAt. Fused-label AUC is NOT a ruler (circular).
- **Family B** (`runpod/fastmri_pretrain.py`): fastMRI singlecoil → on-pod volumes (1,172; flipud = fastMRI+ frame;
  320→336) → HeatNet box-heatmap pretraining of the same encoder → `family_a.py --init_encoder` fold 0 = clean A/B
  vs fa-f0-v2. Running on pod `q1a2we500s1ydu` (job `fb-f0-v2`, result `girishbose/rsna-knee-rp-fb-f0-v2-out`).
  Pretrain ep1 held-out tile AUC: meniscus 0.957, cart_full 0.993, bone_edema 0.966, acl 0.956.
  Prep cost ~1.9 h (xz decompression is serial).
- Links: `~/.rsna_ext/fastmri_links.txt` lines `knee_singlecoil_train <url>` — passed to pods only via env
  `FASTMRI_LINKS` (launcher `extra_env`); never print/commit; prep takes only `knee_singlecoil_*` links.
- Decision rule (agreed with user): fastMRI init gains ≥~0.02 weak5 vs report labels → build fastMRI knee DICOM
  pipeline (batch1 134 GB + batch2 30 GB; unlabeled → pretraining / teacher pseudo-labels, judged only on report
  labels + holdout-29). Else skip DICOMs; fix the Family A recipe / stack levers.
- Inference budget: external data adds 0 submission time; an extra 5-fold Family A/B leg ≈ +1–1.5 h (unmeasured:
  current stack runtime — read it from run 56754655 before adding a leg).
- Money: RunPod ≈ $7 of $20 spent after fb-f0-v2. Pods: `launcher.launch(..., corpus="dense80",
  trainer="family_a.py", gpu_class="48", extra_env=...)`; arms may start `@script.py` to switch trainer per arm.
  Run the launcher with `PATH=.venv/bin:$PATH` (needs the kaggle CLI). Community hosts vary 8–50 MB/s.

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
