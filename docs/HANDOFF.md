# Handoff — 2026-10-09 02:22 IST (public forks queued for 05:30 IST; user: rebuild the 0.949 public formula on RSNA + fastMRI, no OAI)

Source of truth: `docs/STATUS.md` (wins on conflicts). Read the entries of 8-9 Oct (from "17:5x IST (8 Oct)" to the end).

## New-chat opener (copy/paste)

```
You are continuing a Kaggle campaign in /Users/girish11/RSNA_Knee_Abnormality_Detection_Model (branch agent/autorun).
Read docs/HANDOFF.md in full, then docs/STATUS.md from the first "(8 Oct)" entry to the end. STATUS wins. Run the
FIRST 10 MINUTES list, then work through NEXT ACTIONS without waiting for me unless it says it is my decision.

WHERE WE ARE (9 Oct, all leaderboard unless said otherwise)
- Best 0.945 (ref 56754655, the older public stack + our swaps), rank ~800; bronze line 0.950, #100 0.953, #20 0.959,
  #1 0.964. Deadline 22 Oct. Our own models alone: 0.934-0.937 (best: four ConvNeXt-S, first-round labels, 0.937).
- Closed with leaderboard readings: label rounds, graded labels, D1 / D2, more own models / backbones, stack + own
  blends (all findings and per finding: 0.942-0.944).
- New public notebooks score ~0.95. Their main model (weights `nartaa/rsna-knee-publication-swa-weights-20261007`,
  0.949 alone) was trained with 2,399 OAI knees as external supervision; the host has not ruled on OAI-trained
  weights (forum topics 741819, 747455). I was told; my decision at final selection: keep one non-OAI submission.

MY INSTRUCTIONS
- Kaggle activities need no permission (pushes, submissions, probes); 5 submissions a day (reset 05:30 IST); say in
  each description what it is and its odds. Every number: say whether it is a leaderboard score.
- NO MONEY: RunPod $3.64 left and no top-ups; Claude subagent reads are stopped (reader test parked, harness in
  scripts/llm_reader*.py); DeepSeek $27 (only `deepseek-flash` sees images). Kaggle GPU: 30 h a week, resets Sat
  10 Oct 05:30 IST (`kaggle quota`), 2 sessions, 12 h each.
- "Use the same formula as the 0.950 notebook and do not change it; no OAI - use the RSNA data and fastMRI."
- Forking public notebooks is fine. Kaggle CLI for the forum. Commit only if I ask. Keep the Mac on the charger -
  three unattended scripts have died with the battery.

RUNNING BY ITSELF (check outputs/auto_fork.log)
- `scripts/auto_fork.sh`: four unchanged public forks (`outputs/kernels/pub/*`: versia-5, velciraptor, apex
  grandmaster stack, dinosaur-v5) -> submits at the 05:30 IST slot reset. `scripts/auto_recipe_probe.sh`:
  `cnxs-r1-allwin-mirror` = our four ConvNeXt-S scored their way (39 windows x 4 passes + mirror) vs 0.937.
  If the Mac slept: submit by hand with `python scripts/submit_gate.py submit --kernel <dir> --version 1 --i-approve
  --why "..."`.

- (9 Oct 10:0x) Forks scored 0.950 x 4 (new best, refs 56979517 / 537 / 558 / 590); our models scored their way 0.938.
  REBUILD STARTED: `scripts/auto_cx.sh` (log outputs/auto_cx.log) finishes the cache kernels `cx-cache-s0..5`
  (public ConvNeXt-reader preprocessing) and pushes `cx-train-a` (their train.py, folds 3 and 4). Tools:
  scripts/build_cx_cache.py, build_cx_train.py (`--runs fold:seed,fold:seed --args "..."`), labels
  data/processed/cx/labels_v0.csv. Collect: `kaggle kernels output girishbose/cx-train-a -p <dir>` ->
  runs/cx_v0/fold{3,4}.pt + oof csv. Next steps are listed in the last STATUS entry of 9 Oct.

- (10 Oct 09:45) Scored: public reader alone 0.935; 0.950 notebook + our models at w 0.10 / 0.20 = 0.950 / 0.950.
  Reader rebuild stopped (weak alone). Bronze line 0.952. Newest public notebooks cannot be forked (weights retired).
  NEXT if I say go: fastMRI+ expert labels as masked external supervision - steps in the last STATUS entry of
  10 Oct. Check `kaggle kernels list --competition rsna-knee-abnormality-detection --sort-by scoreDescending`
  every morning and fork any new runnable top notebook before 05:30 IST. Final selection is mine: one 0.950
  fork + 56754655 (0.945, no OAI weights).

FIRST 10 MINUTES
1. `cd` to the repo; `export PATH="$PWD/.venv/bin:$PATH"`; `date`; `cat outputs/auto_fork.log`.
2. `kaggle competitions submissions rsna-knee-abnormality-detection --csv | head -8`; record each score
   (`python scripts/submit_gate.py record --ref <ref> --score <lb>`) + rows in the STATUS tables. `kaggle quota`.

NEXT ACTIONS, IN ORDER
A. Tell me the fork scores and the probe score plainly. Best fork = our new base.
B. THE FORMULA, rebuilt without OAI (their name: raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa; README and
   manifest in outputs/kernels/pub/_recipe/). Their trainer is not public; ours (`runpod/train_knee.py`) is the same
   lineage. Map, one item at a time, nothing else changed:
   - alldata -> `--fold -1`; t16 -> `--k 16`; CoAtNet `--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384`
     (their "efficiency" twin: the 224 px CoAtNet, 0.945 with OAI - `coatnet_rmlp_2_rw_224`, ~7 min an epoch on an A40);
   - blendjev -> blend of the public label sets `stevenleehans/rsna-knee-llm-report-labels`,
     `pilkwang/rsna-knee-llm-labels`, JEV labels (riadmohamed42) - build it, do not use our labels_r1 / r4;
   - d96 r384 -> a 96-slot corpus at native 384 px (140 mm crop); the slot quotas are in the forked notebook
     (`_dense_quotas`, `INFERRED96_WIDTHS`); build with Kaggle CPU kernels like dense80 / dense512;
   - oai -> fastMRI as the masked external set (`--ext_dir`, `--ext_weight`; fastMRI has no labels, so it enters
     with teacher pseudo-labels; the pod cache is gone, rebuild with runpod/fastmri_dicom_cache.py);
   - swa -> SWA over the best epochs; scoring: every window (their K94), anatomical mirror (`--canon` + leg v2).
   Compute: the 224 px run fits the $3.64 (~$1-1.5) or one Kaggle session; the 384 px run is ~$6-7 of A40 or two
   Kaggle sessions (check the trainer can resume first). First reading: alone on the leaderboard against their
   0.945 / 0.949 and our 0.937. Stated odds (9 Oct): ~1 in 2 to beat 0.937, ~1 in 6 to reach 0.949 without OAI.
C. Public ConvNeXt reader (`goodpjw2008/rsna-knee-2-5d-convnext-reader`, code in outputs/kernels/pub/_recipe/): train
   the folds they did not ship (3, 4) and seeds with their train.py unchanged; add to the fork's blend.
D. Blend our models into the best fork (give `scripts/build_ensemble_kernel.py` a `--base` option; its STASH / leg /
   BLEND cells work on any notebook that writes submission.csv). One submission per idea, odds stated.

GOTCHAS
- `kaggle kernels pull -m` keeps the author's `id_no`: remove it before pushing a fork (else 403).
- zsh: `${k}:tag` not `$k:tag`; a command kept in a variable is not word-split. Kernel logs are empty while running.
- The 58 experts and the report ruler `said` both mislead; only leaderboard probes judge (a four-model family alone
  scores in 20-45 min).
- Every STATUS timestamp comes from `date`.
```

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
- Tests: `.venv/bin/python -m pytest -q tests/ --ignore=tests/test_rank_architecture.py` (123 pass).

