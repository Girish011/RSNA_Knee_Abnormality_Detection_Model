# Handoff — campaign paused (2026-10-04 20:50 IST)

Source of truth: `docs/STATUS.md` (wins on conflicts; its last three sections are the closing state),
`docs/PLAN_TO_TOP.md` (results log up to 2026-10-03), `docs/DECISIONS.md` (dated decisions since August).

## New-chat opener (copy/paste)

```
Read docs/STATUS.md (last three sections) and docs/HANDOFF.md. Repo files beat chat; STATUS wins. Branch agent/autorun; the work is in commit 023d186 plus the HANDOFF commit after it (not pushed).

STATE (2026-10-04 20:50 IST): the campaign is PAUSED on the user's decision. Do not launch pods, spend, or submit to the leaderboard without a new explicit go-ahead. No pods are running.

Goal was #1 on the Kaggle RSNA Knee Abnormality Detection LB (macro AUC over 12 findings; hidden test graded by expert radiologists; deadline 2026-10-22). Board #1 0.963, #10 0.958. Our best 0.945 (ref 56754655, kernel coat-up15-b4). The best public notebook alone scores 0.946.

Why it was paused: asked directly, the assistant said it was not confident of reaching #1. The 0.945 is essentially the public stack (forked at 0.943, our swaps added ~0.002). Our own models never passed what is public, and no identified step is worth more than a few thousandths against a 0.018 gap.

The core problem: training labels come from radiology reports, the test is graded by experts reading images, and only 58 training studies have expert labels. Report labels agree with experts at ~0.88 AUC; the best training target we built (reports + our own out-of-fold models) reaches 0.94-0.945; students cannot beat their target. Compute was never the limit.

WHAT WORKED
- Forking the public D4 stack (0.943), swapping in our soup of four full-data CoAt members (0.944), raising the CoAt weight by 0.15 (0.945).
- Own dense80 input cache matching the submit geometry (about +0.006 over the public 44-slice density).
- Finding the leak: every run warm-started from public v5 had seen all train studies, so its "out-of-fold" scores were inflated by 0.03-0.04. Clean baseline = n-5bc0a0f45d75 (ImageNet init, LB 0.920).
- Clean pseudo-label round 1 (labels_r1): single model 0.920 -> 0.928 on LB; five folds 0.929 -> 0.932.
- CoAtNet @224 with 16 slices: 0.933 on LB with five folds, same as CoAt-384 at under half the cost.
- Tooling: scripts/submit_gate.py (ledger, 5/day gate), launch_round.py, collect_round.py, oof_ruler.py (870-study ruler, noise 0.001-0.002), label_rounds_page.py, build_ensemble_kernel.py --negate (per-finding LB probes).
- Per-finding LB probes: the stack scores ~0.914 on Synovitis, ~0.950 on Lateral Meniscus, ~0.948 on the other ten. No single weak finding; the gap is general.

WHAT DID NOT WORK (all measured; do not repeat without new evidence)
- Aug to mid-Sep, own pipeline on frozen DINOv2 with keyword / NLI report labels: more slices, 336 px, unfreezing the backbone (-0.114), training longer and three rounds of label gap-fills all failed. Superseded 2026-09-23 by forking the public pipeline.
- Own model in the stack at the final level: three blends, all 0.001-0.002 below the stack (0.941 vs 0.943; 0.943 and 0.944 vs 0.945).
- Stack tweaks: per-finding tails, label transplants, more soup slots, CoAt weight +0.30, fused RadImageNet heads (+0.017 on the 58 experts, 0.944 on LB), fine-tuned DINO (0.944).
- New architectures on the same labels: RegionMIL (Family A), fastMRI+ encoder init (Family B), native-resolution region specialists at 512 px, ResNet50, ConvNeXt-T. All equal to or below the clean CoAt.
- Recipe changes on labels_r1 (fold 0, 870-study ruler): 32 slices, affine augmentation + drop-path, backbone lr 1e-4. All null.
- Label round 2 (labels_r2g, report weight 0.3 + expert-calibrated silent cells): worse in all ten fold readings (five-fold holdout-29 0.924 vs 0.941).
- Other label rules: every rule tried lands at 0.94-0.945 against the experts.
- Blending our own families: +0.0024 on the 4,349-study ruler. EMA instead of best-epoch checkpoint: +0.0012.
- Severity-graded report labels: checked by reading the 58 expert reports; wording does not separate confirmed from rejected cells except at the severe end.
- The DeepSeek strategist: 40 turns, $3.65, nothing usable; paused 2026-10-02.

MEASUREMENT RULES LEARNED
- Only LB probes judge. The 29 / 58 expert rulers have noise of 0.01-0.02, and differences below ~0.01 or per-finding expert numbers do not transfer.
- The 58 experts are contaminated for anything selected or fused on them.
- oof_ruler.py scores agreement with reports, so it judges label-neutral recipe changes only, not label changes.
- Teachers must be ImageNet-init, never init:v5. labels_r1 is the training label set.

UNFINISHED (terminated by the user at 20:45 IST, not null): r2-c224e30-f0 (30 epochs, stopped at 20), r2-c224pw1-f0 (plain BCE via --pos_weight_cap 1, stopped at 5), r2-c224reg40-f0 (augmentation, 40 epochs, stopped at 9). About $5 to rerun.

IF REOPENED, the only ideas with any prior: (1) more expert-grade labels (needs a rules check by the user); (2) a clean out-of-fold rebuild of the stack as pseudo-label teacher (estimate +0.002-0.005, unmeasured). Neither is expected to reach #1.

Money: RunPod ~$84 of the $250 budget; balance $21.36. The user terminates pods and tops up. 5 Kaggle submits per day, reset 05:30 IST.
Open on the user's side: select 56754655 as a final submission on Kaggle; the launchd loop com.rsna.agentloop is still running (spends nothing, strategist paused).
Monitoring page: scripts/label_rounds_page.py -> outputs/label_rounds.html, republish to https://claude.ai/artifact/4yrR2cL71M3cXrhHYgSVkU.
Give a handoff prompt at ~80% context.
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

