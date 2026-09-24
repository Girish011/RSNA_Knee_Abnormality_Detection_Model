# STATUS

Last updated: 2026-09-24

## Phase
**Public-floor pivot + autonomous search loop** (DECISIONS 2026-09-23). Budget cap $300.
User priority (2026-09-24): **top 5**; do not wait for the Kaggle GPU reset; key rotation not required.

## Autonomous loop (live since 2026-09-24 10:32 IST)
- Process: launchd agent `~/Library/LaunchAgents/com.rsna.agentloop.plist` (runs `caffeinate -i python -m rsna_knee.agent.loop run --interval 600`, starts at login, restarts on crash; log `agent/runs/loop.log`). Dashboard: `com.rsna.dashboard`. Restart after code changes: `launchctl kickstart -k gui/$(id -u)/com.rsna.agentloop` (safe any time the loop is sleeping; state is on disk). Remove: `launchctl bootout gui/$(id -u)/com.rsna.agentloop`.
- Control: `python -m rsna_knee.agent.loop status | pause | resume | approve | reject`; `touch agent/runs/STOP` exits it. Dashboard shows loop heartbeat and APPROVAL NEEDED.
- Needs you only for: approvals (any action > $10), leaderboard submits, finals, budget changes. Keep/kill of a new best uses `agent/runs/holdout.json` (never shown to the planner).
- Design and checks: DECISIONS 2026-09-24 "Autonomous loop goes live". First loop node = baseline (public CoAtNet recipe, dread labels) under the fold-0 / expert-holdout protocol.

## In flight (do not poll Kaggle; RunPod progress page is fine)
| Job | Where | What | Pull when done |
|---|---|---|---|
| loop node `7421b1b6a6fa` (baseline) | pod `lyk09i0lfpu0wr`, `https://lyk09i0lfpu0wr-8000.proxy.runpod.net/STATUS` | public recipe, dread, fold-0 + holdout protocol | loop does it |
| ~~`labels-ab-v1`~~ DONE: wash (0.9148 vs 0.9148), dread stays | RunPod pod `mw4c7rb992jk7u` (community RTX 3090, $0.22/hr; ep0 = 17.6 min → ~5 h/arm, ETA ~2026-09-24 06:30 UTC), progress `https://mw4c7rb992jk7u-8000.proxy.runpod.net/STATUS` (+`/job.log`, `/<arm>/train.log`) | CoAtNet-384 16 ep on public 44-slot corpus; arms `dread_s42` vs `blendv2_s42` | `kaggle datasets download girishbose/rsna-knee-rp-labels-ab-v1-out`; pod self-terminates after upload |
| dense80 cache | Kaggle CPU `girishbose/rsna-knee-dense80-s0..s3` | 80-slot 2–98% train cache, ~10 GB per shard | `kaggle kernels output girishbose/rsna-knee-dense80-s{k}` (onto RunPod, not the Mac) |

## Done this session (2026-09-24)
- Label audit (`scripts/audit_public_labels.py`): public pilkwang LLM labels **0.870** macro AUC on 57 experts vs keyword v1 0.628 / v7 0.651 (paired +0.24 [0.19, 0.29]). dread omits the experts; Spearman 0.81 with pilk. Keyword teachers retired.
- Calibration: pilk UNK cells are expert-positive 13.6% vs its flat 0.28 → `labels_llm_blend_v2` uses dread where pilk is UNK.
- Found the public trainer (`dreaddevelopment/knee-mri-training-the-twelve-finding-model`), vendored as `runpod/train_knee.py`; CoAtNet-384 16 ep ≈ 3 h on a 4090.
- RunPod runner: `src/rsna_knee/agent/runpod.py`, `runpod/run_job.sh`, `scripts/runpod_launch.py` (bundle → private Kaggle dataset → pod → results dataset → self-terminate; capacity fallbacks 4090 → 3090 → secure 4090).
- Dense80 cache builder (public widedense test-time picker, 64→80 slots) pushed as 4 Kaggle CPU kernels.

## Credentials (outside repo)
- Kaggle: `~/.kaggle/access_token` holds only the Kaggle token (the old 3-line file is `~/.kaggle/access_token.bak_multiline`). RunPod + DeepSeek: `~/.rsna_agent/secrets.env`.
- RunPod balance $44.9 before the A/B pod. DeepSeek untested on this machine.
- RunPod host lessons: 3 community pods failed (torch 'CUDA unknown error' on 2, silent self-terminate on 1) and 1 secure pod hung in pip (torch upgrade via timm). Fixes in `run_job.sh`: pinned torch/torchvision/numpy, GPU guard with retries that terminates bad hosts; launcher filters host CUDA ≥ 12.4 and falls back secure 4090 → community 4090 → community 4090/3090.
- User decided rotation is not required (keys were echoed into a session log on 2026-09-24).

## Public assets (licenses checked 2026-09-23)
| Slug | What | License |
|---|---|---|
| `dreaddevelopment/knee-raptor-corpus` (+`-ext`) | train slice stacks, 44 slots x 336, 15-85% span | CC0 |
| `dreaddevelopment/raptor-knee-widedense` | CoAtNet-384 weights, 64-slot geometry, 0.924 single, 0.9054 on 58 experts | CC0 |
| `dreaddevelopment/raptor-knee-maxspan` | CoAtNet weights, max-span sampling | CC0 |
| `dreaddevelopment/rsna-knee-labels` | report-distilled soft labels (no experts) | CC0 |
| `pilkwang/rsna-knee-llm-labels` | LLM-read report labels | CC0 |
| `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | 41-checkpoint bundle, public 0.941 | **other** (floor test only) |

## Blocked
- **Kaggle GPU weekly quota** until the weekly reset: no submits until then. Fork ready at `outputs/kernels/public_floor_rsna_base/`.

## Next 3 actions
1. Loop runs the baseline node, then DeepSeek-planned nodes/relabels. Check the dashboard; answer approvals.
2. When dense80 shards finish (user reports): check `meta_s{k}.json`, add a dense80 corpus path to `run_job.sh`/`genome.py` and flip `CORPORA["dense80"] = True` so the loop can use it.
3. Build the dense80 submit notebook; on GPU quota reset, submit the floor fork + best loop node (user approves).

## Legacy (log only)
- `girishbose/gf-mri-core-seed42-5fold` v2 (thin cache). Record verdict when user reports; does not block.

## Do not
- Poll Kaggle; reopen thin-recipe levers; treat 0.6144/0.728 as floors
- Reports at test time; KneeCoT
- Hill-climb the public LB; select finals on public LB alone
