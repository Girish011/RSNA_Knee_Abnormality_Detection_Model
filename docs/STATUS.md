# STATUS

Last updated: 2026-09-24

## Phase
**Public-floor pivot + autonomous search loop** (DECISIONS 2026-09-23). Budget cap $300.
User priority (2026-09-24): **top 5**; do not wait for the Kaggle GPU reset; key rotation not required.

## In flight (do not poll Kaggle; RunPod progress page is fine)
| Job | Where | What | Pull when done |
|---|---|---|---|
| `labels-ab-v1` | RunPod pod `v4gyvfi50q6vwh`, progress `https://v4gyvfi50q6vwh-8000.proxy.runpod.net/STATUS` (+`/job.log`, `/<arm>/train.log`) | CoAtNet-384 16 ep on public 44-slot corpus; arms `dread_s42` vs `blendv2_s42` | `kaggle datasets download girishbose/rsna-knee-rp-labels-ab-v1-out`; pod self-terminates after upload |
| dense80 cache | Kaggle CPU `girishbose/rsna-knee-dense80-s0..s3` | 80-slot 2–98% train cache, ~10 GB per shard | `kaggle kernels output girishbose/rsna-knee-dense80-s{k}` (onto RunPod, not the Mac) |

## Done this session (2026-09-24)
- Label audit (`scripts/audit_public_labels.py`): public pilkwang LLM labels **0.870** macro AUC on 57 experts vs keyword v1 0.628 / v7 0.651 (paired +0.24 [0.19, 0.29]). dread omits the experts; Spearman 0.81 with pilk. Keyword teachers retired.
- Calibration: pilk UNK cells are expert-positive 13.6% vs its flat 0.28 → `labels_llm_blend_v2` uses dread where pilk is UNK.
- Found the public trainer (`dreaddevelopment/knee-mri-training-the-twelve-finding-model`), vendored as `runpod/train_knee.py`; CoAtNet-384 16 ep ≈ 3 h on a 4090.
- RunPod runner: `src/rsna_knee/agent/runpod.py`, `runpod/run_job.sh`, `scripts/runpod_launch.py` (bundle → private Kaggle dataset → pod → results dataset → self-terminate; capacity fallbacks 4090 → 3090 → secure 4090).
- Dense80 cache builder (public widedense test-time picker, 64→80 slots) pushed as 4 Kaggle CPU kernels.

## Credentials (outside repo)
- Kaggle: `~/.kaggle/access_token` holds only the Kaggle token (the old 3-line file is `~/.kaggle/access_token.bak_multiline`). RunPod + DeepSeek: `~/.rsna_agent/secrets.env`.
- RunPod balance $45 at launch. DeepSeek untested on this machine.
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
1. When `labels-ab-v1` uploads: download, apply the pre-registered read-out (experiments 2026-09-24), pick the teacher.
2. When dense80 shards finish: check `meta_s{k}.json` (errors, empty studies), then RunPod job `dense80-v1` = CoAtNet-384 on dense80 with the chosen teacher, k_eval over all windows; save per-window logits for window-count dreams.
3. Build the dense80 submit notebook (same picker, our weights, timed) and, on quota reset, submit the floor fork + our first model.

## Legacy (log only)
- `girishbose/gf-mri-core-seed42-5fold` v2 (thin cache). Record verdict when user reports; does not block.

## Do not
- Poll Kaggle; reopen thin-recipe levers; treat 0.6144/0.728 as floors
- Reports at test time; KneeCoT
- Hill-climb the public LB; select finals on public LB alone
