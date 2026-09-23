# STATUS

Last updated: 2026-09-23

## Phase
**Public-floor pivot + autonomous search loop** (DECISIONS 2026-09-23). Budget cap $300.
Target: medal-range main final + competitive efficiency final. Top 5 main = stretch.

## Done this session
- `src/rsna_knee/agent/` harness: `journal.py` (MCGS graph, Eq. 7 reward, primary-edge backprop),
  `search.py` (progressive UCT + elite switch, stagnation operators), `memory.py` (RRF retrieval,
  ban list), `dream.py` (window/blend replay on stored logits, Dream-RSI tree replay). 10 tests pass.
- `agent/memory/cold_start.md`: kill ledger + public density results as retrievable records.

## Credentials (outside repo)
- Kaggle: `~/.kaggle/access_token` OK (user `girishbose`). RunPod + DeepSeek: `~/.rsna_agent/secrets.env`.
- RunPod OK, $45 balance. DeepSeek blocked by corporate Zscaler on this laptop -> call it from the RunPod host.
- All three keys were pasted in chat; rotate once the loop is running.

## Public assets (licenses checked 2026-09-23)
| Slug | What | License |
|---|---|---|
| `dreaddevelopment/knee-raptor-corpus` (+`-ext`) | train slice stacks, 44 slots x 336, 15-85% span, 16.4 GB | CC0 |
| `dreaddevelopment/raptor-knee-widedense` | CoAtNet-384 weights, 64-slot 6-94% geometry, 0.924 single | CC0 |
| `dreaddevelopment/raptor-knee-maxspan` | CoAtNet weights, max-span sampling | CC0 |
| `dreaddevelopment/rsna-knee-labels` | report-distilled soft labels | CC0 |
| `pilkwang/rsna-knee-llm-labels` | LLM-read report labels | CC0 |
| `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | 41-checkpoint bundle, public 0.941 | **other** (floor test only) |

Dense 64-80 slot training geometry is not public as slices, only as weights; the geometry lever needs our own dense cache from DICOM (Kaggle CPU).

## Blocked
- **Kaggle GPU weekly quota (30 h) exhausted.** `girishbose/rsna-knee-public-floor` is forked locally at `outputs/kernels/public_floor_rsna_base/` but cannot be pushed until the weekly reset. Every submit needs a short Kaggle GPU commit, so training stays off Kaggle GPU entirely.

## Next 3 actions
1. On quota reset: `kaggle kernels push -p outputs/kernels/public_floor_rsna_base`, then submit its `submission.csv`; log score in `experiments.md`.
2. Now (no Kaggle GPU needed): RunPod pod pulls corpus + both public soft-label sets; start the dense cache builder as a Kaggle **CPU** notebook.
3. Build our gated soft labels; audit vs the two public label sets on the 58 experts before the first train.

## Legacy (log only)
- `girishbose/gf-mri-core-seed42-5fold` v2 (thin cache). Record verdict when user reports; does not block.

## Do not
- Poll Kaggle; reopen thin-recipe levers; treat 0.6144/0.728 as floors
- Reports at test time; KneeCoT
- Hill-climb the public LB; select finals on public LB alone
