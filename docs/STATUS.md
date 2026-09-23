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

## Blocked on user
1. Kaggle API token at `~/.kaggle/access_token` on this machine (CLI installed, no token).
2. RunPod account + API key (`RUNPOD_API_KEY`), ~$40 initial credit.
3. LLM API key for the planner/coder (DeepSeek or Qwen), cap ~$50.
4. License confirmation for Max-Span Dense Corpus + twelve-findings weights (logged-in check).

## Next 3 actions
1. Fork rsna-base, submit once, log public score + runtime in `experiments.md` (floor ~0.936).
2. Pull the dense corpus to RunPod via Kaggle API; build gated multilingual soft labels; audit on 58.
3. First own train on RunPod: public geometry, our soft labels, save per-window logits.

## Legacy (log only)
- `girishbose/gf-mri-core-seed42-5fold` v2 (thin cache). Record verdict when user reports; does not block.

## Do not
- Poll Kaggle; reopen thin-recipe levers; treat 0.6144/0.728 as floors
- Reports at test time; KneeCoT
- Hill-climb the public LB; select finals on public LB alone
