# Handoff — campaign to submit/LB AUC ≥ 0.958 (2026-09-29, ensemble pivot)

Paste the **New-chat opener** at the bottom into a fresh chat. Source of truth: this file,
`docs/STATUS.md`, `docs/DECISIONS.md` (esp. 2026-09-23/25/27/28), `docs/AGENT_PLAN.md`,
`agent/runs/loop_state.json`, `agent/runs/journal.json`, `agent/runs/holdout.json`.
**STATUS beats this file if they disagree.** Repo files beat chat history.

## Where we are
- Branch: **`agent/autorun`**. Budget cap **$300**; spent ~**$34**.
- **True goal:** Kaggle **public / final macro AUC ≥ 0.958** (stretch talk of 0.970 is not the gate). Fold-0 proxy is compass only.
- Loop + dashboard under launchd (`com.rsna.agentloop`, `com.rsna.dashboard`). DeepSeek owns idle ticks (coder allowlist + `launch_runpod_train`). One-pod / $300 / >$10 / **no auto-LB**.
- Settled recipe: **CoAtNet-384 + dense80 + better labels**, **k=8**, **k_eval=24**.

## 2026-09-29 pivot (read first)
- Our single model = LB 0.920; public D4 stack = 0.943; top-20 = 0.955–0.960. Submits are now **D4 stack + our members** (`scripts/build_ensemble_kernel.py --name … --members … --w …`).
- Member track: `python -m rsna_knee.agent.members queue --node <id> --seeds 42,1337 --labels a,b [--init v5 --epochs 6]`; `members status`. Full-data, multi-arm, one pod, auto-launched by the loop tick; loop does not plan while a member job is queued/active.
- LB submits only via `scripts/submit_gate.py submit --kernel … --version N --why … --i-approve` (after the user approves). `submit_gate.py ledger` / `record --ref --score`.
- v5 warm start (`--init_model`, CC0 public checkpoint) is member-only (fold-0 proxy would be leaky).

## Workflow (how the system works)
1. RunPod trains one genome at a time → uploads OOF to a private Kaggle result dataset.
2. Mac loop `observe` downloads → scores **online proxy** (fold-0 vs agreement ruler) + gold_dev veto + hidden holdout.
3. Idle `plan`: free **dream** blends over saved OOF, then DeepSeek proposes one-field diffs / allowlisted code / relabel.
4. LB submit is **manual / user-approved**: build kernel under `outputs/kernels/`, push, then `kaggle competitions submit -k … -v N -f submission.csv`. **Do not background-poll** kernels.

## Drawbacks / failure modes (do not ignore)
- **Proxy≠LB:** first own-weights submit `5bc0` proxy **0.9366** → public **0.920** (gap **−0.017**). Do not treat proxy ≳ 0.936 as 0.958-ready.
- **VRAM:** k>8 / k_eval>24 CUDA-OOM on 24GB after dense80 — hard ban + `genome.clamp`.
- **dense80 pull:** ~40 GB; IncompleteRead common; use `fetch_dense80.py` Range-resume. Boot stall kill at 5 h. Hung downloads have burned hours.
- **Phantoms / capacity:** community pods vanish; launcher verifies; loop infra-retries — still noisy at night.
- **Host crashes block DeepSeek:** e.g. stale `greedy_rank_blend` import crashed every tick before planner ran — needs launchd restart; DeepSeek never sees host exceptions.
- **Duplicate LB burns:** two submits both scored **0.920**; do not resubmit near-clones. Quota 5/day — use only for diagnosed lifts (SWA/EMA/eval/blend/new parent).
- **lr/epoch/wd micro-diffs** often fail to beat best online; prefer unused labels + submit-gap closers.

## Scores (STATUS wins)
| Kind | Best | Notes |
|---|---:|---|
| **Public LB** | **0.920** | `submit-dense80-5bc0` v2 (refs 56643440, 56646862) |
| Online proxy | **0.9366** | `5bc0a0f45d75` epochs→20, dense80 + `relabel_8c56cdf0` |
| Dream | **0.9391** | greedy blend — free CPU, **not** LB |
| Recent miss | 0.9342 | `564c` head_lr→0.002 |

## Live (check STATUS)
- `n-ca830ada7798` on `s0yb5c4835xdkr` ($0.34/h) — bb_lr→1e-5, k=8. One-pod; do not double-launch.

## Settled — do not reopen without new failure
- MENU `k=[8]`, `k_eval=[24]`, `arch=[coatnet384]`; dream-first; no new backbones / MONAI / Jev-RLCD this week.
- Efficiency student later from same weights.
- dense80 via Range-resume; no campaign restart from scratch.
- Approvals: >$10, LB submits, finals, budget changes.

## Next
1. Score live pod; if bb_lr fails, prefer unused labels over more lr micro-sweeps.
2. Next LB only with a **gap story** (SWA/EMA / denser k_eval / blend) expected to beat **0.920**.
3. Submit ledger + 5/day gate + duplicate filter: **shipped** (`scripts/submit_gate.py`).
4. End session: update `docs/STATUS.md`.

## New-chat opener (copy/paste)

```
Read docs/STATUS.md, docs/DECISIONS.md (esp. 2026-09-28 speed + LB calibration, plus 2026-09-23/25/27), and docs/HANDOFF.md. Repo files beat chat history; STATUS wins on conflicts.

Branch agent/autorun. Goal: Kaggle public/final macro AUC ≥ 0.958 ($300 cap; spent ~$34). Fold-0 proxy is compass only — not the trophy. Stretch 0.970 is not the gate. Approvals: >$10, LB, finals. Dual finals later; efficiency student later from same weights.

Workflow: one RunPod train → OOF upload → Mac loop scores online proxy + dreams → DeepSeek proposes one-field genome / allowlisted code / relabel. LB is user-approved kernel push + competitions submit (no auto-LB; do not background-poll Kaggle).

Live: check docs/STATUS.md (members queue: `python -m rsna_knee.agent.members status`). One-pod; do not double-launch. launchd com.rsna.agentloop.

Pivot 2026-09-29: submits = public D4 stack (0.943) + our full-data members rank-blended (scripts/build_ensemble_kernel.py); LB only via scripts/submit_gate.py with user approval; v5 warm start is member-only.

Scores: public LB 0.920 (submit-dense80-5bc0; proxy was 0.9366 → gap −0.017). Best online 0.9366 (5bc0). Best dream 0.9391 (free blend, not LB). 564c head_lr=0.002 → 0.9342 miss.

Known drawbacks: proxy≠LB; VRAM ban k>8; dense80 download hangs/phantoms; host tick crashes can idle DeepSeek; duplicate submits both hit 0.920 — do not burn 5/day on clones.

Speed policy (settled): MENU k=8 / k_eval=24 / coatnet384; dream-first; GPU only labels/lr/epoch/wd; no new backbones. dense80 via fetch_dense80.py Range-resume.

Next: (1) score ca830; (2) next LB only with diagnosed gap-closer expected to beat 0.920; (3) prefer unused labels over micro-lr thrash; (4) update docs/STATUS.md end of session. Never print secrets.
```
