# Handoff — weak-finding push (2026-10-01)

Paste the **New-chat opener** at the bottom into a fresh chat. Source of truth: `docs/STATUS.md`
(wins on conflicts), `docs/DECISIONS.md` (esp. 2026-09-29 → 2026-10-01), `docs/experiments.md`,
`docs/HIGHER_PATH.md`, `agent/runs/members.json`, `agent/runs/submits.json`. Repo beats chat.

## Where we are
- **Public LB 0.944, rank 258/4660** (ref 56704765): public D4 stack (`outputs/kernels/_base_d4`, 0.943) with our
  **4-seed bestof soup** (`soupbestof4`) in the v5 maxspan arms. Top of board 0.961; 86 teams ≥ 0.950.
- Lock finals **2026-10-16** (official deadline 2026-10-22 23:59 UTC; Oct 15 = entry/merger only, we are entered).
- Budget: campaign ≈ $80 of $300; **RunPod ≈ $9.7 (top up ~$20 before more GPU)**; DeepSeek ≈ $30.

## What is measured (AUC on 58 expert studies = mean of gold-dev 29 + holdout 29)
| Model | Macro | Lat Men | Lat OA | PF OA | Synovitis | Contusion |
|---|---:|---:|---:|---:|---:|---:|
| public v5 (eval-only, our protocol) | 0.9256 | 0.855 | 0.896 | 0.857 | 0.801 | 0.927 |
| bestof_v1 members, k=8 (n=4) | 0.9330 | 0.877 | 0.891 | 0.855 | 0.863 | 0.943 |
| bestof_v1 members, **k=16** (n=2) | 0.9266 | 0.857 | 0.881 | **0.882** | 0.857 | 0.915 |
| soup bestof4 (= LB 0.944 swap) | **0.9365** | 0.862 | 0.890 | 0.857 | 0.865 | 0.957 |
Strong findings (ACL, MCL, Med Men, Med OA, Effusion, Baker's) are 0.96–0.99; Fracture ~0.93.

## Closed (do not reopen without new evidence)
- lr/wd/head_lr/epoch micro-diffs; fold-0 ImageNet trains (`loop.FOLD0_TRAINS=False`).
- **Self-training / label fusion**: fused labels agree with experts (0.946–0.949 vs report 0.883) but members trained
  on them = 0.9330 (same as bestof). Hard-blocked by `agent/runs/policy.json` {"fold_jobs_closed": true}.
- More seeds of a tested recipe (2- and 4-seed soups equal): `members.SEEDS_PER_RECIPE = 2`.
- Tail-leg blending a weak model across all findings (cost −0.002 on LB).
- Two different members in the two maxspan arms (stack loads ONE model for forward+reverse) → use a soup.

## Next plan (user-agreed direction: focus the 5 weak findings, then soup)
1. **Per-finding loss weights**: add trainer flag `--label_weights` (multiply BCE per finding; 2–3× for Lateral
   Meniscus, Lateral OA, PF OA, Synovitis, Contusion); member option `--label_weights`; 2 seeds bestof_v1 k=8 v5 6 ep.
2. PF OA specialist already exists: k=16 members `m5b5942352cf`, `m440e7701125` (PF OA 0.882).
3. **Per-finding tail** in `scripts/build_ensemble_kernel.py`: BLEND weight per finding (0 on strong findings,
   ~0.3–0.5 on the weak five) with our specialist leg; keep soup4 in the v5 slot. Submit via `scripts/submit_gate.py`.
4. Soup the per-finding winners; dual finals by Oct 16. Expert bench (public stack on 58 experts) optional/deferred.
- (ROI/compartment crops: trainer flag exists but `roi_localize` + `roi_boxes.npz` were never vendored; skip.)

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
- I (assistant) cannot terminate RunPod pods (permission system) — ask the user.
- Tests: `.venv/bin/python -m pytest -q tests/ --ignore=tests/test_rank_architecture.py` (104 pass).

## New-chat opener (copy/paste)

```
Read docs/HANDOFF.md, docs/STATUS.md, and the 2026-09-29 → 2026-10-01 entries of docs/DECISIONS.md. Repo files beat chat; STATUS wins on conflicts. Branch agent/autorun.

Goal: Kaggle public/final macro AUC as high as possible (aim 0.950+, stretch higher); lock finals by 2026-10-16 (official deadline 2026-10-22). Current LB 0.944 (rank 258) = public D4 stack with our 4-seed bestof soup in the v5 slot.

Plan I agreed: focus ONLY the 5 weak findings (Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion), then soup. Step 1: add per-finding loss weights to runpod/train_knee.py (--label_weights) + members option, train 2 seeds bestof_v1 (k=8, v5, 6 ep). Step 2: reuse k=16 members for PF OA. Step 3: per-finding tail weights in scripts/build_ensemble_kernel.py (0 on strong findings). Step 4: soup winners, dual finals.

Closed: self-training (policy.json), lr micro-diffs, fold-0 trains, >2 seeds per recipe. RunPod balance ~$9.7 — ask me before GPU spend beyond that. LB submits need my OK. You cannot terminate pods; ask me. Edit member state only through members.request_op. Watch context: give me a handoff prompt at ~80%.
```
