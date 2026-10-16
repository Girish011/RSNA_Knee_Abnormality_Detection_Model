# Handoff — 2026-10-13 14:35 IST (account submit lock until 15 Oct 02:30; six submissions scheduled for that morning; GPU quota zero until 17 Oct)

Source of truth: `docs/STATUS.md` (wins on conflicts). Read the entries from "09:55 IST (12 Oct)" to the end; for the
reasoning behind them, the 11 Oct entries from "10:30 IST (11 Oct)".

## LATEST (16 Oct 00:50 IST) - read STATUS "00:50 IST (16 Oct)" first
- Best 0.954 (fork `pub-blend-mine-v1data`). Specialists alone on CPU: Lateral OA 0.896 (= base), Synovitis 0.908
  (= base), PF OA 0.908 (< 0.926). Public level 0.957 (silver line), gold 0.963, #1 0.967.
- Calendar jobs: `com.rsna.submitqueue` (16 Oct 05:33 / 07:00: `v5-neg-latmen`, `cpu-latmen-tf`, `cpu-pfoa-pfold`;
  17 Oct 05:55 / 06:30 / 08:00: the 0.957 fork once COMPLETE), `com.rsna.push17` (17 Oct 05:32: pushes
  `outputs/kernels/pub/r6frozen-9576` = `girishbose/pub-r6frozen-9576`), `com.rsna.scoretimes16`. Log
  `outputs/submit_queue.log`. Remove them after 17 Oct (`launchctl bootout gui/$(id -u)/com.rsna.<label>`, delete plist).

## READ FIRST (13 Oct 14:35 IST) - overrides the opener below where they differ
- **No submission is possible before ~02:30 IST 15 Oct**: Kaggle locked the whole account's submit route, website
  included (`retry-after` 145200 s read at 10:10 IST 13 Oct; my launchd loop of that morning). Best is still 0.950.
  Do NOT call `submit_gate.py submit` or `submit_queue.py` (without `--dry`) before then.
- **What runs by itself:** LaunchAgent `com.rsna.submitqueue` starts `scripts/submit_queue.py` on 15 Oct at 02:40,
  04:30 and 05:32 IST (one call per kernel, holds on a 429, cap 3 per UTC day; queue `agent/runs/submit_queue.json`:
  the 0.956 fork `pub-blend-mine-v1data`, `cpu-latoa-tf`, `cpu-syn-rec` v2 | `cpu-pfoa-pf`, `cpu-latoa-tfold`,
  `cpu-syn-recsf`); `com.rsna.scoretimes14` / `15` log run times. Log `outputs/submit_queue.log`. After 15 Oct remove
  the three agents (`launchctl bootout gui/$(id -u)/com.rsna.<label>`, delete the plists). Check at the start of every
  turn: `launchctl list | grep rsna; tail -5 outputs/submit_queue.log`.
- **Kaggle GPU quota is 0.00 h until Sat 17 Oct 05:30 IST** (user: Kaggle GPU only for the save run of a notebook
  that will be submitted; everything else on RunPod). Until then only CPU notebooks and already-saved kernels can
  be submitted (`v5-neg-latmen / -cont / -frac`, `v5-t2-w20 / -w40` on the 0.950 base are still COMPLETE).
- **CPU "alone" probes (no GPU quota): `scripts/build_region_probe.py --alone`** - 0.5 in every column but the
  specialist's finding; reading: AUC = 12 x score - 5.5 (+-0.006). Nine saved and checked (STATUS 11:40 / 14:20).
  Base columns to beat (0.950 base): Lateral OA 0.896, Synovitis 0.908, PF OA 0.926. One fold's EMA, one pass:
  ~5 h (tf, pf), ~6.5 h (rec) on CPU; a second round does not fit.
- **When the readings are in (15 Oct ~08:00-11:00 IST):** record them (`submit_gate.py record`), STATUS + experiments
  rows; for a specialist at or above its base column launch folds 2-4 the same morning (`scripts/launch_region.py`,
  ~$0.59/h, 1.4-3.1 h a fold; ~$10 of the user's $30 unspent - show the readings and ask for a concrete amount beyond
  it); choose 16 Oct's three probes (`cpu-latoa-tfg`, `cpu-pfoa-pfg`, `cpu-pfoa-pfold`, `v5-neg-latmen` + `cpu-latmen-tf`).
- **17 Oct 05:30 IST (GPU quota back):** fork `heliosli/0-957-latest-dataset-score-in-one-click` (new compact model
  r7, source pulled to `outputs/kernels/pub/`), a three-column reversal probe on the new base, and candidates = base
  + specialists at 0.5 on their findings (`build_region_probe.py --base ...`; the region leg now works in chunks -
  old `v5-rs-*` / `bm-rs-*` kernels must be REBUILT).
Never `launchctl submit` a script that ends. Never loop on a refused submit.

## New-chat opener (copy/paste)

```
You are continuing a Kaggle campaign in /Users/girish11/RSNA_Knee_Abnormality_Detection_Model (branch agent/autorun).
Read docs/HANDOFF.md, then docs/STATUS.md from "09:55 IST (12 Oct)" to the end. STATUS wins. `export PATH="$PWD/.venv/bin:$PATH"`.

WHERE WE ARE (12 Oct; leaderboard unless said otherwise)
- Best 0.950 (public fork). The top public notebook REALLY scores 0.956 (heliosli blend-mine: expert-trained CoAtNet
  0.5 + compact ConvNeXt-S r6 0.5); 19 others 0.954; velci 0.955 needs its author's private model (fork dropped).
  Board (5,810 teams): #1 0.966, gold line 0.962 (rank 21), silver (rank 290) and bronze (rank 581) lines both
  inside the 0.954 block (630 teams, ranks 185-814, ties by submission time); 0.956 = ranks 88-119.
  `python scripts/public_scores.py` prints real notebook scores + the tie blocks (run it every morning).
- Measured this week: a model trained on the expert-trained public checkpoints' ORDERING of the train studies
  (`labels_t2`) scores 0.939 alone vs 0.929 for its identical control (+0.010). The 0.950 base is weak on three
  findings: Lateral OA 0.896, Synovitis 0.908, PF OA 0.926 (others average 0.963). The 58 expert studies rank models
  in the WRONG order (three cases) - never use them or report rulers to close an idea; only leaderboard probes judge.
- How the public steps were made: 0.950 -> 0.954/0.956 = one new independent model (heliosli "Compact": zoomed 100 mm
  side-aware local views + severity-state targets) rank-blended at 0.3-0.5; 0.954 -> 0.955 (velci) = a knee-centred
  COMPARTMENT-view model at 0.20. A third public generalist (0.946 alone) at 0.25 moved nothing (sanketnandkumarpatil).

MY INSTRUCTIONS (user)
- Goal 0.970; pattern to follow: target the weak findings with specialists, blend, keep every confirmed +0.001.
- No odds / probabilities in answers, never suggest a team merge, lead with what moved the leaderboard. Studying
  the top public notebooks for HOW a step was made is wanted; check them every morning and fork what is new.
- Kaggle activities need no permission. THREE submissions a day, two slots stay free (13 Oct; reset 05:30 IST).
  RunPod: the user gave $30 for the specialists on 13 Oct, "extra funds based on the outputs": show the readings and
  ask for a concrete amount. Another activity (`definition_campaign`, another tool) shares this repo, RunPod and
  Kaggle: leave its pods, files and budget alone.
  Commit only when asked. Handoff at ~80 % context. Keep the Mac on the charger (unattended scripts die with it).

RUNNING BY ITSELF (check at the start of every turn: `pgrep -fl auto_t3; launchctl list | grep rsna`)
- `scripts/auto_t3.sh` under launchd (label com.rsna.autot3; log outputs/auto_t3.log) submits at 05:30 IST on 13 Oct:
  blend-mine fork (expected 0.956 = new base), `bm-neg3` (fork with Lateral OA, Synovitis, PF OA reversed),
  `v5-rs-latoa` (0.950 notebook with Lateral OA replaced by our 3 Oct specialist). Waiting, built and checked:
  `v5-rs-pfoa`, `bm-t2-w15`, `bm-rs-oa-w50`. Unattended scripts died three times on 12 Oct: start them with
  `launchctl submit -l com.rsna.<name> -- /usr/bin/caffeinate -i /bin/zsh <script>`; remove with `launchctl remove`.
  If it died: `python scripts/submit_gate.py submit --kernel <dir> --version 1 --i-approve --why "..."` (texts in the script).
- Nine pods, launched 02:15-02:45 IST 13 Oct, $0.59/h, capped at $27.7 in total: jobs `rs3-<spec>-f<fold>` =
  tf, pf, rec x folds 0, 1 and recsf, tfg, pfg x fold 0 (log agent/runs/region_jobs.json; its `rs2-*` rows are failed
  starts, pod install broken by a pyarrow update, fixed in runpod/run_job.sh). Each uploads
  `girishbose/rsna-knee-rp-rs3-<spec>-f<fold>-out` (tag `rs3<spec><fold>`) and deletes itself after 4-6.5 h; the
  assistant cannot terminate pods. A dataset holding only STATUS + job.log = a failed pod: read job.log.
  `scripts/auto_launch_region.sh` is no longer used.

DO NEXT, IN ORDER
1. Morning (three slots a day: keep one for the best candidate): `python scripts/public_scores.py`; record every score (`submit_gate.py record --ref .. --score ..`), STATUS +
   experiments rows. Readings: fork = new base B; `bm-t2-w15` vs B; `bm-neg3` -> sum of the three weak AUCs on the new
   base = (3 + 12 x (B - score)) / 2 (it was 2.730 on the 0.950 base; +-0.006): if it is ~2.73 the specialists have
   their full room on the new base, if it is ~2.80+ the compact model already repaired them and the bar is higher;
   `v5-rs-latoa` / `v5-rs-pfoa` -> old specialist AUC = 0.896 / 0.926 + 12 x (score - 0.950).
2. Check the pods' result datasets (`kaggle datasets files girishbose/rsna-knee-rp-rs3-<spec>-f<fold>-out`), their
   `raptor_ft_<tag>.json` (oof per finding is a REPORT-label ruler: for sanity only, never to close an arm).
3. Probes of the new specialists for 14 Oct, one finding each: `scripts/build_region_probe.py --base
   outputs/kernels/pub/rsna-versia-5 --dataset <result dataset(s)> --side raptor_rs_side_last.pt,raptor_rs_side.pt
   --dataset girishbose/rsna-knee-rp-rs-f0-v2-out --spec "Finding|raptor_rs_<tag>_last.pt,raptor_rs_<tag>.pt,...|1.0"`
   (swap on the 0.950 base = that specialist's AUC); for a specialist that beats the base column, the CANDIDATE is the
   same `--spec` at weight 0.5 with `--base outputs/kernels/pub/blend-mine-in-private-lb-luck-to-all-upvoters`.
   Verify "[region 3/3] blended" in the kernel log before submitting. Pairs to read: `tf` vs `tfg` (Lateral OA),
   `pf` vs `pfg` (PF OA), `rec` vs `recsf` (Synovitis). Later slots: `v5-t2-w20`, `v5-t2-w40`, `v5-neg-latmen / -cont / -frac`.
4. Stop rule per specialist: three rounds without beating the base column -> dropped. Five folds of whatever beats it.
   Still open for the OA specialists: (a) a DeepSeek pass for three signs per compartment (osteophytes, subchondral
   marrow change, bone-on-bone; "not mentioned" = masked) costs ~$20 of the ~$27 credit (the 6 Oct graded pass cost
   $20.1): ask the user before running it, and only if `tfg` / `pfg` beat `tf` / `pf`; (b) fastMRI+ cartilage boxes
   (3,600) as expert labels for the lateral compartment in coronal views; data stays on the pod.
5. RUN TIME on the 0.956 base: the notebook budgets 8.3 h for itself, Kaggle stops at 9 h. `outputs/score_times.log`
   (scripts/score_times.py, starts 05:35 IST 13 Oct) shows how long the unchanged fork really ran. Region part: ~1.3 h
   per finding (one fold, last + EMA, two passes), guarded (falls back to the base if start + need > 8.6 h - then the
   submission reads as the base and the slot is lost). If the fork runs long: EMA-only checkpoints, one pass.
   `bm-rs-oa-w50` (old specialists at 0.5 on the 0.956 base) is built and checked: submit on 14 Oct only for a
   column whose 13 Oct swap reading is at or above the base column (rebuild with that column alone if only one is).
6. Kaggle GPU quota ~2.9 h until Sat 17 Oct 05:30 IST: inference kernels only (a blend-mine probe run = 3.5-5 min).

TOOLS ADDED 11-12 Oct
- `scripts/public_scores.py`; `scripts/score_times.py`; `scripts/auto_launch_region.sh`; time guard in `build_region_probe.py`; `runpod/region_spec.py --aux_grades` (specs `tfg`, `pfg`);
  `scripts/build_ensemble_kernel.py --base <fork dir> --negate "A;B;C"` (no `--w`) = reversal probe on any fork.
- `scripts/build_teacher_dump.py`, `teacher_dump_read.py`, `make_teacher_labels.py` (teacher scores -> labels_t2);
  `build_v5_two_teachers.py` (224 arm = central 0.8 crop; no gain); `specificity_ruler.py`;
  `build_kaggle_train.py` (smoke run first), `k/<kernel>:<tag>` members in `build_ensemble_kernel.py` / `build_own_probe.py`;
  `runpod/region_spec.py --predict / --chan seqfuse`, `scripts/build_region_probe.py` (+ `--parity`), `launch_region.py`;
  `colab/rsna_knee_queue_colab.ipynb` (draft, not run).
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

