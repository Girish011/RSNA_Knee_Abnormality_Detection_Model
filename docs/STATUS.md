# STATUS

Last updated: 2026-10-04 (~20:50 IST). **Campaign paused by the user; see the last section.** Plan: docs/PLAN_TO_TOP.md (results log + next build).

## Leaderboard
| ref | kernel | public |
|---|---|---:|
| 56651119 | D4 + 5bc0 tail w=0.2 | 0.941 |
| 56665556 | swap m6edbb53fbab → v5 | 0.943 |
| 56680743 | swap m0b16e9a09f1 → v5 | 0.943 |
| **56704765** | **swap soupbestof4 → v5** | **0.944 (rank 258/4660)** |
| 56744344 | + PF OA ← soupk16 w0.4 | 0.944 |
| 56744346 | + PF OA, Synovitis ← soupk16 w0.35 | 0.944 |
| 56754652 | LEARNING raptor-all-b4: soup4 in v5+v8+v10 | 0.944 |
| **56754655** | **coat-up15-b4: soup4 swap + CoAt weight +0.15** | **0.945 (best)** |
| 56760305 | coat15-radfused: + fused-label RadImageNet heads | 0.944 |
| 56761295 | LEARNING coat30-radfused: CoAt +0.30 + fused heads | 0.944 |
| 56771185 | coat15-dinox30: base + fine-tuned DINO β0.3 after calibrator | 0.944 |
| 56779149 | LEARNING coat15-radexfx: best + Fracture in Rad blend (d4-blend tweak) | 0.945 |
| 56814815 | coat-up15-c224-w20: best stack + own c224x5 leg, rank w0.2 | 0.943 |
| 56814822 | coat-up15-c224-w30: best stack + own c224x5 leg, rank w0.3 | 0.944 |

## Best models (AUC on 58 experts)
- soupbestof4 0.9365 (in LB 0.944); bestof_v1 k=8 members 0.9330; public v5 0.9256.
- k=16 bestof members 0.9266 overall but **PF OA 0.882** (vs 0.855) → `soupk16` (2-seed soup, uploaded
  `girishbose/rsna-knee-soup-k16`, not yet expert-evaluated) = PF OA specialist leg.
- Weak findings everywhere: Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion.

## Live
- DINO fine-tune finished (3.3 h free T4x2, 20/20, pixel corr 0.998). Builder `scripts/build_dino_ft_kernel.py`, gate `scripts/eval_dino_ft.py`.
- Fused RadImageNet heads: gold OOF +0.017 (E13) but LB 0.944 vs 0.945 → **gold-58 gains do not transfer**.
  Gold-58 is contaminated for models trained on our labels (bestof selection + fusion fit on experts): our 5-fold
  CoAt alone scores 0.931 on gold-58 but our own-weights LB was 0.920. LB probes are the judge.
- No RunPod pods. **Strategist paused** (policy `strategist: paused`, user 2026-10-02); loop still monitors pods/members.
## Money
- Campaign ≈ $85 of $300. **RunPod ~$20 (topped up 2026-10-02)**; Claude may spend it on new training tries that need GPU capacity; default $0 until a lever is LB-supported. DeepSeek ≈ $30.

## Submission policy
Only submit candidates expected to beat 0.944 (or explicit learning probes). Gate ≤5/day.

## Family A (own model, 2026-10-03)
- `runpod/family_a.py` + rulers built. Fold-0 `fa-f0-v2` FAILED rulers (weak5 vs pilk 0.842 vs CoAt 0.884; blend lowers). Fused-label AUC is not a ruler.
- Family B (fastMRI+ encoder init) A/B fb-f0-v2: no gain on report labels (weak5 pilk 0.833 vs 0.842); holdout-29 +0.03 (noisy). DICOMs skipped per rule. RunPod ≈ $6 spent; budget raised to $250 (user, 2026-10-03; check in each $50).
- 56779149 score not yet recorded (Kaggle CLI call was blocked in-session; user to run `submit_gate.py record`).

## Region specialists (2026-10-03, native-res plan)
- dense512 cache built as 8 Kaggle CPU kernels (`rsna-knee-dense512-s0..7`); geometric localiser (no training) +
  `runpod/region_spec.py` (regions tf / pf / rec / bm / side). Details: PLAN_TO_TOP "Region specialists — build log".
- 56779149 coat15-radexfx = 0.945 (ties best). 
- dense512 complete: 4,407 studies, 0 errors, 0 empty (8 shards, 6–12 min each).
- **Results 14:30 (PLAN_TO_TOP build log):** (1) the CoAt baseline `f-7cdb6e8357` is v5-init = leaky; clean baseline is
  `n-5bc0a0f45d75` (LB 0.920). (2) Option 2: bestof report labels lose to fused → keep fused. (3) `tf-fused`
  native-res specialist = clean CoAt on pilk (0.869 vs 0.877 on its 4 findings): no single-model gain.
  (4) Cross-family rank blend CoAt+TF+FA: +0.006 pilk / +0.011 dread; same-family blend +0.
- **Running:** `rs-f0-v2` pod dugehqdl1phzyr ($0.49/h): pf-fused → tf-rep; ends ~17:15 IST, self-deletes.
  `rs-f0-v1` (h4l6xr6wl6trwu) died at smoke (cv2 missing; fixed in run_job.sh) — self-deleted, user to confirm.
  `fa-f0-rep` (prno0rfoyp7hh9) finished, self-deleting.
- **RunPod spend line (budget $250, check-in each $50): ≈ $8 + fa-f0-rep $1.9 + rs-f0-v1 $0.2 + rs-f0-v2 ~$2.7 ≈ $13.**

## Label rounds monitor
- Page: https://claude.ai/artifact/4yrR2cL71M3cXrhHYgSVkU (snapshot; regenerate with `scripts/label_rounds_page.py` →
  `outputs/label_rounds.html`, then republish that file). Job registry: `agent/runs/label_rounds.json`.
- 2026-10-03 16:15: `r0-coat-f2` / `r0-coat-f4` ($0.34 community 4090s) vanished (API 404, no results) → relaunched as
  `r0-coat-f2b` (7a4c97ja2hutxq) / `r0-coat-f4b` (mnv3cfz8bprivf) on 48 GB hosts. Prefer `gpu_class="48"` for teachers.

## Label rounds — round 0 done, round 1 launched (2026-10-03 21:30 IST)
- Round-0 teachers: all 10 folds collected (CoAt gold AUC per fold 0.904–0.914). `r0-fa-f2` crashed at ep 11
  (mmap length > file size) and finished on run_job.sh's built-in retry.
- `data/processed/labels_r1.parquet` written (4,349 studies, coverage 1.000). **Holdout-29 gate PASSED:** report 0.883 ·
  clean OOF ensemble 0.931 · round target **0.945**. Per finding target vs report: ACL 0.97/1.00, MCL 0.99/0.99,
  MedMen 0.95/0.90, LatMen 0.87/0.82, MedOA 0.98/0.94, LatOA 0.99/0.93, PF OA 0.98/0.88, Effusion 0.93/0.77,
  Synovitis 0.83/0.71, Baker's 0.98/0.97, Contusion 0.94/0.83, Fracture 0.93/0.86.
- Round-1 students launched with `scripts/launch_round.py --round 1 --labels labels_r1.parquet` (5 CoAt + 5 FA folds,
  ImageNet init, 48 GB hosts; jobs `r1-coat-f0..4`, `r1-fa-f0..4` in `agent/runs/label_rounds.json`). The launcher skips
  folds already in the registry; relaunch a vanished fold with `--folds N --suffix b`.
- The older session ("Native-resolution region specialists") agreed not to launch; this session owns launches + page.
- **RunPod spend line (budget $250, check-in each $50): ≈ $14 + round 0 ≈ $20 = $34; round 1 ≈ $27 committed → ≈ $61.**

## Round 1 done; probes pending; 224 px families launched (2026-10-04 03:00 IST)
- Round-1 students all collected. Holdout-29 (5 folds combined): CoAt 0.931 -> 0.934, Family A 0.918 -> 0.930, both
  0.931 -> 0.936; per-fold CoAt best-epoch gold 0.920-0.927 vs teachers 0.904-0.914. `labels_r2.parquet` written (target
  0.949) but a plain round 2 on the same two families is NOT launched: label quality +0.004 => expected student gain ~0.001.
- **LB probes submitted 01:45-02:20 IST (pending):** 56806013 `probe-r1-coat-f0` (fold-0 student alone, vs 0.920),
  56806016 `probe-r1-coat5` (5 students mean), 56806452 `probe-r0-coat5` (5 teachers mean = ensembling control).
  Builder `scripts/build_probe_kernel.py`. Record with `submit_gate.py record --ref .. --score ..`. 2 submits left 10-04 UTC
  day resets 05:30 IST.
- Recipe comparison: our CoAt = public single CoAt (0.920 fold-0 vs 0.924 full data) -> no pipeline gap. Forum 0.95
  singles are students distilled from strong, multi-source teacher ensembles, small models at 224-288 px. => lever =
  stronger / more diverse teacher ensemble, not more rounds of the same two families.
- **Running (launched 02:45 IST):** `r1-r50-f0..4` (ResNet50 @224, k16) and `r1-c224-f0..4` (CoAtNet @224, k16) on
  labels_r1, ImageNet init; ~2-3 h, ~$10. In the registry (families r50, c224).
- Language check: `scripts/label_quality_by_language.py` (9 languages; silence pattern differs by language, reader
  agreement 0.93-0.98; no language-specific change justified yet). Rerun each round.
- **RunPod spend line: ~$34 after round 0 + round 1 ~$21 = ~$55; 224 px pods ~$10 committed -> ~$65 of $250.**

## Probes scored; recipe > labels (2026-10-04 06:40 IST)
| ref | probe (own model alone) | public |
|---|---|---:|
| 56643440 | single CoAt-384 fold 0, report labels (old baseline) | 0.920 |
| 56806013 | `probe-r1-coat-f0`: fold-0 CoAt-384 student, labels_r1 | 0.928 |
| 56806452 | `probe-r0-coat5`: 5 CoAt-384 teachers mean, report labels | 0.929 |
| 56806016 | `probe-r1-coat5`: 5 CoAt-384 students mean, labels_r1 | 0.932 |
| 56810671 | `probe-r1-c224x5`: 5 CoAtNet@224 k16 students mean, labels_r1 | 0.933 |
- Labels: +0.008 single model, only +0.003 at 5-fold level -> a plain round 2 is not worth it.
- **Recipe: CoAtNet @224 with k=16 slices beats CoAt-384 k=8** on holdout-29: 5-fold 0.941 vs 0.934 (per fold
  0.923-0.930 vs 0.920-0.927), ~7 min/epoch. ResNet50 @224 (untuned) 0.895 — weak, not a useful teacher as is.
  Family combos did not beat c224 alone on holdout-29 (0.933-0.939).
- **Running (launched 06:38 IST, fold 0, labels_r1):** `r1-c224k32-f0` (kd9u2w1tuvrkfp; k=32 @224, ~14 min/epoch).
  `r1-c288-f0`/`f0b` FAILED by design: the 224 px CoAtNet weights cannot run at 288 (fixed relative position bias); both
  pods uploaded their log and self-deleted (~$0.35). Recipe removed. Compare best gold with r1c2240 = 0.9265. Recipes live in `scripts/launch_round.py`.
- **RunPod spend line: ~$55 after round 1 + 224 px families ~$10 + 2 variants ~$4 -> ~$69 of $250.**

## 09:15 IST — c224 probe 0.933: own five-fold legs plateau at 0.929-0.933
- Holdout-29 said c224 +0.006 over CoAt-384; LB says +0.001 (0.933 vs 0.932). **Holdout-29 differences below ~0.01 do
  not transfer; only LB probes judge.** k=32 fold-0 variant tracks k=16 epoch for epoch (0.917 at ep 8-9) -> no gain
  from more slices; let it finish, do not take it to 5 folds.
- Own leg is ~0.012 below the stack (0.945). Neither labels (+0.003 at 5 folds) nor recipe (+0.001) closes that.
  c224 is as good as CoAt-384 at less than half the cost -> use it for all further own-model work.

## 10:40 IST — blend probes submitted; round 2 (labels_r2g) launched
- **Final-level blend probes (scoring):** 56814815 `coat-up15-c224-w20`, 56814822 `coat-up15-c224-w30` = best stack
  (coat-up15-b4, 0.945) + own c224x5 leg at rank weight 0.2 / 0.3. Both runs checked before submit: leg loaded the five
  `r1c224*` checkpoints at 224 px, blend applied, soupbestof4 in the receipt, no v5 file. 3 of 5 submits used for the
  Kaggle day ending 2026-10-05 05:30 IST.
- **labels_r2g.parquet** = c224 + CoAt-384 + RegionMIL round-1 students, `--a-said 0.3 --a-unk 0.2 --silent-gold 10`
  (silent cells through Platt(e -> gold) fitted on the 29 training experts; Family A has no clean predictions for them, so
  the calibration uses c224 + CoAt). Holdout-29: report 0.883, ensemble 0.937, target **0.947** (labels_r1 0.945).
  Shrinkage matters: C <= 1 flattens silent cells and drops the target to 0.89-0.93; C 10-100 = 0.947; report weight
  0.3 vs 0.5 = +0.002. All of this is inside holdout noise - the LB probe of the students is the judge. Target prevalence
  moves where reports are silent: Synovitis 0.16 -> 0.25, Fracture 0.04 -> 0.08 (Fracture corr with labels_r1 0.54).
- **Running (launched 10:33-10:40 IST):** `r2-c224-f0..4` (c224 k16 recipe, ImageNet init, labels_r2g; ~3 h, ~$7.5).
  `r1-c224k32-f0` at ep 17/20, best gold 0.927 = k16 (0.9265); collect when done, do not extend.
- Page: `label_rounds.json` got `label_source` (the label recipe the page scores) and the page now lists LB submissions.
- **RunPod spend line: ~$69 + round 2 ~$7.5 -> ~$77 of $250. Account balance $12.0 at 10:40 IST: covers round 2 only;
  the next launch needs a top-up.**

## 10:50 IST — label design is exhausted offline (no GPU spent)
- Scored every label design against all 57 experts with report labels (dev half + holdout half; expert predictions put on
  the population scale = percentile inside each model's own OOF distribution). Macro AUC vs experts:
  image ensemble alone 0.935 · report 0.874 · rule a_said .5/.2 (labels_r1) 0.940 · a_said .3/.2 0.943 · .2/.1 0.944 ·
  expert-fitted cell model P(gold | report state, e), cross-fitted, 0.939-0.942. **Every design lands at 0.94-0.945:
  the target is the image ensemble plus ~0.008 from the report.** Students cannot pass ~0.935-0.94 from these teachers,
  whatever the blend rule. The image evidence has to get better, not the label recipe.
- What report silence means, all 58 experts (gold+ / silent cells): Synovitis 14/41, Fracture 4/26, PF OA 3/14,
  Baker's 1/29, Lateral OA 0/22, Medial OA 0/14, others 0-1. Silence = absent except Synovitis, Fracture, PF OA.
  Report YES is confirmed by experts only 44-81 % (MCL 9/18, Contusion 16/36); report NO is reliable (0-2 misses).
- Known weakness of labels_r2g: its silent-cell calibration was fitted on within-29 expert ranks, but experts sit at
  population percentile ~0.64 (enriched sample), so silent-cell targets are too high (Fracture mean 0.07 -> 0.15,
  Baker's and Lateral OA up although silence means absent there). It changes levels, not the order inside silent cells;
  the r2 probe still answers whether silent-cell work moves LB. If it does, redo with the population-scale percentile
  and only for Synovitis / Fracture / PF OA.
- Cross-finding blends (e.g. Effusion into Synovitis) on our image ensemble: no consistent gain on both expert halves.
- Public stack as a clean teacher: not possible. Its CoAt members are full-data models picked on gold-58; only the weak
  DINO family has fold weights.
- Board moved: #1 0.963, #10 0.958 (2026-10-04 10:45 IST).

## Recipe gap found: the trainer has no spatial augmentation (10:55 IST)
- `runpod/train_knee.py` trains with a +-10 % gain jitter only: no rotation / scale / shift, no stochastic depth, no
  mixup. Added `--aug_geo` (per-window affine +-12 deg, +-12 % scale, +-7 % shift, gamma per study; no flips) and
  `--drop_path`; both default off, checkpoints load unchanged, unit-checked locally, 123 tests pass.
- Recipe `c224reg` in `scripts/launch_round.py` (= c224 + `--aug_geo 1.0 --drop_path 0.1`). Fold 0 on labels_r1
  launched 11:05 IST as `r2-c224reg-f0` (~$1.7); compare with r1c2240 (best gold 0.9265) and probe alone on LB.

## 11:12 IST — user topped up $25; c224reg launched; per-finding probes submitted
- RunPod balance $35.7 at 11:03 IST. `r2-c224reg-f0` (pod qwczh1wndejuic, labels_r1, fold 0) launched 11:05 IST.
- **Per-finding LB probes (user go-ahead):** 56815408 `probe-neg-synov`, 56815410 `probe-neg-latmen` = coat-up15-b4
  (0.945) with one column's order reversed (`build_ensemble_kernel.py --negate`). Checked before submit: other 11
  columns identical to the stack, target column reversed. Read as AUC_f = (1 + 12 * (0.945 - LB)) / 2, +-0.006.
- **5 of 5 submits used** for the Kaggle day ending 2026-10-05 05:30 IST; the round-2 probe goes in after that.

## 11:33 IST — k32 collected and closed
- `r1-c224k32-f0`: best gold 0.9270 (k16 0.9265), holdout-29 0.937 (k16 0.931); EMA / last / SWA 0.923-0.924 vs
  0.924-0.928. No gain from 32 slices at twice the cost per epoch -> closed, not taken to 5 folds (~$2.4).
- `r2-c224reg-f0` is on a slow host: 28 min into the corpus download (shard 2 of 4, one broken connection, resuming).

## 13:15 IST — round 2 (labels_r2g) is worse on both rulers; fold 2 vanished
| fold | labels_r1 best gold | labels_r1 holdout-29 | labels_r2g best gold | labels_r2g holdout-29 |
|---|---:|---:|---:|---:|
| 0 | 0.9265 | 0.931 | 0.9120 | 0.922 |
| 1 | 0.9292 | 0.941 | 0.9189 | 0.925 |
- Same recipe, same folds, same seed: labels_r2g is 0.009-0.016 lower in all four readings. The silent-cell expert
  calibration (wrong scale, see 10:50) hurts. **No LB probe for round 2 unless folds 3-4 reverse this.**
  (Correction 13:55: fold 2 was not lost. Its result dataset became visible a few minutes after the pod deleted itself;
  a single "vanished" poll right after a pod finishes is not a loss - `collect_round.py --wait-min` waits 3 polls.)
- `r2-c224-f3/f4` finish ~13:40 IST; `r2-c224reg-f0` at ep 10: gold 0.911 vs 0.919 for the base recipe at ep 10 (slower
  start is expected with augmentation; the last epochs decide), ends ~14:25 IST.
- All four LB submissions still pending (blends since 10:38 IST, per-finding probes since 11:11 IST).
- **RunPod spend line: ~$69 + round 2 ~$7 + c224reg ~$1.7 -> ~$78 of $250. Balance $28.4 at 13:15 IST.**

## 13:58 IST — round 2 complete: labels_r2g is worse, closed without an LB probe
- All five `r2-c224` folds collected. Per fold vs the same fold on labels_r1: best gold -0.006 to -0.017, holdout-29
  -0.006 to -0.025 (ten readings, all lower). Five-fold ensemble on holdout-29: **0.924 vs 0.941**. Largest drops: ACL
  0.955 -> 0.874, Fracture 0.916 -> 0.853, Lateral Meniscus 0.876 -> 0.840; Synovitis 0.755 -> 0.735.
- Reading: lowering the report weight to 0.3 costs the findings where the report is reliable (ACL report AUC 1.00), and
  the expert-scale silent calibration inflates silent cells (Fracture). **Closed: no LB submit, no further label-rule
  variants** (10:50 analysis: every rule ends at 0.94-0.945 vs experts). labels_r1 stays the training label set.
- Round 2 cost ~$7.4. Checkpoints stay in `girishbose/rsna-knee-rp-r2-c224-f{0..4}-out` (not used).

## 14:25 IST — c224reg (affine aug + drop-path) fold 0: no gain, closed
| fold 0, labels_r1 | best gold | last gold | EMA / SWA gold | holdout-29 |
|---|---:|---:|---:|---:|
| c224 (base) | 0.9265 | 0.9248 | 0.924 / 0.928 | 0.931 |
| c224reg | 0.9205 | 0.9156 | 0.914 / 0.914 | 0.926 |
- Slower start, plateau at 0.912-0.920 from epoch 13, train loss 0.771 vs 0.754: regularised, not better. No LB submit;
  not taken to 5 folds. Untested variant: 40 epochs (~$3 a fold) - only on new evidence. `--aug_geo` / `--drop_path`
  stay in the trainer, default off. Cost ~$1.6.
- Our 10-model ensemble (c224 + CoAt-384 students) on all 57 experts, per finding: Synovitis 0.841, Lateral OA 0.842,
  Lateral Meniscus 0.853, PF OA 0.882; the other eight 0.93-0.98. Model Effusion does not beat model Synovitis for gold
  Synovitis (0.809 vs 0.841; blends 0.82-0.83).
- **Today's own-model experiments are all negative** (labels_r2g, k32, augmentation). No pods running.
- **RunPod spend line: ~$69 + round 2 $7.4 + c224reg $1.6 -> ~$78 of $250. Balance $27.1 at 14:25 IST.**

## 17:32 IST — first per-finding probe: stack Lateral Meniscus public AUC ~0.950
- 56815410 `probe-neg-latmen` = **0.870** -> AUC = (1 + 12 * (0.945 - 0.870)) / 2 = **0.950 +-0.006**. Lateral Meniscus
  is at the stack's macro average on the public LB, not a weak finding there (our own ensemble on 57 experts: 0.853 -
  the expert rulers overstate this weakness, or the stack is much better than our leg on it).
- Still pending at 17:32 IST: 56815408 (Synovitis probe), 56814815 / 56814822 (blends, submitted 10:38 IST).

## 17:38 IST — second per-finding probe: stack Synovitis public AUC ~0.914
- 56815408 `probe-neg-synov` = **0.876** -> AUC = (1 + 12 * (0.945 - 0.876)) / 2 = **0.914 +-0.006**.
- With Lateral Meniscus 0.950, the other ten findings average (12 * 0.945 - 0.914 - 0.950) / 10 = **0.948**. The stack has
  no single sink: Synovitis is ~0.03 below the rest (worth +0.003 macro if lifted to 0.95), everything else sits near
  0.95. The 0.018 gap to #1 is spread over all findings, i.e. it is general model strength, not one finding.
- The expert rulers mislead per finding: Synovitis is 0.74-0.84 on our 29 / 57 experts but 0.914 on the public LB.
  Do not aim specialist work from holdout per-finding numbers.
- Blends 56814815 / 56814822 still pending (7 h).

## 17:55 IST — large offline ruler + fold-0 recipe sweep launched (user: do the trial and error on RunPod)
- **`scripts/oof_ruler.py`**: recipe A/Bs on the fold's 870 held-out studies instead of 29 experts; noise +-0.001-0.002.
  Readings vs base `r1-c224-f0:r1c2240` (said 0.8910 / target 0.9673): k32 +0.0003 / -0.0009, c224reg -0.0010 / -0.0006
  (both true nulls), CoAt-384 student -0.0016 / -0.0014 (LB: c224 +0.001 over CoAt-384 at 5 folds - same sign and
  size). Limits: it scores agreement with reports / teachers, so it judges recipes and backbones on the same labels,
  not label changes (labels_r2g: -0.0001 here, -0.017 on holdout-29), and it cannot score the public stack (its models
  saw these studies).
- **Sweep launched ~17:55 IST, fold 0, labels_r1, ~$10:** `r2-c224lr-f0` (bb_lr 1e-4), `r2-c224e30-f0` (30 epochs),
  `r2-c224reg40-f0` (aug + drop-path, 40 epochs), `r2-cnx224-f0` (ConvNeXt-T in22k @224, bb_lr 1e-4). Ends ~20:30-23:30
  IST. Judge: `oof_ruler.py --base r1-c224-f0:r1c2240 --runs r2-c224lr-f0:r2c224lr0,r2-c224e30-f0:r2c224e300,
  r2-c224reg40-f0:r2c224reg400,r2-cnx224-f0:r2cnx2240` plus best gold / holdout-29. A run counts as a gain at
  >= +0.004 on both ruler readings (2 SE) and no loss on the experts; winners go to 5 folds and an own-alone LB probe.
- **RunPod spend line: ~$78 + sweep ~$10 -> ~$88 of $250 (check in at $100). Balance $27.1 before the sweep.**

## 18:35 IST — check-in: blends still pending; own families add +0.002 when blended; plain-BCE run launched
- Blends 56814815 / 56814822 still PENDING at 18:28 IST (7 h 50 min; the stack-only probes took 6 h 20 min, the blends
  carry the five-model leg on top). If Kaggle returns a timeout instead of a score, the leg needs fewer models or one
  eval pass before any re-submit.
- **Own-family blend on the big ruler** (5-fold OOF, 4,349 studies, `said` = macro AUC vs explicit report YES / NO,
  rank mean inside each fold): c224 0.8901 · CoAt-384 0.8896 · RegionMIL 0.8818 · ResNet50 0.8490 ·
  c224 + CoAt-384 0.8920 · c224 + CoAt-384 + RegionMIL 0.8924 (**+0.0024 +-0.0005** over c224 alone) · any blend with
  ResNet50 is lower. Our families are near-copies of each other: a 10- or 15-model own leg is worth ~+0.002, not a step.
- **Which checkpoint ships** (c224, 5 folds, same ruler; the LB leg loads the best-gold-epoch file, the primary OOF is
  the mean of five checkpoints): best 0.8893 · last 0.8874 · EMA 0.8905 · SWA 0.8904 · mean of five 0.8901. EMA is
  +0.0012 +-0.0003 over best: use `raptor_ft_<tag>_ema.pt` in any future own leg (free), but it is not a step.
  Per-checkpoint OOF files for r1-c224 are now on disk (`raptor_oof_<tag>_{best,last,ema,swa,epens}_fold<k>.npz`).
- Reading of the three ruler results (families +0.002, checkpoints +0.001, k32 / aug 0): on the same labels every
  label-neutral change is worth ~0.001-0.002. The `said` reading sits at 0.889-0.892 for everything we train.
- No public notebook above 0.946 (Kaggle code list, 18:30 IST).
- **`--pos_weight_cap`** added to `runpod/train_knee.py` (default 10 = unchanged; 123 tests pass). The base loss
  up-weights positives by (1-prev)/prev: MCL and Fracture x10, Baker's 5.9, ACL 5.2, Lateral Meniscus 4.9 - and report
  YES is the noisy side (experts confirm 44-81 %). Recipe `c224pw1` (= c224, plain BCE); fold 0 launched 18:33 IST as
  `r2-c224pw1-f0` (pod p0891toalsgixe, ~$1.7), same ruler and base as the sweep.
- Sweep at 18:28 IST: `r2-cnx224-f0` ep 8/20 (best gold 0.901 so far, ~1.8 min/epoch), `r2-c224lr-f0` ep 3/20,
  `r2-c224e30-f0` ep 2/30, `r2-c224reg40-f0` still downloading the corpus (30 min, slow host).
- **RunPod spend line: ~$88 + c224pw1 ~$1.7 -> ~$90 of $250 (check in at $100). Balance $26.1 at 18:32 IST.**

## 19:30 IST — blends scored 0.943 / 0.944: the own leg stays out of the stack; ConvNeXt-T closed
- **56814815 (w0.2) = 0.943, 56814822 (w0.3) = 0.944** vs 0.945 for the stack alone (scored 19:23 IST after 8 h 45 min;
  recorded in the ledger). With D4 + 5bc0 w0.2 (0.941 vs 0.943) this is three final-level blends of an own CoAt leg, all
  0.001-0.002 below the stack. The leg is a weaker copy of the stack's CoAt members, not a diverse one. **Per the 17:55
  branch: own leg stays out, no 10-member leg, no per-finding split-half blends.**
- **`r2-cnx224-f0` (ConvNeXt-T in22k @224): closed.** Ruler vs r1c2240: said -0.0063 +-0.0023, target -0.0043 +-0.0014;
  best gold 0.9067 vs 0.9265; holdout-29 0.940 vs 0.931 (inside its noise). As a blend partner on fold 0: c224 + cnx
  0.8914 vs c224 0.8910 (CoAt-384 as partner: 0.8920). Not taken to 5 folds (~$0.6).
- Still running at 19:25 IST: `r2-c224lr-f0` ep 11/20 (best gold 0.910; base was 0.919 at ep 10), `r2-c224e30-f0`
  ep 10/30, `r2-c224pw1-f0` and `r2-c224reg40-f0` still downloading the corpus on slow hosts (shard 2 and 4 of 4; reg40
  lost 83 min to it and ends ~00:30 IST).
- **RunPod spend line: ~$90 of $250 (check in at $100). 5 of 5 Kaggle submits used until 2026-10-05 05:30 IST.**

## 20:36 IST — c224lr (backbone lr 1e-4) fold 0: null, closed
- `r2-c224lr-f0` vs r1c2240: said -0.0021 +-0.0018, target +0.0001 +-0.0008; best gold 0.9226 vs 0.9265 (last / EMA /
  SWA 0.919-0.921 vs 0.924-0.928); holdout-29 0.931 vs 0.931. No gain from a 3x backbone lr -> closed, not taken to
  5 folds (~$1.3).
- Running: `r2-c224e30-f0` ep 20/30 (best gold 0.920), `r2-c224pw1-f0` ep 5/20, `r2-c224reg40-f0` ep 9/40 (0.900).

## 20:40 IST — severity-graded report labels: checked on the 58 experts, weak lead (no GPU, no LLM spend)
- Question (user: why no progress despite GPU access): are the report-YES cells that experts reject (108 of 311 over all
  findings) a severity-threshold mismatch that a graded extraction could fix? All label sets we have are presence-only
  (pilk YES / NO / UNK; `agent/labeler.py` v1 asks for a presence probability, hedging-based, no grade).
- Read the report wording for every report-YES cell of Effusion (34 / 50 confirmed), Contusion (16 / 36), MCL (9 / 18):
  strong wording is confirmed (large effusion, complete MCL rupture), but the middle is not separable by wording - the
  identical template sentence "Small joint effusion" is expert 0, 1, 0; "moderate joint effusion" 0; "grade II sprain
  (partial tear)" of the MCL both 1 and 0; "bone marrow edema" both. Most of the disagreement is reader-to-reader, not
  a threshold the report text reveals. A graded extraction can only help at the severe end, which reports already
  label YES. **Not pursued as a main lever**; a 58-report graded-prompt test (cents) stays available if wanted.

## 20:50 IST — USER DECISION: own-model track stopped; pods terminated
- The user asked for a straight answer on reaching #1. Assessment given: not confident; no identified step is worth more
  than a few thousandths against a 0.018 gap. 0.945 is essentially the public stack (public d4-blend alone 0.946; our
  swaps added ~0.002 over the 0.943 fork). Own models went 0.920 -> 0.933 alone but never passed what is public and do
  not add in the blend (0.943-0.944).
- **User stopped the track and terminated the three running pods (~20:45 IST):** `r2-c224e30-f0` (at ep 20/30, best
  gold 0.920), `r2-c224pw1-f0` (ep 5/20), `r2-c224reg40-f0` (ep 9/40). No results from them; marked `terminated` in
  `agent/runs/label_rounds.json` (collector and page skip them). They are unfinished, not null: plain BCE
  (`--pos_weight_cap 1`, recipe `c224pw1`) and 30 / 40 epochs remain untested.
- **No pods running. RunPod spend ~$84 of $250; balance $21.36 at 20:46 IST (spend $0/h).** 5 of 5 Kaggle submits used
  until 2026-10-05 05:30 IST.

## Next actions (2026-10-04 20:50 IST) — campaign paused on the user's decision
1. **Lock the result:** select 56754655 (`coat-up15-b4`, public 0.945) as a final submission on Kaggle (user action).
2. Nothing is queued. Do not launch pods or submit without a new go-ahead from the user.
3. If the campaign is reopened, the open items are: the three unfinished fold-0 runs above (~$5), more expert-grade
   labels (rules check needed), a clean out-of-fold rebuild of the stack as teacher (estimate +0.002-0.005, unmeasured).
   Closed today: labels_r2g, k32, augmentation (20 ep), ConvNeXt-T, backbone lr 1e-4, own leg in the stack, 10-member
   own leg, checkpoint choice (+0.001), severity-graded report labels (weak lead).
