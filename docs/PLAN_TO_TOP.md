# Plan to the top (2026-10-01 → lock 2026-10-16)

Target: public LB 0.950+ (top-86 band) on the way to the top band (0.955–0.961). Current 0.944.

## The insight this plan is built on (measured 2026-10-01, 58 experts, pooled)
| Model family | Macro | Weak five | Lat Men | PF OA | Synovitis |
|---|---:|---:|---:|---:|---:|
| Public DINO base (OOF) | 0.843 | 0.800 | 0.67 | 0.80 | 0.78 |
| Public RadImageNet (OOF) | 0.811 | 0.789 | 0.69 | 0.81 | 0.76 |
| **Our CoAtNet (held-out)** | **0.882** | **0.867** | **0.88** | **0.87** | 0.81 |

Our CoAtNet line is the strongest family, and our fused labels agree with experts at **0.946** (report labels 0.883).
The public stack gives ~40% of its weight to the two weakest families, and every public model learned from report labels.

## Strategy: three moves, then one grand soup
1. **Cross-architecture distillation ("bring our intelligence into every model").** Retrain the weak public families
   on our fused labels with their own public trainers (`scripts/fork_public_trainer.py`, free Kaggle GPU). Fused labels
   were flat for CoAt (self-distillation) but are new information for DINO / RadImageNet, which never saw it.
   Order: RadImageNet E13 + main heads (running) → A5 folds → DINO members (largest share, plan quota).
2. **Rebalance the stack toward the strongest family.** We now hold OOF for DINO, RadImageNet and our 5-fold CoAt on
   all 4,407 train studies → fit per-finding branch weights on 4,407 studies (targets: fused labels; validated on the
   58 experts), instead of the hand-tuned public weights. Probe `coat-up15-b4` measures the direction tonight.
3. **Strengthen the CoAt family itself.** Our soups in every Raptor slot (probe `raptor-all-b4`), then v8/v10
   fine-tunes on fused/bestof labels (RunPod, after top-up); specialists (k16 PF OA, weak5 Synovitis) folded in.
4. **Grand soup / ensemble selection.** Every upgraded component + learned weights; greedy forward selection on OOF,
   confirmed on experts and LB. Two finals: best LB, best OOF+expert.

## Calendar (one LB submit per day minimum; only candidates expected to beat the current best)
| Date | Work | LB candidate |
|---|---|---|
| Oct 1 | probes raptor-all / coat-up15; 4 RadImageNet forks (bestof + fused) | probes |
| Oct 2 | read forks; wire winning heads (`--rad-heads`); OOF branch-weight fitter | heads swap |
| Oct 3 | learned branch weights (+ heads) | rebalanced stack |
| Oct 4–6 | A5 transplant (fused); v8/v10 fine-tunes if probe positive | best combined |
| Oct 7–11 | DINO members transplant (fused), quota-planned | best combined |
| Oct 12–14 | grand soup + ensemble selection | finals candidates |
| Oct 15–16 | lock finals A (best LB) / B (best OOF+expert) | — |

## Gates (each step is kept only if it measures up)
- A fork wins when its trainer's gold OOF macro (58) beats the upstream audit (E13 0.8576, main 0.8543).
- A stack change ships to LB when expert/OOF evidence says it beats the current best.

## Roles
- Claude: architecture, approvals, submissions (user-approved policy: only candidates expected to beat best).
- DeepSeek strategist: analysis + proposals only (GPU spend via host approval).
- Kaggle GPU (free, 2 concurrent, 30 h/week): forks. RunPod (top-up): CoAt fine-tunes.

## Results log
- 2026-10-01 22:30 — RadImageNet forks on bestof_v1 (gold OOF macro, 58): **E13 0.8612 vs upstream 0.8576 (+0.004;
  blend α0.5 0.8701 vs 0.8654)**; main V52 0.8388 vs 0.8543 (−0.016: MCL/MedMen/Contusion/Synovitis down). Mixed →
  not wired yet. Fused-label forks (expert agreement 0.946) running: fork-e9main-fused-full, fork-e13-fused-full.
  Heads/OOF saved under outputs/rad_heads/ (gitignored weights).
- 2026-10-02 04:15 — **LB coat-up15-b4 = 0.945 (new best)**; raptor-all-b4 = 0.944 (more soup in v8/v10 slots: no).
  **Fused-label forks win:** E13 heads gold OOF 0.8745 vs upstream 0.8576 (+0.017; bestof 0.8612); main heads 0.8594 vs
  0.8543 (+0.005; LatOA +0.05, Syn +0.035, MedMen +0.036, Fracture +0.06; LatMen −0.06, PF −0.04 — LatMen is CoAt-only
  in the stack). Cross-architecture distillation confirmed → A5 / DINO next on fused labels.
  Heads: `girishbose/rsna-knee-rad-heads-fused`. Candidate `coat15-radfused` (coat +0.15 + both fused heads) pushed.
- 2026-10-02 04:55 — `coat15-radfused` verified (ref heads 829bc…, E13 d87c…, coat +0.15) → **submitted ref 56760305**.
  `coat30-radfused` (coat +0.30, same heads) auto-submits after the 05:30 IST quota reset as a direction probe.
- 2026-10-02 12:00 — `coat15-radfused` = **0.944** (base 0.945): fused RadImageNet heads, +0.017 gold OOF, did not
  transfer to LB. Diagnostic (gold-58, OOF): DINO 0.840, RAD e13-fused 0.875, main-fused 0.859, our 5-fold CoAt 0.931;
  every transformer mix lowers gold-58 — yet our CoAt alone is 0.920 on LB. Gold-58 overstates anything trained on our
  labels → use it only as a sanity gate; ship decisions on LB probes. DINO fine-tune kernel built (CPU smoke passed:
  fingerprints 1e-5, prefix SHA = stack pin 0a55b893) and running on Kaggle.
- 2026-10-02 15:00 — coat30-radfused = 0.944 (no gain past +0.15). **DINO fine-tune (20/20, 3.3 h free T4x2) passes
  gate: gold OOF 0.8386 → 0.8543** (Syn +0.07, ACL +0.044, Eff +0.028; LatMen/LatOA/PF flat), fused agreement
  0.917 → 0.929, pixel check corr 0.998. Weights `girishbose/rsna-knee-dino-ft-fused`; candidate `coat15-dinoft`
  (= coat-up15-b4 + DINO swap) pushed. Strategist paused (user); RunPod ~$20 at Claude's discretion.

## Next build: DINO warm-start fine-tune on fused labels (design, ready to implement)
- Members: `pilkwang/rsna-knee-weights` (20 × `m_*.pt` + `manifest.json`), all ONE pixel group: 336 px, 12 slices,
  group 3, crop 130 mm, slots SAG/COR/AX_FLUID_FS + SAG_FLUID_NOFS + COR_T1 + SAG_T1, dinov2-small, unfreeze_last 6;
  4 seeds (base-s336x12, -seed2, -seed3, -seed4) × folds 0–4. Public OOF: `oof.npz` (pred/ids/gold_mask, 4407).
- Fold rule (public trainer): `grp = int(md5(Report)[:8], 16) % 5`; member fold k = holdout grp k; gold studies stay in
  training at weight 3 unless in the holdout fold.
- Kernel = fork of `pilkwang/rsna-knee-baseline-v1` (has build_cache / build_model / fingerprint / predict). New main:
  decode train once → for each member: load + check_fingerprint → fine-tune 2–3 epochs at ~1/5 public LR on
  tr = {grp != fold} with targets = fused labels (`labels_fused_full.csv`, w = .25+.75·certainty; gold w=3) → predict
  its holdout fold (OOF) → recompute fingerprint → save `m_<newid>.pt` + manifest v2.
- Gate: rank-mean OOF of the 20 new members on the 58 gold > public `oof.npz` (gold macro 0.8425).
- Wire: stack cell 15 (`SHARED_DINO_PREFIX_SHA256`, 20/20 fingerprint gate, manifest/dataset path) — frozen prefix
  (first 6 blocks) is unchanged by fine-tuning, so the prefix SHA should still match.
- Then the same recipe for A5 (`mattiaangeli/knee-mri-fold-weights` m_f0..4; trainer not yet located).
- 2026-10-02 12:35 — **LB: coat15-radfused 0.944, coat30-radfused 0.944** (best stays 0.945 coat-up15-b4).
  Clean check (pooled AUC on holdout-29, never used by the fusion): fused heads ARE better — E13 0.8185 → 0.8400,
  main 0.8128 → 0.8478 (worse on dev-29, so no fusion leak). **Cause found:** stack cell 25 feeds a fitted linear
  calibrator (`_RAD_CAL_PAYLOAD`) with baseline_rank, reference_rank, second_rank and their differences, fitted to the
  PUBLIC heads (and public DINO base); on gated findings it supplies 40% of the value. Swapping inputs mis-calibrates it.
- **Rule from now on:** transplanted components enter OUTSIDE the calibrator. Keep the public heads/DINO feeding the
  calibrator unchanged; blend our component's rank after `final` (cell 25) or at the branch level, with weight β.
  Gate every component on pooled AUC over holdout-29 (not all-58: fused labels used dev-29).
- Next: (1) builder patch `--rad-extra β`: compute our heads' ranks from the same encoded features, blend into `final`
  after calibration; LB probe on top of coat-up15-b4. (2) DINO fine-tune kernel (design above), same outside-calibrator
  wiring, gate on holdout-29. (3) per-finding coat shift (the one lever that has moved LB).

## 2026-10-02 night — PIVOT: own-model track (user: go all in for #1)
LB board: #1 0.961, #10 0.958, #49 0.954; best public notebook 0.946 (`pjmathematician/rsna-knee-d4-blend`); us 0.945.
The 0.01 gap is teams' own models — public-stack tweaking cannot close it. Lessons that shape the build:
1. **Effective weight decides LB visibility.** After CoAt +0.15 the transformer branch holds ~0.10–0.25 of the final
   weight; our DINO at β0.3 inside it was ~0.03–0.08 → invisible (0.944 ×3). d4-blend moved LB with a *diverse* family
   at **w=0.45 on the final submission**. Single-finding tails are invisible (1/12 of a per-finding delta).
   → New families enter at the FINAL level, on all findings they help (≥5), with w 0.3–0.45.
2. **Ruler.** No clean offline ruler exists for stack-level changes (public components trained on all train studies,
   incl. gold). For OUR families: (a) pooled AUC on holdout-29 (never used by label fusion), (b) 5-fold OOF AUC vs
   report-derived labels (`pilk`/`dread`, independent of our fusion) on ~4,349 studies, (c) rank-correlation vs the
   stack (lower = more blend value). A family ships to LB only if (a) and (b) both beat our CoAt soup on the weak five.
3. **Inputs we already own:** dense80 cache (80 slots × 336 px, 4,407 train; Sag fluid 22 / Sag non-fluid 18 / Cor fluid
   15 / Cor other 10 / Ax 15; test-time builder = public widedense code) on Kaggle kernel outputs
   `girishbose/rsna-knee-dense80-s0..s3`, fetched to pods by `runpod/fetch_dense80.py`. Kaggle-side 12-slice 336 px
   DINO decode (`build_dino_ft_kernel.py`) also works (33 GB memmap, 1 h decode).

### Family A — weak-five specialists (RunPod, $20 budget, start now)
- Region-aware 2.5D model on dense80: per finding, attend over the plane/sequence slots that show it (Lat Men/Lat OA:
  Cor fluid + Sag fluid lateral half; PF OA: Ax + Sag; Synovitis/Effusion: Ax + Sag fluid; Contusion: all fluid);
  crops at higher effective resolution (e.g. lateral/medial half of coronal at 448) instead of the whole 140 mm field.
- Backbone different from CoAt for diversity (ConvNeXt-S / EfficientNetV2-S, ImageNet-21k), slice-attention MIL head.
- 5-fold (md5(Report) folds), labels = fused (gold w 3), 2 seeds; rulers (a)(b)(c); then inference cell in the stack
  using the dense80 test builder, blended at final level on the weak five (w 0.3–0.45).
### Family B — fastMRI+ (when the user's NYU access arrives; docs/EXTERNAL_DATA.md)
- 974 coronal PD/PDFS volumes, 16,168 boxes (meniscus tear, cartilage loss, contusion/edema, effusion, ACL, MCL, cysts).
  Pretrain a localized meniscus/cartilage/edema detector → initialise Family A's coronal branch. Links file
  `~/.rsna_ext/fastmri_links.txt` (never print / commit). Derived data private only.
### Calendar (lock moves to ~Oct 20; official deadline Oct 22)
| Oct 3–5 | Family A data module + model, 1-fold smoke on RunPod, rulers | — |
| Oct 6–9 | Family A 5-fold × 2 seeds; stack inference cell; LB probe at final level | probe |
| Oct 8–14 | Family B (fastMRI+) pretrain → A coronal init; iterate on weak five | best |
| Oct 15–20 | grand blend (stack + A + soups), finals A/B | finals |

### Family A — build log
- 2026-10-03 — Built `runpod/family_a.py` (RegionMIL): ConvNeXt-S in22k shared tile encoder; tiles = 3-slot 2.5D windows,
  Sag/Ax full 320 px, coronal split into two overlapping halves upsampled ×1.5 (384×288); tokens + series/half and
  slice-position embeddings → 2-layer transformer mix → 12 finding queries with routing masks (PF OA ← Ax+Sag,
  Syn/Eff/Baker ← Ax+Sag, menisci/OA ← Cor+Sag, Contusion ← fluid series). Consistent 3D mirror aug (cor/ax flip +
  sagittal order reversal). Folds `folds_nonexpert_5.json` (same as CoAt f-7cdb6e8357), fused labels
  certainty-weighted, weak five ×2 loss, dev-29 at w3, holdout-29 never trained. No epoch selection: members = EMA + last.
  Pods: `launcher.launch(..., corpus="dense80", trainer="family_a.py")` (run_job.sh `TRAINER`). Tests `tests/test_family_a.py`.
- Rulers: `scripts/family_a_rulers.py --fa '<result dirs>'`. CoAt 5-fold baseline: holdout-29 macro 0.943 / weak5 0.912
  (saturated: LatOA 1.000 → weak ruler); OOF weak5 vs pilk 0.887, vs dread 0.920. (b) is the main ruler.
- 2026-10-03 — **fa-f0-v2 (fold 0, 10 ep, A40, $1.1) FAILS the rulers.** Weak5: holdout-29 0.892 vs CoAt 0.912; OOF vs
  pilk 0.842 vs 0.884, vs dread 0.875 vs 0.923; rank blend w0.3 LOWERS CoAt (pilk 0.884→0.878); rank-corr vs CoAt
  0.85–0.96 (not diverse). Its fused-label OOF looked great (weak5 0.937 vs CoAt 0.911) — **misleading**: fused@0.5 keeps
  only clear-cut positives (ACL prev 0.14 vs pilk 0.29) and on Syn/Contusion/Effusion the fused label is dominated by the
  old image model (Syn prev 0.26 vs report 0.12), which FA reproduced (0.98) = circular. **Rule: never judge a
  fused-label-trained model on fused labels; only report labels + holdout-29.** Results `outputs/runpod_results/fa-f0-v2`.
- Family B running (`fb-f0-v1`): fastMRI+ heatmap pretraining of the same encoder → identical fold-0 fine-tune = clean A/B.
- 2026-10-03 — **Family B A/B (fb-f0-v2, fold 0): fastMRI+ init does NOT pass.** Pretrain (1,172 vols, 10 ep, 70 min)
  held-out tile AUC meniscus 0.959 / cart_full 0.982 / bone_edema 0.955. Fine-tune vs ImageNet start (fa-f0-v2):
  OOF weak5 vs pilk 0.833 vs 0.842, vs dread 0.869 vs 0.875 (−0.006…−0.010; CoAt 0.884/0.923); holdout-29 weak5
  0.923 vs 0.892 (CoAt 0.912; blend50 w/ CoAt macro 0.948 vs 0.943) — n=29 too noisy to overrule 870 OOF studies.
  Fused-label OOF: early lead on LatMen/Contusion faded by ep6. Rank-corr FM vs ImageNet start 0.90–0.96.
  → per decision rule: **skip fastMRI DICOMs**. Cost: prep 1.9 h (serial xz). Results `outputs/runpod_results/fb-f0-v2`.
