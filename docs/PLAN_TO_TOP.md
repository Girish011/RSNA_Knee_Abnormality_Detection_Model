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
