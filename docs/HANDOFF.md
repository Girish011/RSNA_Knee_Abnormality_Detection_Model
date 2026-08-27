# Handoff — 2026-08-27 (cloud-agent session)

Paste this into a new chat. Repo docs are the source of truth: read `docs/STATUS.md`,
`docs/DECISIONS.md`, and the tail of `docs/experiments.md` first.

## TL;DR / immediate next action
- **LB probe DONE:** `girishbose/rsna-knee-submit-v6c` Version 1 → **public LB 0.682**.
  Calibration: full-58 gold OOF 0.7023 ≈ LB + 0.02; weak-label OOF ~0.75 is not usable.
  Do **not** select as final. Next needs rotated Kaggle token → competitor label cross-check + v8 build/retrain.

## CRITICAL: GitHub is a stale mirror
- The GitHub repo (`Girish011/RSNA_Knee_Abnormality_Detection_Model`, branch `main`) is an
  OLD snapshot. The **authoritative current code is the Kaggle dataset `girishbose/rsna-knee-code`**
  (has `consensus_labels.py`, `label_query.py`, `pretrain_report_alignment.py`, updated
  `train_baseline_fold.py` with `--seed/--disable-amp/--model-type/--feature-dir`, notebooks 15–22).
- This session committed NEW work to GitHub `main` that is NOT yet in the Kaggle code dataset:
  `src/rsna_knee/text/fill_policy.py`, `src/rsna_knee/text/weak_labels_v7.py`,
  `src/rsna_knee/evaluation.py`, `scripts/oof_report.py`, `configs/labels_v3_robust.yaml`,
  robust losses in `training/loss.py`, plus tests. To use these on Kaggle, fold them into the
  `rsna-knee-code` dataset (package + `kaggle datasets version`).

## Kaggle access
- User: `girishbose`. Auth: KGAT token via `export KAGGLE_API_TOKEN=<token>` and/or
  `~/.kaggle/access_token` (chmod 600). CLI at `~/.local/bin/kaggle` (not on default PATH).
- **SECURITY: rotate the API token** — it was pasted in a chat this session.

## Competition facts (verified this session)
- Live: RSNA 2026 Knee Abnormality Detection. Deadline **2026-10-22**. Metric **macro AUC** over
  12 binary labels. **Test is images + series metadata ONLY** (test.csv has just StudyInstanceUID;
  reports are TRAIN-ONLY). Code competition, ≤9h, internet OFF at submit.
- **External public data IS allowed** (incl. pretrained models), bundled offline.
- Train: 4407 studies; **only 58 fully-labeled ("gold")**; rest have multilingual reports.
- 7 report languages: EN, Turkish, Spanish, German, **Greek (Greek script)**, Dutch, French,
  + ~428 unplaceable. **Turkish negates AFTER the term** ("efüzyon izlenmedi" = NO effusion).
- Host grades **borderline / "on the fence" as NEGATIVE** (favor specificity). Ground truth =
  2 MSK radiologists + adjudicator, whole-exam, single knee.
- The 58 gold are NOT representative (ACL prevalence 41% vs ~20% corpus) → do NOT calibrate on them.
- External top-team signal (public): bigger backbones / ensembles / TTA / extra corpora ≈ 0;
  LB noise-limited in the 3rd decimal; the lever is "how you learn from a noisy teacher."

## Current best & honest numbers
- **Adopted labels: v6c** (constrained-Qwen fills + keyword skeleton, dropping the coin-flip
  ACL/MCL/LatOA LLM fills). Kaggle dataset `girishbose/rsna-knee-weak-v6c`.
- **Model: frozen DINOv2-B, 8 epochs, never unfreeze, cache_v1 (3×12×224), pos_weight 1.0.**
  Checkpoints = kernel outputs `girishbose/train-b-fold{0..4}-weak-v6c` (`fold{F}_best.pt`).
- **Full 5-fold OOF vs 58 gold: v6b 0.6895 → v6c 0.7023** (weak-label OOF ~0.75 overstates by ~0.06).
- Per-label v6c gold (noisy, n=58): MedialOA .904, Effusion .795, Synovitis .766, PFOA .745,
  MedMen .719, Baker's .706, Contusion .691, MCL .667, LatMen .666, LatOA .609, **ACL .604**,
  **Fracture .556**.
- vs ~0.94 public top → large remaining gap.

## Hard-won lessons (do NOT relitigate)
1. **Per-label gold AUC at n≤58 is NOISE.** Proof: v6c and v6d share identical ACL+MCL labels
   yet ACL gold swung 0.77↔0.60 (only Lateral OA differed). Only the **full-58 macro** is stable.
   Sub-0.02 deltas are unresolvable. STOP tuning labels on the 58; stop reading 2-fold gold.
2. **External-image datasets don't help here.** KneeMRI (Croatia, `sohaibanwaar1203/kneemridataset`,
   736 sag volumes, ACL prevalence 24.8%) ruler: our v6c model scored AUC **0.53** (≈chance) even
   after adapter fixes → domain shift + multi-series mismatch. MRNet is gated (Stanford DUA).
   Matches "extra corpora ≈ 0". Ruler kernel: `girishbose/knee-acl-ruler-v6c`.
3. **Coin-flip labels poison training.** The v6→v6b unblock was dropping LLM fills whose 58-expert
   fill-precision < 0.5 (exactly MCL). Encoded in `src/rsna_knee/text/fill_policy.py`.

## v7 multilingual extractor (built this session, on GitHub main)
- `src/rsna_knee/text/weak_labels_v7.py` (+ tests). Adds Turkish + Greek with correct
  post-negation (Turkish), normalcy→negative, borderline→abstain; fixes Turkish dotted-i
  (İ/ı) `re.IGNORECASE` fold bug; delegates other languages to v2.
- v7 vs Qwen agree **88% (Turkish) / 77% (Greek)** where both commit → both capture real signal.
- v6c already labels TR/EL via Qwen, so v7 ALONE isn't a clear win. **Best use = v7∩Qwen
  consensus** (label where both agree, else abstain) for high-precision multilingual supervision.
- NOT yet retrained/measured (blocked on the measurement problem → the LB probe / a real ruler).

## Kaggle assets created this session
- Datasets: `girishbose/rsna-knee-weak-v6b`, `-v6c`, `-v6d` (candidate label CSVs).
- Train kernels: `train-b-fold{0..4}-weak-v6b`, `train-b-fold{0..4}-weak-v6c`,
  `train-b-fold{0,1}-weak-v6d` (frozen-B fold runs; outputs incl. `fold{F}_best.pt`, `fold{F}_oof.csv`).
- `knee-acl-ruler-v6c` (external ruler, negative). `rsna-knee-submit-v6c` (LB probe, ready).
- Pre-existing assets: `dinov2-vitb14-rsna-knee` (`dinov2_vitb14_pretrain.pth`),
  `dinov2-vits14-rsna-knee`, `rsna-knee-cache-v1` (kernel output → `cache_v1`),
  `mri-core-vitb-rsna-knee` (killed path).

## Kaggle kernel gotchas (must-follow)
- **Pin GPU + image or you get `CUDA: no kernel image`**: kernel-metadata must include
  `"machine_shape": "NvidiaTeslaT4"` and
  `"docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"`.
- **Max 2 concurrent GPU sessions** → launch folds in pairs.
- Attach kernel OUTPUTS via `kernel_sources`; datasets via `dataset_sources`; comp via `competition_sources`.
- Use an O(n) filename index, not per-item `rglob`, when iterating many files.
- Frozen-B 8ep ≈ 3.4h/fold. Fetch results with `kaggle kernels output <slug> -p <dir>`; logs are JSON lines.

## Next levers (ranked; measure before trusting)
1. **Get the LB number** (the pending Submit) → calibrate 0.70 gold-OOF; unblocks all decisions.
2. **v8 = v7∩Qwen consensus labels** → retrain frozen-B 5-fold → compare full-58 gold to v6c 0.7023
   (only trust the full-58 macro, and the LB if you probe again).
3. In-domain label cross-check vs competitors' public RSNA-knee label sets
   (`barun2104/rsna-knee-stratified-folds-and-llm-soft-labels`, `dreaddevelopment/rsna-knee-labels`,
   `yunusgmsoy/rsna-knee-llm-report-labels`) — zero domain shift; find our label errors.
4. Fold the v6b/v6c drop-coin-flip rule and v7 into `consensus_labels.py` in the code dataset so
   labels regenerate deterministically.

## Do NOT
- Submit repeatedly / burn finals before OOF (or a calibrated LB) clearly beats the bar.
- Tune labels on the 58 or trust per-label / 2-fold gold deltas (noise).
- Reopen killed paths without new evidence: unfreeze, expert head-FT, cache_v2, 384 rank,
  MRI-CORE, report-alignment-as-init, external-image training.
- Use reports at inference. Commit secrets (kaggle token, weights, DICOMs).
