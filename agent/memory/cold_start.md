# Cold-start knowledge base

Static half of the agent's retrospective memory. One bullet = one retrievable record.
Sources: `docs/DECISIONS.md` (2026-08 to 2026-09-21), public Kaggle discussion/datasets.
Add a bullet only when a result is measured; never add opinions.

## Settled kills (thin recipe: frozen DINOv2-S, 3 series x 12 slices x 224, weak_v1)

- KILL train longer: gf_v0c gold OOF peaked at epoch 4 (0.6317) then declined while weak-val kept rising; teacher overfit.
- KILL backbone unfreeze on noisy report labels: paired gold collapse 0.6317 to 0.5177 on the first unfrozen epoch.
- KILL 12 to 24 slices on the thin recipe: true OOF gold 0.6080 vs 0.6144, inside noise, no gain.
- KILL 224 to 336 on the thin recipe: gold 0.7281 to 0.6925 (fold0 ruler, directional only).
- KILL Med Meniscus gap-fill inside the existing study pool: +20 cells, Med Men gold +0.001; earlier +0.13 was a study-admission confound.
- KILL expert-only head fine-tune on 58 gold studies: 0.7591 to 0.745.
- KILL MCL LLM fills: measured fill precision 0.23 to 0.27 on the 58 experts; reports describe MCL inconsistently.
- KILL keyword weak_labels_v2 as default: fold0 0.718 vs v1 0.725.

## Ruler facts

- Gold-58 OOF macro seed sd is 0.0214; single-seed deltas under ~0.043 are noise. Use gold-58 only as a veto.
- Weak-ruler OOF sd is 0.0085 on ~2449 studies; quieter but measures agreement with the teacher, not truth.
- Seed averaging (3 seeds) beat every single seed (0.6173 vs best 0.6144); treat seed ensembling as part of the model.
- Best-on-weak-val checkpoint picking cost ~0.017 gold vs a fixed epoch; prefer fixed epoch or last-k averaging.
- Worst per-label seed sd: Effusion 0.133, Lateral OA 0.090, Baker's 0.086, MCL 0.080.

## Public results (other teams, measured on the public LB)

- Density beats span: 44 slices over 6-94% scored 0.926; the same 44 over 2-98% scored 0.917; 64 slices over 2-98% restored 0.928; 80 slices scored 0.932.
- Window count on fixed weights: 42 windows 0.924, 62 windows 0.927, 78 windows 0.932 (same checkpoint).
- Three backbones (ConvNeXtV2-B 336, CoAtNet 384, EffNetV2-L 480) rank-mean blend 0.915 vs 0.914 single; ensembling bought ~0.001.
- Changing slice sampling of the corpus was worth 0.006 to 0.011 LB, ~3x the model lever.
- SWA averaged out to nothing across six pairs.
- Single-model DINOv2-based submissions reach 0.915 to 0.942 at 224px; 224 is not the bottleneck.
- Public notebook rsna-base (DINOv2-S + Max-Span Dense Corpus + twelve-findings weights) scores 0.936 in ~3 minutes on T4x2.
- Public leaders train on report-distilled soft labels, not hard keyword labels.
- Fluid_Sensitive and Fat_Suppression flags are identical on all 24,371 training series; treat as one feature.

## Label audits (report labels vs 57 expert studies, 2026-09-24)

- Public pilkwang/rsna-knee-llm-labels scores macro AUC 0.870 [0.831, 0.900] vs experts; keyword v1 0.628, v7 0.651; pilk minus v1 = +0.242 [0.193, 0.290] paired.
- pilk per-label AUC: ACL 0.997, MCL 0.976, Med Men 0.943, Baker's 0.932, Med OA 0.908, PF OA 0.891, Fracture 0.871, Lat Men 0.841, Effusion 0.830, Lat OA 0.789, Contusion 0.768, Synovitis 0.694.
- pilk fails the precision >= 0.5 gate only on Contusion (0.44); keyword v1/v7 fail PF OA (0.25).
- dreaddevelopment/rsna-knee-labels omits all 58 expert studies; it cannot be graded on them. Mean Spearman with pilk on non-expert studies is 0.81; with keyword v1 0.33.
- pilk's labeler was prompt-checked against the annotated studies, so its expert AUC is likely optimistic by an unknown amount.

## Own trains (RunPod, public 44-slot corpus, CoAtNet-384 16 ep, seed 42, all 58 experts as val)

- Teacher A/B dread vs blend_v2 (pilk+dread, dread where pilk UNK): gold-58 best 0.9148 vs 0.9148 (paired CI -0.015..+0.015), SWA 0.9138 vs 0.9132; teacher swap is a wash.
- blend_v2 per label vs dread: Lateral Meniscus +0.04 to +0.06, Fracture +0.02 to +0.03, MCL -0.03 to -0.05, Contusion -0.02 to -0.03 (single seed, noise-scale).
- blend_v2 learned faster early (epoch 1 gold 0.811 vs 0.783) but both peaked at epoch 10; SWA of top-3 epochs did not beat the best epoch.
- One 16-epoch CoAtNet-384 run is ~17.5 min/epoch on an RTX 3090 (~4.7 h), ~$1.0-3.5 depending on host.

## Own trains (protocol fold-0 + expert holdout)

- Baseline CoAtNet-384 / dread / seed42 fold0: proxy 0.9215, gold_dev 0.8878, holdout 0.9251 (node 7421b1b6a6fa).
- Teacher A/B dread vs blend_v2 on all-58 was a wash (0.9148 vs 0.9148); prefer targeted relabels over whole-teacher swaps.
- relabel_8c56cdf0 (dread + pilk_unk fills for Lat Men + Fracture): Lat Men dev AUC 0.965 (pilk 0.886), Fracture 0.912 (pilk 0.878); train it next.
- Campaign keeps searching until fold-0 proxy >= 0.958 or budget is exhausted; never park for hours on a planner stop.

## Constraints

- Reports and any text model are train-only; test.csv has no Report field.
- External data must be public and free; MRNet/OAI registration-gated status is unruled by hosts.
- Submission notebook: offline, under 9 hours, writes submission.csv, reads test DICOM itself.
