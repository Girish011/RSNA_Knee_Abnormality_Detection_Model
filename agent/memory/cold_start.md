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

## Constraints

- Reports and any text model are train-only; test.csv has no Report field.
- External data must be public and free; MRNet/OAI registration-gated status is unruled by hosts.
- Submission notebook: offline, under 9 hours, writes submission.csv, reads test DICOM itself.
