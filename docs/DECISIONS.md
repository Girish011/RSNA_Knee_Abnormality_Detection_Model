# DECISIONS

Format: date | decision | why | rejected

## 2026-08-08 — Backbone: Meta DINOv2
- **Decision:** Use public DINOv2 (`vits14` baseline/efficiency, `vitb14` main).
- **Why:** Already leading early Model Hub usage; competition allows public pretrained models; we compete on wrapper/data/labels not novel ViTs.
- **Rejected:** Train ViT from scratch; primary 3D ConvNet path (too slow/data-hungry for first iterations).

## 2026-08-08 — Reports train-only
- **Decision:** Reports used for weak labels / pretraining only; never at inference.
- **Why:** `test.csv` has no reports; leakage if we design multimodal test-time fusion.
- **Rejected:** Late-fusion multimodal inference depending on report text.

## 2026-08-08 — Dual finals
- **Decision:** Final #1 max-AUC ensemble; Final #2 distilled DINOv2-S efficiency specialist.
- **Why:** Efficiency prize requires eligible accurate+fast submission selected as a final.
- **Rejected:** Single submission hoping to win both; two near-identical slow ensembles.

## 2026-08-08 — Compute/storage
- **Decision:** Kaggle notebooks/datasets as primary data+train environment; local Mac for code/docs/tests only.
- **Why:** Full dataset 569 GB; local free disk ~129 GB.
- **Rejected:** Download full raw DICOM tree to laptop as training store.

## 2026-08-08 — Continuity via repo, not chat
- **Decision:** STATUS / DECISIONS / experiments / WEEKLY + Cursor rule are source of truth.
- **Why:** 2-month campaign; chat context rot is guaranteed.
- **Rejected:** Relying on long agent threads as project memory.

## 2026-08-08 — Kaggle-first data layout
- **Decision:** Keep full DICOM tree on Kaggle; locally download **CSV metadata only**; train/cache/submit on Kaggle notebooks.
- **Why:** Dataset is 569 GB; Mac has ~129 GB free.
- **Rejected:** Full local `train_series/` download.

## 2026-08-08 — Auth via access_token
- **Decision:** Use modern `~/.kaggle/access_token` (KGAT), not legacy `kaggle.json`.
- **Why:** Matches current Kaggle API settings UI; CLI 1.7+/2.x supports it.
- **Rejected:** Requiring only legacy kaggle.json in scripts.

## 2026-08-08 — Expert labels are tiny
- **Decision:** Treat report weak supervision as first-class (not optional); expert 58 studies are fine-tune / audit set.
- **Why:** Only **58 / 4407** studies have expert labels; all reports non-empty; multilingual (e.g. Spanish in sample).
- **Rejected:** Expert-only training as the main path.

## 2026-08-09 — Weak labels EN+ES v1
- **Decision:** Ship keyword weak labels with Spanish synonyms (LCA, Derrame, etc.); keep high precision / lower recall; confidence threshold 0.5.
- **Why:** English-only missed common ES reports (macro F1 ~0.33 → ~0.44 on expert audit).
- **Rejected:** Jumping straight to paid LLM labeling before a measurable keyword baseline.
