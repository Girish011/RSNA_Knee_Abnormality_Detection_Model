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

## 2026-08-11 — Kill weak_labels_v2 for main train
- **Decision:** Train on `weak_labels_v1` until a new label recipe beats fold0 ablate.
- **Why:** Matched fold0 ablate (`pos_weight=1`): v1 **0.725** > v2 **0.718**; earlier v2+pw1.5 was 0.707.
- **Rejected:** Shipping multilingual v2 as default; full 5-fold on v2.

## 2026-08-15 — Report labels v3 via multilingual zero-shot NLI (train-only)
- **Decision:** Generate `weak_labels_v3.csv` with public mDeBERTa-XNLI + v1 fallback + expert override; never use text at inference.
- **Why:** Keyword v2 and expert head-FT failed; FR-heavy reports need a multilingual reader, not more image tweaks.
- **Rejected:** Paid LLM as the first v3 path; training a text model on 58 experts only.

## 2026-08-15 — Kill expert-only head fine-tune as default
- **Decision:** Do not ship expert head-FT; keep frozen-B (or S) weak_v1 checkpoints.
- **Why:** B FT: 0.7591 → 0.7601 then drop to 0.745 on 45 gold studies.
- **Rejected:** Longer expert FT / unfreezing last blocks on this tiny set.

## 2026-08-15 — Keep cache_v1 (3×12); kill cache_v2 as default
- **Decision:** Train on `cache_v1` until a new cache recipe beats fold0 0.764.
- **Why:** Frozen S on 4×16 cache_v2 peaked at **0.738** (below 0.764 / 0.759).
- **Rejected:** 5-fold or B-first on cache_v2 after a losing S smoke.

## 2026-08-12 — No backbone unfreeze until frozen-B wins
- **Decision:** Default train recipe keeps DINOv2 backbone frozen for all epochs; unfreeze only as a deliberate later experiment with tiny LR / last-block-only.
- **Why:** B fold0 freeze→×0.05 LR unfreeze collapsed 0.729→0.615 (same as early S).
- **Rejected:** Staged unfreeze as the default main-track schedule.
