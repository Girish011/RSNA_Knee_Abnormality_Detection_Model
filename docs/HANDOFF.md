# Handoff — 2026-08-30 (paste into new chat)

Read first: `docs/STATUS.md`, `docs/DECISIONS.md`, tail of `docs/experiments.md`.
GitHub branch: **`cursor/rank1-gpu-launch-1252`** (PR #5). Older rank1 stack also on `cursor/rank1-v6c-57b3` (PR #4).

**⚠ Do not commit Kaggle tokens to git.** Paste token into the new chat or set as a cloud secret `KAGGLE_API_TOKEN`.

---

## TL;DR / immediate next action

1. **GPU weekly quota exhausted** until refresh **2026-09-05** (`kaggle quota`). Used ~33h / 30h.
2. **When quota resets:** push `train-b-fold4-rank1-v6c` first (~3.4h), then GPU `submit-v6c-5fold` (S01b decode-once) → competition submit.
3. **Rank1 KEEP lean:** folds 0–3 done; 2/4 beat v6c weak-val. Full keep/kill = full-58 gold ≥ **0.7073** after fold4.
4. **S01** (ref **55851760**) **TIMED OUT** on hidden test (no score) — not a scored kill. Baseline LB still **0.682**.
5. **Adopted:** v6c labels, frozen DINOv2-B, cache_v1. **Killed:** v8, GCE (null), yunus gap-fill, label micro-tuning.

```bash
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_API_TOKEN="…"   # see chat paste / secret
mkdir -p ~/.kaggle && printf '%s' "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

kaggle quota
kaggle competitions submissions rsna-knee-abnormality-detection -v

# After 2026-09-05:
python3 scripts/push_kaggle_kernels.py train-b-fold4-rank1-v6c-metadata.json
# then S01b:
python3 scripts/push_kaggle_kernels.py submit-v6c-5fold-metadata.json
# when COMPLETE:
kaggle competitions submit rsna-knee-abnormality-detection \
  -k girishbose/submit-v6c-5fold -m "S01b: 5-fold v6c GPU decode-once"
```

---

## Kaggle access

- **User:** `girishbose`
- **CLI:** `~/.local/bin/kaggle` (not on default PATH)
- **Competition:** `rsna-knee-abnormality-detection`
- **Auth:** `KAGGLE_API_TOKEN` env **or** `~/.kaggle/access_token` (chmod 600)
- Token for this campaign was provided in chat — **rotate after handoff** at https://www.kaggle.com/settings

---

## Calibrated scores

| Ruler | Score | Notes |
|---|---|---|
| **Public LB (fold0 probe)** | **0.682** | ref **55818692** |
| **S01 5-fold** | **TIMEOUT / no score** | ref **55851760** — runtime exceeded (CPU, 5× decode) |
| **Full-58 gold OOF v6c** | **0.7023** | adopted baseline; keep margin +0.005 → **0.7073** |
| Full-58 gold OOF v8 | 0.6935 | **KILL** |
| Public LB top | ~0.942–0.952 | gap ~0.26 |

**Calibration:** gold-OOF ≈ public LB **+ 0.02**. Weak-val OOF overstates by ~0.07 — smoke only.

---

## Rank1 results (plane routing + per-label pos_weight)

| Fold | rank1 BEST | v6c BCE | Δ | Status |
|---|---|---|---|---|
| 0 | **0.7558** | 0.7683 | −0.0125 | COMPLETE |
| 1 | **0.7640** | 0.7493 | **+0.0147** | COMPLETE |
| 2 | **0.7706** | 0.7624 | **+0.0082** | COMPLETE |
| 3 | **0.7606** | 0.7636 | −0.0030 | COMPLETE (tie) |
| 4 | — | 0.721 | — | **blocked: GPU quota** |

Local downloads: `outputs/kaggle_rank1/fold{0..3}/` (ckpt + OOF; gitignored).
Kaggle kernels: `girishbose/train-b-fold{0..3}-rank1-v6c`.

---

## Submit queue

| ID | Hypothesis | Status | Kill if |
|---|---|---|---|
| S01 | 5-fold uniform >> fold0 | **TIMEOUT** (no LB) | — |
| **S01b** | GPU + decode-once 5-fold | **queued** (quota) | runtime fail again |
| S02 | Per-label AUC-weighted blend | queued after scored S01b ≥ 0.690 | ≤ S01b |
| Rank1 5-fold | plane routing beats v6c gold | folds 0–3 done; fold4 pending | gold < 0.7073 |

---

## Current production recipe (v6c)

- Labels: `girishbose/rsna-knee-weak-v6c`
- Backbone: frozen DINOv2-B, `girishbose/dinov2-vitb14-rsna-knee`
- Cache: `girishbose/rsna-knee-cache-v1` (3×12×224)
- Checkpoints: `girishbose/train-b-fold{0..4}-weak-v6c`
- Code: `girishbose/rsna-knee-code` + patch `girishbose/rsna-knee-rank1-patch` (full `src/rsna_knee` + folds + decode-once `infer.py`)

---

## Rank1 image stack (in repo + Kaggle patch)

- Plane routing: `plane_routing.py`, `multiseries.py` (`label_plane_routing=True`)
- Config: `configs/rank1_v6c.yaml` (12ep frozen, per-label pos_weight)
- Train kernels: `kernels/train-b-fold{0..4}-rank1-v6c.py` + `*-metadata.json`
- Submit: `kernels/submit-v6c-5fold.py` (S01b: GPU, decode-once, prefers patch on `PYTHONPATH`)
- S02: `kernels/submit-v6c-5fold-s02.py` (per-label AUC weights from OOF)
- Helpers: `scripts/push_kaggle_kernels.py`, `scripts/package_rank1_patch.py`
- cache_v3 builder ready but not trained yet: `kernels/build-cache-v3.py`

---

## Killed (do NOT retry)

v8 TR/EL gap-fill · GCE-as-shipped · yunus gap-fill · cache_v2 · unfreeze · MRI-CORE · 384 · external KneeMRI · v6d label micro-tune

---

## Kernel gotchas

```json
"machine_shape": "NvidiaTeslaT4",
"docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
```

- Max **2 concurrent GPU** sessions.
- Frozen-B ~**3.4h/fold** (12ep rank1 similar).
- S01 failed because CPU + decoding DICOMs once per checkpoint (~126s/study × 5). S01b fixes: GPU + decode-once.
- Live logs: `timeout 60 kaggle kernels logs girishbose/<slug> -f`
- Push: `python3 scripts/push_kaggle_kernels.py <name>-metadata.json` (uses `-p` staging dir)

---

## Next 3 actions

1. **2026-09-05:** push fold4 rank1 → download OOF → full-58 gold vs 0.7023.
2. **S01b** GPU submit + competition submit; record public LB; queue S02 if ≥ 0.690.
3. If rank1 gold wins: consider `build-cache-v3` + retrain; else keep v6c for S01b/S02 only.

---

## Do NOT

- Final anything below ~0.80 projected LB.
- Retry killed experiments.
- Trust 2-fold / weak-val alone for keep/kill (full-58 gold rule).
- Use reports at inference. Commit secrets / weights / DICOMs.
- Burn both finals on near-identical models.

---

## Competition facts

- RSNA 2026 Knee Abnormality Detection — deadline **2026-10-22**
- Metric: **macro AUC**, 12 labels
- Test: images + series metadata only
- Code competition, ≤9h, internet OFF at submit
- Dual finals: max-AUC + efficiency student
