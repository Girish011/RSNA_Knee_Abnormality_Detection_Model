# Handoff — 2026-08-30 (paste into new chat)

Read first: `docs/STATUS.md`, `docs/DECISIONS.md`, tail of `docs/experiments.md`.
GitHub branch: **`cursor/rank1-gpu-launch-1252`** (PR #5).

**⚠ Do not commit Kaggle tokens to git.** Paste token into the new chat or set as a cloud secret `KAGGLE_API_TOKEN`.

---

## TL;DR / immediate next action

1. **GPU weekly quota exhausted** until **2026-09-05** (`kaggle quota`). Used ~33h / 30h.
2. **Rank1 matched-4 gold = KILL lean:** 0.7124 vs v6c 0.7365 (Δ **−0.024**). MCL collapsed to 0.465. Skip fold4 by default.
3. **When quota resets:** run **S01b first** (GPU decode-once 5-fold v6c submit), then competition submit. Queue S02 if LB ≥ 0.690.
4. **S01** (ref **55851760**) **TIMED OUT** on hidden test (no score). Baseline LB still **0.682**.
5. **Adopted:** v6c labels, frozen DINOv2-B, cache_v1. **Killed / kill-lean:** v8, GCE, yunus, label micro-tune, **rank1 plane+pw**.

```bash
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_API_TOKEN="…"   # chat paste / secret — rotate after use
mkdir -p ~/.kaggle && printf '%s' "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

kaggle quota
kaggle competitions submissions rsna-knee-abnormality-detection -v

# After 2026-09-05 — S01b FIRST (not fold4):
python3 scripts/push_kaggle_kernels.py submit-v6c-5fold-metadata.json
# when COMPLETE:
kaggle competitions submit rsna-knee-abnormality-detection \
  -k girishbose/submit-v6c-5fold -m "S01b: 5-fold v6c GPU decode-once"
# If LB ≥ 0.690 → queue S02 (submit-v6c-5fold-s02)
```

---

## Calibrated scores

| Ruler | Score | Notes |
|---|---|---|
| **Public LB (fold0 probe)** | **0.682** | ref **55818692** |
| **S01 5-fold** | **TIMEOUT / no score** | ref **55851760** |
| **Rank1 matched-4 gold** | **0.7124** | vs v6c **0.7365** → **KILL lean** |
| **Full-58 gold OOF v6c** | **0.7023** | adopted; keep bar **0.7073** |
| Public LB top | ~0.94–0.95 | gap ~0.26 |

---

## Rank1 weak-val (smoke only; superseded by gold)

| Fold | rank1 | v6c | Δ | Status |
|---|---|---|---|---|
| 0 | 0.7558 | 0.7683 | −0.0125 | COMPLETE |
| 1 | 0.7640 | 0.7493 | +0.0147 | COMPLETE |
| 2 | 0.7706 | 0.7624 | +0.0082 | COMPLETE |
| 3 | 0.7606 | 0.7636 | −0.0030 | COMPLETE |
| 4 | — | 0.721 | — | **skip by default** |

Audit: `docs/audit/rank1_matched4_keepkill.json`.

---

## Next 3 actions

1. **2026-09-05:** S01b GPU submit + competition submit; record LB.
2. If S01b ≥ 0.690 → S02 per-label AUC blend.
3. New image lever only after S01b LB read (cache_v3 not yet justified).

## Do NOT

- Final below ~0.80 projected LB; burn fold4 before S01b; retry killed exps.
- Trust weak-val / 2-fold gold alone; reports at test time; commit secrets/weights/DICOMs.
