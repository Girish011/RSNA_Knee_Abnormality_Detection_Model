# Handoff — 2026-08-28 (paste into new chat)

Read first: `docs/STATUS.md`, `docs/DECISIONS.md`, tail of `docs/experiments.md`.
GitHub branch with latest work: **`cursor/rank1-v6c-57b3`** (PR #4).

---

## TL;DR / immediate next action

1. **Check S01 public LB** — ref **55851760**. Baseline fold0-only = **0.682**. Expected **~0.70–0.72**. Kill if **< 0.690**; queue S02 if ≥ 0.690.
2. **GPU quota restored** — push rank1: refresh `rsna-knee-rank1-patch`, then `python3 scripts/push_kaggle_kernels.py train-b-fold{0,1}-rank1-v6c-metadata.json` and start both on GPU.
3. **Cloud blocker:** this env has **no `KAGGLE_API_TOKEN`**. Add secret or paste score + token to continue.
4. **Adopted:** v6c labels, frozen DINOv2-B, cache_v1. **Killed:** v8, GCE (null), yunus gap-fill.
5. Branch with launch tooling: **`cursor/rank1-gpu-launch-1252`** (PR #5).

---

## Kaggle access

- **User:** `girishbose`
- **CLI:** `~/.local/bin/kaggle` (not on default PATH)
- **Auth:** token in `~/.kaggle/access_token` (chmod 600) **OR** `export KAGGLE_API_TOKEN=<token>`
- **Competition slug:** `rsna-knee-abnormality-detection`
- **⚠ SECURITY:** Token was pasted in chat sessions — **rotate at kaggle.com/settings** after handoff. **Do not commit token to git.**

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
export PATH="$HOME/.local/bin:$PATH"
```

---

## Calibrated scores (source of truth)

| Ruler | Score | Notes |
|---|---|---|
| **Public LB (fold0 probe)** | **0.682** | submission ref **55818692**, `rsna-knee-submit-v6c` v1 |
| **Public LB (S01 5-fold)** | *pending* | ref **55851760**, PENDING 2026-08-28 |
| **Full-58 gold OOF v6c** | **0.7023** | 5-fold, adopted baseline |
| Full-58 gold OOF v8 | 0.6935 | **KILL** (Δ −0.0088) |
| Weak-val v6c BCE (fold0/1) | 0.7683 / 0.7493 | confounded; do not use for keep/kill |
| Public LB top | ~0.942 | gap ~0.26 |

**Calibration:** gold-OOF ≈ public LB **+ 0.02**. Weak-label OOF ~0.75 overstates by ~0.07.

**Keep/kill rule (trains):** full-58 gold vs v6c **0.7023**, margin **0.005**.

---

## Current model recipe (production)

- **Labels:** `girishbose/rsna-knee-weak-v6c` (`weak_labels_v6c_candidate.csv`)
- **Backbone:** frozen DINOv2-B, 8ep, never unfreeze, `dinov2_vitb14_pretrain.pth`
- **Cache:** `girishbose/rsna-knee-cache-v1` → `cache_v1` (3 series × 12 slices × 224)
- **Checkpoints:** `girishbose/train-b-fold{0..4}-weak-v6c` → `fold{F}_best.pt`, `fold{F}_oof.csv`
- **Code:** `girishbose/rsna-knee-code` (Kaggle) — **GitHub is stale mirror**; fold new modules into Kaggle dataset before train

---

## Submit queue (hypothesis-driven)

| ID | Hypothesis | Status | Kill if |
|---|---|---|---|
| **S01** | 5-fold uniform mean >> fold0 | SUBMITTED (check LB) | LB < 0.690 |
| S02 | Per-label AUC-weighted 5-fold blend | queued | ≤ S01 |
| S03 | Best-3-fold subset | queued | < S01 − 0.003 |
| S05+ | rank1 plane routing (needs GPU train) | blocked quota | gold < 0.7073 |

**Submit command:**
```bash
kaggle competitions submit rsna-knee-abnormality-detection \
  -k girishbose/submit-v6c-5fold -m "S01: 5-fold v6c uniform blend" -v 1
```

**Check score:**
```bash
kaggle competitions submissions rsna-knee-abnormality-detection -v
```

---

## Killed experiments (do NOT retry)

| Experiment | Result | Why |
|---|---|---|
| **v8** (TR/EL consensus gap-fill) | gold 0.6935 | Re-poisoned ACL/MCL/LatOA |
| **GCE on v6c** | identical to BCE | Stale trainer ignored `loss.mode`; patch only shadowed `loss.py` |
| **yunus gap-fill** | +24k mostly-neg cells | strict intersect = 0 cells |
| cache_v2 4×16 | S fold0 0.738 | Below cache_v1 0.764 |
| unfreeze backbone | 0.729→0.615 | Collapse |
| external KneeMRI ACL | AUC 0.53 | Domain shift |
| v6d label micro-tune | ≈ v6c | 2-fold gold noise |

---

## Rank-1 stack (shipped, not yet trained)

Branch `cursor/rank1-v6c-57b3`:

- **Per-label plane routing** — `src/rsna_knee/models/plane_routing.py`, `multiseries.py` (`label_plane_routing=True`)
- **ACL sagittal slice bias** — `dicom.py`, `build_cache.py --sagittal-acl-bias`
- **cache_v3** — 4×16 + ACL bias; kernel `kernels/build-cache-v3.py`
- **Config** — `configs/rank1_v6c.yaml` (frozen-B 12ep, per-label pos_weight)
- **Kaggle patch** — `girishbose/rsna-knee-rank1-patch` (shadows stale code)
- **Train kernels** — `kernels/train-b-fold{0,1}-rank1-v6c.py` (not pushed — GPU quota)
- **Submit** — `kernels/submit-v6c-5fold.py` + `submit-v6c-5fold-metadata.json` (GPU)

---

## Top-team post (0.937 LB) — lessons mapped to us

1. **Elimination > addition at 0.93+** — we are at 0.68; image/ensemble levers still matter for us.
2. **Real bugs can score zero** — GCE lesson; always 5-fold + full-58 gold.
3. **Public LB 3rd decimal is noise at top** — not relevant until ~0.85+.
4. **Never trust 2-fold gold** — pre-register rules; full-58 macro only.
5. **Remaining gap for leaders = noisy teacher** — we already mined labels (v6c); our gap also needs **image signal** (ACL gold ~0.60).

---

## Kaggle kernel gotchas

```json
"machine_shape": "NvidiaTeslaT4",
"docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
```

- Max **2 concurrent GPU** sessions → launch folds in pairs.
- Attach outputs via `kernel_sources`; datasets via `dataset_sources`.
- Empty script kernels → instant COMPLETE, no outputs (happened on GCE v1–v3).
- Frozen-B 8ep ≈ **3.4h/fold**. Submit inference ≈ **6min/study on GPU**, much slower on CPU.
- Fetch: `kaggle kernels output <slug> -p <dir>`; logs are JSON lines.

---

## Key Kaggle assets

| Asset | Slug |
|---|---|
| Code (authoritative) | `girishbose/rsna-knee-code` |
| Labels v6c | `girishbose/rsna-knee-weak-v6c` |
| Rank1 patch | `girishbose/rsna-knee-rank1-patch` |
| Loss GCE patch | `girishbose/rsna-knee-loss-gce` |
| Weights | `girishbose/dinov2-vitb14-rsna-knee` |
| Cache v1 | kernel `girishbose/rsna-knee-cache-v1` |
| Train v6c | `girishbose/train-b-fold{0..4}-weak-v6c` |
| Submit fold0 | `girishbose/rsna-knee-submit-v6c` |
| **Submit 5-fold S01** | `girishbose/submit-v6c-5fold` |

---

## GitHub vs Kaggle sync gap

GitHub has (not all in Kaggle code dataset):

- `fill_policy.py`, `weak_labels_v7.py`, `label_consensus.py`, `evaluation.py`, `oof_report.py`
- Robust losses in `training/loss.py` (GitHub trainer reads `loss.mode`; Kaggle trainer may not)
- Rank1 plane routing, `infer_ensemble.py`, `submit-v6c-5fold.py`

**To train on Kaggle:** upload patch dataset or version `rsna-knee-code` via `scripts/package_kaggle_code.py`.

---

## Next 3 actions

1. **Record S01 public LB** in `docs/experiments.md`; queue S02 if ≥ 0.690.
2. **When GPU quota resets:** push GPU `submit-v6c-5fold` v2 + `train-b-fold{0,1}-rank1-v6c`.
3. **Rotate Kaggle token** (exposed in chat).

---

## Do NOT

- Final anything below ~0.80 projected LB (max-AUC slot).
- Retry v8, GCE-as-shipped, yunus gap-fill, label micro-tuning.
- Trust 2-fold or weak-val for keep/kill.
- Reopen: unfreeze, cache_v2, MRI-CORE, 384, external-image train.
- Use reports at inference. Commit secrets / weights / DICOMs.

---

## Competition facts

- **RSNA 2026 Knee Abnormality Detection**, deadline **2026-10-22**
- Metric: **macro AUC**, 12 binary labels
- Test: **images + series metadata only** (no reports)
- Code competition, ≤9h runtime, internet OFF at submit
- Dual finals: max-AUC + efficiency (separate student track)
