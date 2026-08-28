# Label cross-check + v8 build (2026-08-27)

## Public LB calibration (confirmed via API)
- `girishbose/rsna-knee-submit-v6c` → submission `55818692` → **publicScore 0.682**
- Rank ~1999 / 2516 (live snapshot). Do not select as final.

## In-domain cross-check (v6c vs public soft-label sets)
Soft labels hardened with `--soft-lo 0.2 --soft-hi 0.7` (mid band abstains).

| Source | Agree where both commit | Notes |
|---|---|---|
| **yunusgmsoy** | **91.2%** (15954/17494) | Strongest external agree; Effusion 97%, MedOA 93%, ACL 83% |
| **dreaddevelopment** | 80.9% (16934/20944) | Weak on **MCL 38%**, LatOA 49%, Effusion 67%; we are more positive (3395 we1 vs 615 they1) |
| **barun2104** `pseudo_*` | n/a | Almost all mass at 0.5 → abstain band; hard cols = 58 gold only (agree 100% because v6c expert-overrides gold) |

Per-label hot spots (yunus disagreements we care about):
- We over-call: Lateral Meniscus, Medial Meniscus, Synovitis, Baker's
- They over-call vs us: **PF OA**, **ACL**

`barun` is not a useful corpus-wide teacher under this hardening.

## v8 = v6c ⊕ (v7 ∩ Qwen) on TR/EL, **additive gap-fill**
- Languages: other 3540 / tr 546 / el 321
- v7∩Qwen agree **85.1%** (2593/3048) on cells both commit
- Overlay: only where v6c is NaN (safe). Net **+495 known cells** (23526 → 24021), almost all ACL/MCL/Lateral OA on TR/EL — exactly the coin-flip columns v6c dropped, re-supplied only on consensus.
  - ACL +108 (11 pos / 97 neg)
  - MCL +120 (29 pos / 91 neg)
  - Lateral OA +267 (74 pos / 193 neg)
- 58-gold TR/EL only **9 studies** → cannot validate model impact from gold alone; need full-58 OOF after retrain.
- Consensus-vs-gold on tiny n=35 committed cells: macro prec ~0.72 (MCL still noisy at n=2). Not a keep signal by itself.

## Artifacts
- Kaggle dataset: `girishbose/rsna-knee-weak-v8` (`weak_labels_v8_candidate.csv`)
- Local (gitignored data): `data/processed/weak_labels_{v6c,v7,v8}.csv`
- Agree tables: `docs/audit/label_crosscheck_v6c/agree_ours_vs_*.csv`, `docs/audit/v8_v7_qwen_agree.csv`

## Next measure
Frozen-B 5-fold on v8 → full-58 gold OOF vs **0.7023** (keep if ≥ +0.005). Projected LB ≈ OOF − 0.02. No further public submit until that clears.
