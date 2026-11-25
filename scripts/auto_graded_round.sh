#!/bin/zsh
# After the graded LLM pass completes: build labels_g1 and launch four ConvNeXt-S students on single RunPod pods
# with the balance that is left (user, 6 Oct: use up the existing money, then RunPod is closed).
# Started 6 Oct 23:40 IST. Log: outputs/auto_graded_round.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_graded_round.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
say "waiting for the graded pass (outputs/graded_labels_run2.log)"
n=0
until grep -q "^graded [0-9]*/4407 reports" outputs/graded_labels_run2.log 2>/dev/null; do
  n=$((n + 1)); [ $n -gt 240 ] && { say "graded pass did not finish in 4 h - stop"; exit 1 }
  sleep 60
done
say "graded pass: $(grep '^graded [0-9]*/4407 reports' outputs/graded_labels_run2.log)"
rows=$(python -c "import pandas as pd; print(len(pd.read_parquet('data/processed/labels_graded_v1.parquet')))" 2>/dev/null)
[ "${rows:-0}" -ge 4380 ] || { say "labels_graded_v1.parquet has ${rows:-0} rows (< 4380) - stop"; exit 1 }
python scripts/make_graded_round.py --out labels_g1.parquet --a 0.4 --decor none \
  --family 'outputs/runpod_results/r2-c224hvy40-f*/*' --family 'outputs/runpod_results/r2-effv2s336-f*/*' \
  --family 'outputs/runpod_results/r2-cnxs336-f*/*' --family 'outputs/runpod_results/r4-cnxs336-f*/*' > outputs/make_graded_round_g1.log 2>&1
rc=$?
say "make_graded_round exit $rc | $(grep '^macro' outputs/make_graded_round_g1.log)"
[ $rc -eq 0 ] && [ -f data/processed/labels_g1.parquet ] || { say "labels_g1 not built - stop"; exit 1 }
tgt=$(grep '^macro' outputs/make_graded_round_g1.log | sed "s/.*'target': \([0-9.]*\).*/\1/")
python -c "import sys; sys.exit(0 if float('${tgt:-0}') >= 0.94 else 1)" || { say "target on the experts ${tgt} < 0.94 - stop, look at outputs/make_graded_round_g1.log"; exit 1 }
bal=$(python -c "from rsna_knee.agent import runpod; print(runpod.balance()['clientBalance'])" 2>/dev/null)
say "RunPod balance ${bal}"
folds=$(python -c "b=float('${bal:-0}'); print('0,1,2,3' if b >= 12 else '0,1,2' if b >= 9.5 else '')")
[ -n "$folds" ] || { say "balance too low for a student wave - stop"; exit 1 }
say "launching round 8 (labels_g1, cnxs336) folds $folds"
python scripts/launch_round.py --round 8 --labels labels_g1.parquet --families cnxs336 --folds $folds >> $LOG 2>&1
say "launch_round exit $?"
