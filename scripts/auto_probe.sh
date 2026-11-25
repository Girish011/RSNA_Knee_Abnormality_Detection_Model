#!/bin/zsh
# Push the two own-ensemble probe kernels as soon as a Kaggle GPU slot is free, check their logs, submit them.
# Started 6 Oct 23:20 IST (user: Kaggle activities need no permission). Log: outputs/own_probe_auto.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/own_probe_auto.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
one() {
  name=$1; why=$2; need=$3
  n=0
  until kaggle kernels push -p outputs/kernels/$name 2>&1 | grep -q "successfully pushed"; do
    n=$((n + 1)); [ $n -gt 400 ] && { say "$name: gave up pushing"; return 1 }
    sleep 90
  done
  say "$name: pushed"
  while true; do
    st=$(kaggle kernels status girishbose/$name 2>&1 | tail -1)
    case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) say "$name: kernel failed ($st)"; return 1;; esac
    sleep 60
  done
  kaggle kernels logs girishbose/$name > outputs/kernels/$name/run.log 2>/dev/null
  if ! grep -q "leg exit 0" outputs/kernels/$name/run.log || ! grep -q "submission.csv (3, 13)" outputs/kernels/$name/run.log; then
    say "$name: log check failed (no 'leg exit 0' / '(3, 13)') - NOT submitted"; return 1
  fi
  if [ -n "$need" ] && ! grep -q "$need" outputs/kernels/$name/run.log; then
    say "$name: log check failed (no '$need') - NOT submitted"; return 1
  fi
  say "$name: log ok ($(grep -c 'loaded /kaggle' outputs/kernels/$name/run.log) members loaded); submitting"
  python scripts/submit_gate.py submit --kernel outputs/kernels/$name --version 1 --why "$why" --i-approve >> $LOG 2>&1
  say "$name: submit_gate exit $?"
}
one own20-alone "LEARNING PROBE (user-approved 6 Oct): our own twenty-model ensemble ALONE, no public stack (5 folds each of c224hvy, EffNetV2-S, ConvNeXt-S, ConvNeXt-S second-round; rank mean, leg v2). Reads the own system's leaderboard score (blends with the stack are flat 0.944 from w 0.3 to 0.6, so ~0.94-0.944 expected). Cannot beat 0.945; reference for every later own-model change" ""
one own20-alone-dc1 "LEARNING PROBE (user-approved 6 Oct): own twenty-model ensemble alone + the D1 decorrelation (Lateral OA - 0.3 x Medial OA, PF OA - 0.3 x mean(Medial, Lateral OA), Synovitis - 0.3 x Effusion, Lateral Meniscus - 0.2 x Medial Meniscus). On the stack D1 read 0.944 and D2 0.939 vs 0.945; the 58 experts read +0.006 on these own models. vs own20-alone: is the stack different from our models, or are the 58 experts unlike the test?" "decorrelated 4 columns"
say "done"
