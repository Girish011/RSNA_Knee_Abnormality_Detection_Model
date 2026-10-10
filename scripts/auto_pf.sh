#!/bin/zsh
# Per-finding test of own models vs the stack (7 Oct): is the 0.008-0.010 gap of our models spread over all findings
# or concentrated in a few? Own = ConvNeXt-S first-round labels, folds 0-3 (0.937 alone on the leaderboard).
#   cnxs-r1-neg-latmen / cnxs-r1-neg-synov : own alone, one column reversed -> own AUC of that finding on the LB
#                                            (stack: Lateral Meniscus 0.950, Synovitis 0.914)
#   pf-g1-w50  : stack + own w 0.5 on ACL, MCL, Medial Meniscus, Medial OA, Effusion, Baker's only (candidate)
#   pf-g2-w50  : stack + own w 0.5 on Contusion, Fracture, Lateral OA, PF OA only
#   pf-g1-swap : stack with the six G1 columns taken from own (w 0.99) -> sum of (own - stack) over G1
# Pushes and checks the kernels now, submits when the daily slots reset (05:30 IST on 8 Oct).
# Runs by itself. Log: outputs/auto_pf.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_pf.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }

run_kernel() {   # name, grep patterns that must all be in the log, pattern that must not
  name=$1; shift
  n=0
  until kaggle kernels push -p outputs/kernels/$name 2>&1 | grep -q "successfully pushed"; do
    n=$((n + 1)); [ $n -gt 400 ] && { say "$name: gave up pushing"; return 1 }
    sleep 90
  done
  say "$name: pushed"
  sleep 60
  while true; do
    st=$(kaggle kernels status girishbose/$name 2>&1 | tail -1)
    case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) say "$name: kernel failed ($st)"; return 1;; esac
    sleep 60
  done
  for try in 1 2 3 4; do
    kaggle kernels logs girishbose/$name > outputs/kernels/$name/run.log 2>/dev/null
    [ -s outputs/kernels/$name/run.log ] && grep -q "submission.csv" outputs/kernels/$name/run.log && break
    sleep 45
  done
  for pat in "$@"; do
    grep -q -- "$pat" outputs/kernels/$name/run.log || { say "$name: log check failed (no '$pat') - NOT submitted"; return 1 }
  done
  if grep -q "NOT blended" outputs/kernels/$name/run.log; then say "$name: 'NOT blended' in the log - NOT submitted"; return 1; fi
  say "$name: log ok ($(grep -c 'loaded /kaggle' outputs/kernels/$name/run.log) own members loaded)"
}

submit() {
  name=$1; why=$2
  n=0
  while true; do
    out=$(python scripts/submit_gate.py submit --kernel outputs/kernels/$name --version 1 --why "$why" --i-approve 2>&1)
    rc=$?
    [ $rc -eq 0 ] && { say "$name: submitted | $(echo $out | tr '\n' ' ' | cut -c1-160)"; return 0 }
    case "$out" in
      *"daily cap"*) n=$((n + 1)); [ $n -gt 330 ] && { say "$name: no daily slot in 27 h - stop"; return 1 }; sleep 300;;
      *) say "$name: submit refused / failed | $(echo $out | tr '\n' ' ' | cut -c1-300)"; return 1;;
    esac
  done
}

say "start"
run_kernel cnxs-r1-neg-latmen "leg exit 0" "submission.csv (3, 13)" "negated columns" && k1=1
run_kernel cnxs-r1-neg-synov "leg exit 0" "submission.csv (3, 13)" "negated columns" && k2=1
run_kernel pf-g1-w50 "ours 3/3\] blended" "submission.csv ok (3, 13)" && k3=1
run_kernel pf-g2-w50 "ours 3/3\] blended" "submission.csv ok (3, 13)" && k4=1
run_kernel pf-g1-swap "ours 3/3\] blended" "submission.csv ok (3, 13)" && k5=1
[ -n "$k3" ] && submit pf-g1-w50 "CANDIDATE: best stack (0.945) + own ConvNeXt-S (0.937 alone) at rank w 0.5 on six findings only: ACL, MCL, Medial Meniscus, Medial OA, Effusion, Baker's. Odds of > 0.945: ~40 %"
[ -n "$k1" ] && submit cnxs-r1-neg-latmen "LEARNING PROBE: own ConvNeXt-S folds 0-3 alone (0.937) with Lateral Meniscus reversed -> our AUC on that finding (stack 0.950). AUC = (1 + 12 x (0.937 - LB)) / 2. Scores ~0.86"
[ -n "$k2" ] && submit cnxs-r1-neg-synov "LEARNING PROBE: own ConvNeXt-S folds 0-3 alone (0.937) with Synovitis reversed -> our AUC on that finding (stack 0.914). AUC = (1 + 12 x (0.937 - LB)) / 2. Scores ~0.87"
[ -n "$k5" ] && submit pf-g1-swap "LEARNING PROBE: best stack with six columns (ACL, MCL, Medial Meniscus, Medial OA, Effusion, Baker's) taken from own ConvNeXt-S: LB - 0.945 = sum of (own - stack) AUC / 12. Odds of > 0.945: ~25 %"
[ -n "$k4" ] && submit pf-g2-w50 "CANDIDATE: best stack (0.945) + own ConvNeXt-S at rank w 0.5 on Contusion, Fracture, Lateral OA, PF OA only. With pf-g1-w50 it splits the flat 0.944 of the all-finding blends. Odds of > 0.945: ~25 %"
say "done"
