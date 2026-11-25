#!/bin/zsh
# Leaderboard A/B of the label files, own ConvNeXt-S models alone (no stack), folds 0-3, same recipe:
#   cnxs-r1-f03 (labels_r1), cnxs-r4-f03 (labels_r4) now; cnxs-g1-f03 (graded labels_g1) and cnxs-g1r4-f03
#   (graded + labels_r4, 8 models) once the round-8 pods are collected and their result datasets are on Kaggle.
# Runs by itself (session wake-ups are not reliable). Started 7 Oct 00:45 IST. Log: outputs/auto_ab.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_ab.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }

run_kernel() {   # push when a GPU slot is free, wait for COMPLETE, check the log
  name=$1; nmem=$2
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
    grep -q "leg exit" outputs/kernels/$name/run.log && break
    sleep 45
  done
  if ! grep -q "leg exit 0" outputs/kernels/$name/run.log || ! grep -q "submission.csv (3, 13)" outputs/kernels/$name/run.log; then
    say "$name: log check failed (no 'leg exit 0' / '(3, 13)') - NOT submitted"; return 1
  fi
  got=$(grep -c 'loaded /kaggle' outputs/kernels/$name/run.log)
  [ "$got" -eq "$nmem" ] || { say "$name: $got members loaded, expected $nmem - NOT submitted"; return 1 }
  say "$name: log ok ($got members loaded)"
}

submit() {   # waits for a free daily slot (the gate refuses at 5 a day, UTC)
  name=$1; why=$2
  n=0
  while true; do
    out=$(python scripts/submit_gate.py submit --kernel outputs/kernels/$name --version 1 --why "$why" --i-approve 2>&1)
    rc=$?
    [ $rc -eq 0 ] && { say "$name: submitted | $(echo $out | tr '\n' ' ' | cut -c1-160)"; return 0 }
    case "$out" in
      *"daily cap"*) n=$((n + 1)); [ $n -gt 150 ] && { say "$name: no daily slot in 12 h - stop"; return 1 }; sleep 300;;
      *) say "$name: submit refused / failed | $(echo $out | tr '\n' ' ' | cut -c1-300)"; return 1;;
    esac
  done
}

# --- phase 1: the two families that exist. The 6 Oct probes (auto_probe.sh) go first on the GPU slot.
n=0
until grep -qE "own20-alone-dc1: (pushed|gave up)| done$" outputs/own_probe_auto.log 2>/dev/null; do
  n=$((n + 1)); [ $n -gt 180 ] && break
  sleep 60
done
say "phase 1: label A/B arms r1, r4"
run_kernel cnxs-r1-f03 4 && ok1=1
run_kernel cnxs-r4-f03 4 && ok4=1
[ -n "$ok1" ] && submit cnxs-r1-f03 "LEARNING PROBE label A/B 1/4: own ConvNeXt-S folds 0-3 ALONE (no stack), first-round labels. Reference arm for the graded labels; odds of > 0.945 under 1 in 20"
[ -n "$ok4" ] && submit cnxs-r4-f03 "LEARNING PROBE label A/B 2/4: own ConvNeXt-S folds 0-3 ALONE (no stack), second-round labels. Twin of the graded arm 3/4; odds of > 0.945 under 1 in 20"

# --- phase 2: the graded students (round 8, pods launched 00:27-00:32 IST, ~5 h)
say "phase 2: waiting for the round-8 students"
n=0
while true; do
  st=$(python scripts/collect_round.py --wait-min 0 2>/dev/null | tail -1)
  got=$(echo "$st" | python -c "
import sys, json
s = sys.stdin.read(); d = json.loads(s[s.index('{'):])
print(sum(d.get(f'r8-cnxs336-f{k}', 'collected') == 'collected' for k in range(4)), '|', {k: v for k, v in d.items() if k.startswith('r8')})" 2>/dev/null)
  case "$got" in "4 |"*) break;; esac
  n=$((n + 1)); [ $n -gt 54 ] && { say "round 8 not collected in 9 h ($got) - stop"; exit 1 }
  sleep 600
done
say "round 8 collected | $(python -c "
import json
print({k: round(json.load(open(f'outputs/runpod_results/r8-cnxs336-f{k}/r8cnxs336{k}/raptor_ft_r8cnxs336{k}.json'))['best_gold_auc'], 4) for k in range(4)})" 2>&1)"
python -c "
import json, sys
g = [json.load(open(f'outputs/runpod_results/r8-cnxs336-f{k}/r8cnxs336{k}/raptor_ft_r8cnxs336{k}.json'))['best_gold_auc'] for k in range(4)]
sys.exit(0 if min(g) >= 0.88 else 1)" || { say "a round-8 student reads < 0.88 on its gold-29 - stop, look at it"; exit 1 }
n=0
for k in 0 1 2 3; do   # the checkpoint must be in the result dataset before a kernel can mount it
  until kaggle datasets files girishbose/rsna-knee-rp-r8-cnxs336-f$k-out 2>/dev/null | grep -q "raptor_ft_r8cnxs336$k.pt"; do
    n=$((n + 1)); [ $n -gt 40 ] && { say "r8 fold $k: checkpoint not in its Kaggle dataset after 200 min - stop"; exit 1 }
    sleep 300
  done
done
say "round-8 checkpoints are on Kaggle"
R8=r8-cnxs336-f0:r8cnxs3360,r8-cnxs336-f1:r8cnxs3361,r8-cnxs336-f2:r8cnxs3362,r8-cnxs336-f3:r8cnxs3363
R4=r4-cnxs336-f0:r4cnxs3360,r4-cnxs336-f1:r4cnxs3361,r4-cnxs336-f2:r4cnxs3362,r4-cnxs336-f3:r4cnxs3363
python scripts/build_own_probe.py --name cnxs-g1-f03 --members $R8 >> $LOG 2>&1
python scripts/build_own_probe.py --name cnxs-g1r4-f03 --members $R8,$R4 >> $LOG 2>&1
run_kernel cnxs-g1-f03 4 && okg=1
run_kernel cnxs-g1r4-f03 8 && okb=1
[ -n "$okg" ] && submit cnxs-g1-f03 "LEARNING PROBE label A/B 3/4: own ConvNeXt-S folds 0-3 ALONE (no stack), severity-graded report labels. vs 2/4 = the label change on the leaderboard; odds of > 0.945 under 1 in 20"
[ -n "$okb" ] && submit cnxs-g1r4-f03 "LEARNING PROBE label A/B 4/4: graded-label + second-round ConvNeXt-S, folds 0-3 each (8 models), ALONE. Do the two label families add; odds of > 0.945 under 1 in 20"
say "done"
