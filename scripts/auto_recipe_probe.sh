#!/bin/zsh
# 9 Oct: own ConvNeXt-S r1 folds 0-3 scored the way the 0.950 public notebook scores (every window position,
# mirror on every second pass): waits for the kernel, checks the log, submits when a slot is free. Log: outputs/auto_fork.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_fork.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
name=cnxs-r1-allwin-mirror
while true; do
  st=$(kaggle kernels status girishbose/$name 2>&1 | tail -1)
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) say "$name: kernel failed ($st)"; exit 1;; esac
  sleep 60
done
for try in 1 2 3 4; do
  kaggle kernels logs girishbose/$name > outputs/kernels/$name/run.log 2>/dev/null
  grep -q "leg exit" outputs/kernels/$name/run.log && break
  sleep 45
done
for pat in "leg exit 0" "submission.csv (3, 13)" "k_eval=39 passes=4 flip=True"; do
  grep -q -- "$pat" outputs/kernels/$name/run.log || { say "$name: log check failed (no '$pat') - NOT submitted"; exit 1 }
done
say "$name: log ok | $(grep -o 'timing | wall [0-9.]* s/study' outputs/kernels/$name/run.log | head -1)"
k=0
while true; do
  out=$(python scripts/submit_gate.py submit --kernel outputs/kernels/$name --version 1 --i-approve --why "LEARNING PROBE: own ConvNeXt-S folds 0-3 alone (0.937) scored like the 0.950 public notebook: every window position (39 x 4 passes) + mirror on every second pass. Odds of > 0.945: 1 in 20" 2>&1)
  [ $? -eq 0 ] && { say "$name: submitted | $(echo $out | tr '\n' ' ' | cut -c1-100)"; break }
  case "$out" in *"daily cap"*) k=$((k + 1)); [ $k -gt 150 ] && { say "$name: no slot in 12 h"; break }; sleep 300;; *) say "$name: submit refused | $(echo $out | tr '\n' ' ' | cut -c1-200)"; break;; esac
done
