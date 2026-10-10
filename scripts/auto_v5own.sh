#!/bin/zsh
# 10 Oct: the 0.950 public notebook (versia-5) + our four ConvNeXt-S r1 models as one more reader, rank weight 0.10
# and 0.20 on all findings. Waits for the kernels, checks the logs, submits at the slot reset. Log: outputs/auto_cx.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_cx.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
ok=()
for name in v5-own-w10 v5-own-w20; do
  d=outputs/kernels/$name
  while true; do
    st=$(kaggle kernels status girishbose/$name 2>&1 | tail -1)
    case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) say "$name: kernel failed ($st)"; continue 2;; esac
    sleep 60
  done
  kaggle kernels logs girishbose/$name > $d/run.log 2>/dev/null
  for pat in "Apex Fusion SUCCESS" "ours 3/3\] blended" "submission.csv ok (3, 13)"; do
    grep -q -- "$pat" $d/run.log || { say "$name: log check failed (no '$pat') - NOT submitted"; continue 2 }
  done
  grep -q "NOT blended\|Apex Fusion FALLBACK" $d/run.log && { say "$name: a fallback fired - NOT submitted"; continue }
  say "$name: kernel ok"
  ok+=($name)
done
for name in $ok; do
  w=${name##*w}; k=0
  while true; do
    out=$(python scripts/submit_gate.py submit --kernel outputs/kernels/$name --version 1 --i-approve --why "CANDIDATE: the 0.950 public notebook + our four ConvNeXt-S (0.938 alone) as one more reader at rank weight 0.$w on all findings. Odds of > 0.950: ~1 in 4" 2>&1)
    [ $? -eq 0 ] && { say "$name: submitted | $(echo $out | tr '\n' ' ' | cut -c1-90)"; break }
    case "$out" in *"daily cap"*) k=$((k + 1)); [ $k -gt 250 ] && { say "$name: no slot in 21 h"; break }; sleep 300;; *) say "$name: submit refused | $(echo $out | tr '\n' ' ' | cut -c1-200)"; break;; esac
  done
done
