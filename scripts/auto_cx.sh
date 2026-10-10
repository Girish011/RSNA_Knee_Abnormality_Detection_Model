#!/bin/zsh
# 9 Oct: public ConvNeXt-reader formula on our account. Pushes the sixth cache shard when a CPU session frees, waits
# for all six, then pushes the training kernel cx-train-a (folds 3 and 4, their train.py). Log: outputs/auto_cx.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_cx.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
n=0
until kaggle kernels push -p outputs/kernels/cx-cache-s5 2>&1 | grep -q "successfully pushed"; do
  n=$((n + 1)); [ $n -gt 240 ] && { say "s5: gave up pushing"; exit 1 }
  sleep 60
done
say "cx-cache-s5 pushed"
for k in 0 1 2 3 4 5; do
  while true; do
    st=$(kaggle kernels status girishbose/cx-cache-s$k 2>&1 | tail -1)
    case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) say "cx-cache-s$k failed ($st) - stop"; exit 1;; esac
    sleep 60
  done
  say "cx-cache-s$k COMPLETE | $(kaggle kernels logs girishbose/cx-cache-s$k 2>/dev/null | grep -o 'done [0-9]* series[^"\\]*' | tail -1)"
done
n=0
until kaggle kernels push -p outputs/kernels/cx-train-a 2>&1 | grep -q "successfully pushed"; do
  n=$((n + 1)); [ $n -gt 120 ] && { say "cx-train-a: gave up pushing"; exit 1 }
  sleep 90
done
say "cx-train-a pushed (folds 3, 4 of the public ConvNeXt reader recipe; T4 x 2)"
