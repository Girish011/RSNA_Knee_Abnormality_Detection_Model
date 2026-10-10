#!/bin/zsh
# 9 Oct: the public ConvNeXt reader's three folds ALONE (0.950 notebook with the reader weight set to 1.0).
# Waits for the kernel, checks the log, submits at the slot reset (05:30 IST). Log: outputs/auto_cx.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_cx.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
name=pub-versia-5-reader-alone; d=outputs/kernels/$name
while true; do
  st=$(kaggle kernels status girishbose/$name 2>&1 | tail -1)
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) say "$name: kernel failed ($st)"; exit 1;; esac
  sleep 60
done
kaggle kernels logs girishbose/$name > $d/run.log 2>/dev/null
grep -q "Apex Fusion SUCCESS" $d/run.log || { say "$name: no 'Apex Fusion SUCCESS' in the log - NOT submitted"; exit 1 }
grep -q "Apex Fusion FALLBACK\|Apex Fusion EXCEPTION" $d/run.log && { say "$name: fusion fell back - NOT submitted"; exit 1 }
rm -rf $d/_out; mkdir -p $d/_out; kaggle kernels output girishbose/$name -p $d/_out > /dev/null 2>&1
shape=$(python -c "import pandas as pd; s = pd.read_csv('$d/_out/submission.csv'); print(s.shape[1], int(s.notna().all().all()))" 2>/dev/null)
[ "$shape" = "13 1" ] || { say "$name: bad submission.csv ($shape) - NOT submitted"; exit 1 }
say "$name: kernel ok"
k=0
while true; do
  out=$(python scripts/submit_gate.py submit --kernel $d --version 1 --i-approve --why "LEARNING PROBE: the public 2.5D ConvNeXt reader ALONE (its three public folds; the 0.950 notebook with the reader weight set to 1.0). How strong is the model we are rebuilding? Cannot beat 0.950" 2>&1)
  [ $? -eq 0 ] && { say "$name: submitted | $(echo $out | tr '\n' ' ' | cut -c1-90)"; break }
  case "$out" in *"daily cap"*) k=$((k + 1)); [ $k -gt 250 ] && { say "$name: no slot in 21 h"; break }; sleep 300;; *) say "$name: submit refused | $(echo $out | tr '\n' ' ' | cut -c1-200)"; break;; esac
done
