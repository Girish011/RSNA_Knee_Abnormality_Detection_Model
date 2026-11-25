#!/bin/zsh
# 9 Oct: fork the top public notebooks unchanged (user's go 02:1x IST), check each run wrote a 13-column
# submission.csv, submit when a daily slot is free (reset 05:30 IST). Log: outputs/auto_fork.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_fork.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
ready=()
for n in helios-top rsna-versia-7 rsna-knee-template rsna-knee-blend-gold; do
  d=outputs/kernels/pub/$n
  id=$(python -c "import json; print(json.load(open('$d/kernel-metadata.json'))['id'])")
  st0=$(kaggle kernels status $id 2>&1 | tail -1)
  case "$st0" in
    *RUNNING*|*COMPLETE*|*QUEUED*) say "$n: already on Kaggle as $id, not pushed again";;
    *) k=0
       until kaggle kernels push -p $d 2>&1 | grep -q "successfully pushed"; do
         k=$((k + 1)); [ $k -gt 200 ] && { say "$n: gave up pushing"; continue 2 }
         sleep 90
       done
       say "$n: pushed as $id";;
  esac
  sleep 60
  while true; do
    st=$(kaggle kernels status $id 2>&1 | tail -1)
    case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) say "$n: kernel failed ($st)"; continue 2;; esac
    sleep 60
  done
  rm -rf $d/_out; mkdir -p $d/_out
  kaggle kernels output $id -p $d/_out > /dev/null 2>&1
  shape=$(python -c "import pandas as pd; s = pd.read_csv('$d/_out/submission.csv'); print(s.shape[1], int(s.notna().all().all()))" 2>/dev/null)
  [ "$shape" = "13 1" ] || { say "$n: no valid submission.csv in the output ($shape) - NOT submitted"; continue }
  say "$n: COMPLETE, submission.csv ok"
  ready+=($n)
done
for n in $ready; do
  d=outputs/kernels/pub/$n; k=0
  while true; do
    out=$(python scripts/submit_gate.py submit --kernel $d --version 1 --i-approve --why "FORK, unchanged, of the public notebook $n: reads the newest public level on our account (our best 0.950 = the 8 Oct public notebooks). Expected 0.951-0.954" 2>&1)
    [ $? -eq 0 ] && { say "$n: submitted | $(echo $out | tr '\n' ' ' | cut -c1-100)"; break }
    case "$out" in *"daily cap"*) k=$((k + 1)); [ $k -gt 150 ] && { say "$n: no slot in 12 h"; break }; sleep 300;; *) say "$n: submit refused | $(echo $out | tr '\n' ' ' | cut -c1-200)"; break;; esac
  done
done
say "done"
