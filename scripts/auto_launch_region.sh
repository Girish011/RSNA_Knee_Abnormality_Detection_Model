#!/bin/zsh
# 12 Oct: launch the region specialists as soon as the RunPod balance is in (user: $100 on the evening of 12 Oct).
# Polls the balance every 10 min for 20 h. >= $30: the eight planned runs (tf, pf, rec, recsf x folds 0, 1; ~$18) and
# the four runs with the report severity grade as a side target (tfg, pfg x folds 0, 1; ~$9). $20-30: the planned
# eight only. launch_region.py skips jobs already in agent/runs/region_jobs.json, so a failed launch is retried
# (3 times) without doubling a pod. Log: outputs/auto_launch_region.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_launch_region.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
say "watching the RunPod balance"
tries=0
for k in {1..120}; do
  bal=$(python -c "from rsna_knee.agent import runpod; print(f\"{float(runpod.balance()['clientBalance']):.2f}\")" 2>/dev/null)
  if [ -z "$bal" ]; then say "balance not readable"; sleep 600; continue; fi
  if (( bal >= 30 )); then specs=tf,pf,rec,recsf,tfg,pfg
  elif (( bal >= 20 )); then specs=tf,pf,rec,recsf
  else sleep 600; continue; fi
  say "balance \$$bal -> launching $specs"
  python scripts/launch_region.py --labels labels_t2.parquet --specs $specs --folds 0,1 >> $LOG 2>&1
  rc=$?
  n=$(python -c "import json;print(len(json.load(open('agent/runs/region_jobs.json'))))" 2>/dev/null)
  say "launch exit $rc | jobs in the log: $n"
  [ $rc -eq 0 ] && break
  tries=$((tries + 1)); [ $tries -ge 3 ] && { say "three failed launches - stopped"; break }
  sleep 600
done
say "watcher done"
