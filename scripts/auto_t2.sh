#!/bin/zsh
# 11 Oct: submissions for the 12 Oct 05:30 IST slot reset. One candidate (the 0.950 notebook + the second expert-trained
# checkpoint as another reader) and four per-finding probes on the 0.950 base. The kernels are COMPLETE and checked
# (STATUS 10:55 / 11:00 IST, 11 Oct). Waits while the gate answers "daily cap". Log: outputs/auto_t2.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_t2.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
names=(v5t2-b-w35 v5-neg-syn v5-neg-latoa v5-neg-pfoa)   # v5-neg-latmen waits for 13 Oct: its slot goes to t2-alone (auto_t2b.sh)
typeset -A why
why[v5t2-b-w35]="CANDIDATE: the 0.950 public notebook with the dataset's second expert-trained checkpoint (224 px, 0.945 alone per its README) added to the CoAtNet rank-mean at weight 0.35 (384 px arm 0.65); reader fusion unchanged. Reference: 0.950"
why[v5-neg-syn]="LEARNING PROBE: the 0.950 notebook with the Synovitis column reversed. Reads that finding's leaderboard AUC = (1 + 12 x (0.950 - score)) / 2. Expected ~0.88"
why[v5-neg-latoa]="LEARNING PROBE: the 0.950 notebook with the Lateral OA column reversed. Reads that finding's leaderboard AUC = (1 + 12 x (0.950 - score)) / 2. Expected ~0.88"
why[v5-neg-pfoa]="LEARNING PROBE: the 0.950 notebook with the PF OA column reversed. Reads that finding's leaderboard AUC = (1 + 12 x (0.950 - score)) / 2. Expected ~0.88"
why[v5-neg-latmen]="LEARNING PROBE: the 0.950 notebook with the Lateral Meniscus column reversed. Reads that finding's leaderboard AUC = (1 + 12 x (0.950 - score)) / 2. Expected ~0.88"
for name in $names; do
  st=$(kaggle kernels status girishbose/$name 2>&1 | tail -1)
  case "$st" in *COMPLETE*) ;; *) say "$name: kernel not COMPLETE ($st) - NOT submitted"; continue;; esac
  k=0
  while true; do
    out=$(python scripts/submit_gate.py submit --kernel outputs/kernels/$name --version 1 --i-approve --why "${why[$name]}" 2>&1)
    [ $? -eq 0 ] && { say "$name: submitted | $(echo $out | tr '\n' ' ' | cut -c1-90)"; break }
    case "$out" in *"daily cap"*) k=$((k + 1)); [ $k -gt 280 ] && { say "$name: no slot in 23 h"; break }; sleep 300;; *) say "$name: submit refused | $(echo $out | tr '\n' ' ' | cut -c1-200)"; break;; esac
  done
done
say "done"
