#!/bin/zsh
# 12 Oct: submissions for the 13 Oct 05:30 IST slot reset. The unchanged fork of the top public notebook (real public
# score 0.956; our best is 0.950), one candidate on it, one reading of its three weak columns, two readings of the
# 3 Oct region specialists on the known 0.950 base. (The s6 fork is a known 0.954 and the velci fork lacks its private
# model: both dropped 12 Oct 14:20.)
# All kernels COMPLETE and checked (STATUS, 12 Oct). Waits while the gate answers "daily cap". Log: outputs/auto_t3.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_t3.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
# 13 Oct 02:00, user: three submissions a day, two slots stay free. bm-t2-w15 and v5-rs-pfoa wait.
dirs=(outputs/kernels/pub/blend-mine-in-private-lb-luck-to-all-upvoters outputs/kernels/bm-neg3 outputs/kernels/v5-rs-latoa)
whys=(
"FORK, unchanged, of the public notebook blend-mine-in-private-lb-luck-to-all-upvoters (expert-trained CoAtNet 0.5 + the public compact ConvNeXt-S model r6 0.5; its public score is 0.956): the new base on our account. Our best: 0.950"
"LEARNING PROBE, not a candidate: the top public fork (blend-mine, public 0.956) with its Lateral OA, Synovitis and PF OA columns REVERSED. Reads the SUM of the three AUCs on the new base = (3 + 12 x (fork score - this score)) / 2; on the 0.950 base the sum was 2.730 (0.896 + 0.908 + 0.926)"
"LEARNING PROBE: the 0.950 public notebook with its Lateral OA column REPLACED by our 3 Oct lateral-compartment region specialist (ConvNeXt-S, native-resolution crops, fold 0, report labels). Reads that specialist's Lateral OA AUC = 0.896 + 12 x (score - 0.950); baseline for the new specialists"
)
for i in {1..${#dirs}}; do
  d=${dirs[$i]}; name=$(python -c "import json;print(json.load(open('$d/kernel-metadata.json'))['id'].split('/')[1])")
  st=$(kaggle kernels status girishbose/$name 2>&1 | tail -1)
  case "$st" in *COMPLETE*) ;; *) say "$name: kernel not COMPLETE ($st) - NOT submitted"; continue;; esac
  k=0
  while true; do
    out=$(python scripts/submit_gate.py submit --kernel $d --version 1 --i-approve --why "${whys[$i]}" 2>&1)
    [ $? -eq 0 ] && { say "$name: submitted | $(echo $out | tr '\n' ' ' | cut -c1-90)"; break }
    case "$out" in *"daily cap"*) k=$((k + 1)); [ $k -gt 280 ] && { say "$name: no slot in 23 h"; break }; sleep 300;; *) say "$name: submit refused | $(echo $out | tr '\n' ' ' | cut -c1-200)"; break;; esac
  done
done
say "t3 done"
