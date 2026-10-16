#!/bin/zsh
# 11 Oct: when the paired run `ktrain-t2` (ConvNeXt-T @224, folds 0-1, targets = the expert-trained checkpoints' order)
# is complete: collect, check both arms, build + push the alone-probe `t2-alone`, check its log, submit it at the
# 12 Oct 05:30 IST slot reset. Control: adj-alone = 0.929. Log: outputs/auto_t2.log
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model || exit 1
export PATH="$PWD/.venv/bin:$PATH"
LOG=outputs/auto_t2.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG }
wait_kernel() {   # name -> 0 when COMPLETE; network errors are not an end state
  while true; do
    st=$(kaggle kernels status girishbose/$1 2>&1 | tail -1)
    case "$st" in *COMPLETE*) return 0;; *ERROR*|*CANCEL*) say "$1: kernel ended ($st)"; return 1;; esac
    sleep 300
  done
}
sleep 2400
st=$(kaggle kernels status girishbose/ktrain-t2 2>&1 | tail -1); say "ktrain-t2 after 40 min: $st (still RUNNING = past the smoke run)"
wait_kernel ktrain-t2 || exit 1
python scripts/build_kaggle_train.py --name ktrain-t2 --collect >> $LOG 2>&1
for t in t2kadj0 t2kadj1; do
  f=outputs/kernels/ktrain-t2/_out/$t/train.log
  n=$(grep -c "^ep" $f 2>/dev/null); [ "$n" = 40 ] || { say "$t: $n of 40 epochs - NOT probed"; exit 1 }
  [ -f outputs/kernels/ktrain-t2/_out/$t/raptor_ft_$t.pt ] || { say "$t: no checkpoint - NOT probed"; exit 1 }
  say "$t: $(grep '^ep39' $f | cut -c1-80) | $(grep 'SWA over' $f | cut -c1-90)"
done
python scripts/build_own_probe.py --name t2-alone --members k/ktrain-t2:t2kadj0,k/ktrain-t2:t2kadj1 >> $LOG 2>&1
kaggle kernels push -p outputs/kernels/t2-alone >> $LOG 2>&1
sleep 60
wait_kernel t2-alone || exit 1
kaggle kernels logs girishbose/t2-alone > outputs/kernels/t2-alone/run.log 2>/dev/null
for pat in "leg exit 0" "submission.csv (3, 13)" "raptor_ft_t2kadj0.pt" "raptor_ft_t2kadj1.pt"; do
  grep -q -- "$pat" outputs/kernels/t2-alone/run.log || { say "t2-alone: log check failed (no '$pat') - NOT submitted"; exit 1 }
done
say "t2-alone: kernel ok"
k=0
while true; do
  out=$(python scripts/submit_gate.py submit --kernel outputs/kernels/t2-alone --version 1 --i-approve --why "LEARNING PROBE: two ConvNeXt-T @224 folds (0,1) ALONE, trained on the two expert-trained public checkpoints' ordering of the training studies instead of report labels; everything else as the control adj-alone = 0.929. Reads whether that judgement transfers to our own models" 2>&1)
  [ $? -eq 0 ] && { say "t2-alone: submitted | $(echo $out | tr '\n' ' ' | cut -c1-90)"; break }
  case "$out" in *"daily cap"*) k=$((k + 1)); [ $k -gt 280 ] && { say "t2-alone: no slot in 23 h"; break }; sleep 300;; *) say "t2-alone: submit refused | $(echo $out | tr '\n' ' ' | cut -c1-200)"; break;; esac
done
say "t2b done"
