#!/usr/bin/env bash
# Boot of the multi-GPU queue pod: deps, image cache once, then runpod/mega.py (restarted if it crashes).
# Progress: https://<pod>-8000.proxy.runpod.net/ (STATUS, job.log, mega_state.json, <job>/<tag>/train.log).
set -uo pipefail
Q=/workspace/queue
OUT=/workspace/out
mkdir -p "$OUT" /workspace/input/dense80
exec > >(tee -a "$OUT/job.log") 2>&1
status() { echo "$1" > "$OUT/STATUS"; echo "=== STATUS $1 $(date -u +%FT%TZ)"; }
bye() { status "$1"; sleep 60; curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"; exit 1; }
(while true; do date -u +%s > "$OUT/HEARTBEAT"; sleep 60; done) &

status "installing"
pip freeze | grep -E '^(torch|torchvision|numpy)==' > /tmp/pins.txt; cat /tmp/pins.txt
timeout 900 pip install --progress-bar off --timeout 20 --retries 2 --prefer-binary -c /tmp/pins.txt \
  timm kaggle pyarrow scikit-learn pandas requests 2>&1 | grep --line-buffered -E "^(Collecting|Successfully|ERROR)" || true
python -c "import timm, sklearn, pyarrow, pandas; print('deps ok, timm', timm.__version__)" || bye "pip failed; deleting pod"
# DICOM reading for the fastMRI cache (separate and non-fatal: training arms must not depend on it)
timeout 600 pip install --progress-bar off --prefer-binary -c /tmp/pins.txt pydicom opencv-python-headless \
  pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg 2>&1 | tail -1 || true
apt-get install -y -q xz-utils >/dev/null 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv; free -g | head -2; nproc; df -h /workspace | tail -1
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || bye "GPU FAULT: torch cannot see CUDA; deleting pod"

mkdir -p /kaggle && ln -sfn /workspace/input /kaggle/input
ln -sfn "$Q" /workspace/bundle                 # arm args point at /workspace/bundle/<protocol file>
cp "$Q/train.csv" /workspace/input/train.csv
status "downloading dense80"
python "$Q/fetch_dense80.py" || bye "dense80 download failed; deleting pod"

for i in 1 2 3 4 5 6; do
  status "mega worker running (start $i)"
  python "$Q/mega.py" && exit 0              # exit 0 = STOP handled, pod delete requested
  echo "mega.py exited with an error; restart in 60 s"; sleep 60
done
status "mega worker crashed 6 times; pod left running for inspection"
sleep infinity
