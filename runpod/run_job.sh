#!/usr/bin/env bash
# On-pod job: fetch the public corpus, smoke-test, train each arm in arms.txt sequentially,
# push results to a private Kaggle dataset, then terminate this pod (stop it if upload fails).
# Progress is served at https://<pod>-8000.proxy.runpod.net/ (STATUS, job.log, per-arm dirs).
set -uo pipefail
B=/workspace/bundle
OUT=/workspace/out
mkdir -p "$OUT" /workspace/input
exec > >(tee -a "$OUT/job.log") 2>&1
status() { echo "$1" > "$OUT/STATUS"; echo "=== STATUS $1 $(date -u +%FT%TZ)"; }

finish() {
  status "uploading ($1)"
  U=/workspace/upload; rm -rf "$U"; mkdir -p "$U"
  cp "$OUT"/job.log "$OUT"/STATUS "$U"/ 2>/dev/null
  for d in "$OUT"/*/; do
    t=$(basename "$d"); mkdir -p "$U/$t"
    cp "$d"*.log "$U/$t"/ 2>/dev/null
    [ "$t" = smoke ] || cp "$d"raptor_* "$U/$t"/ 2>/dev/null
  done
  cat > "$U/dataset-metadata.json" <<EOF
{"title": "${RESULT_TITLE}", "id": "${RESULT_DATASET}", "licenses": [{"name": "CC0-1.0"}]}
EOF
  if kaggle datasets create -p "$U" --dir-mode zip -q; then
    status "done ($1); uploaded ${RESULT_DATASET}"
    sleep 30
    curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
  else
    status "UPLOAD FAILED ($1); stopping pod, results on volume"
    curl -s -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop"
  fi
  exit 0
}

status "installing"
pip freeze | grep -E '^(torch|torchvision|numpy)==' > /tmp/pins.txt; cat /tmp/pins.txt
pip install --progress-bar off -c /tmp/pins.txt timm kaggle pyarrow scikit-learn pandas 2>&1 \
  | grep -E "^(Collecting|Successfully|ERROR)" || true
python -c "import timm, sklearn, pyarrow, pandas; print('deps ok, timm', timm.__version__)" || finish "pip failed"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g | head -2; nproc
gpu_ok=0
for i in 1 2 3 4; do
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" && { gpu_ok=1; break; }
  echo "torch cannot see CUDA (try $i)"; nvidia-smi -L; sleep 15
done
if [ "$gpu_ok" != 1 ]; then
  status "GPU FAULT: torch cannot see CUDA on this host; terminating"
  curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
  exit 1
fi

status "downloading corpus"
mkdir -p /kaggle && ln -sfn /workspace/input /kaggle/input
for ds in knee-raptor-corpus knee-raptor-corpus-ext; do
  kaggle datasets download "dreaddevelopment/$ds" -p "/workspace/input/$ds" --unzip -q \
    || finish "download $ds failed"
done
ls -la /workspace/input/*/
cp "$B/train.csv" "$OUT/train.csv"

status "smoke"
mkdir -p "$OUT/smoke" && cp "$B/train_knee.py" "$OUT/smoke/"
first_labels=$(head -1 "$B/arms.txt" | cut -d'|' -f2)
python "$OUT/smoke/train_knee.py" --smoke --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --norm imagenet --grad_ckpt --tag smoke --labels "$B/$first_labels" \
  > "$OUT/smoke/train.log" 2>&1 || { tail -40 "$OUT/smoke/train.log"; finish "smoke failed"; }
tail -5 "$OUT/smoke/train.log"

while IFS='|' read -r tag labels extra; do
  [ -z "$tag" ] && continue
  status "training $tag"
  mkdir -p "$OUT/$tag" && cp "$B/train_knee.py" "$OUT/$tag/"
  # shellcheck disable=SC2086
  python "$OUT/$tag/train_knee.py" --labels "$B/$labels" --tag "$tag" $extra \
    > "$OUT/$tag/train.log" 2>&1 || echo "arm $tag FAILED (continuing)"
  tail -20 "$OUT/$tag/train.log"
done < "$B/arms.txt"

finish "all arms finished"
