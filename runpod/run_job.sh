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

# Host-side watchdog reads this every loop tick. If it goes stale, the Mac kills the pod
# instead of billing a hung/dead community box for hours.
(
  while true; do
    date -u +%s > "$OUT/HEARTBEAT"
    sleep 60
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Package results and push them to the private result dataset. First call creates it, later
# calls add a version, so each finished arm is safe even if the host dies before the next.
PUSHED=0
push_results() {
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
  if [ "$PUSHED" = 0 ]; then
    # A rerun under an existing job name must version the dataset, not fail on create.
    { kaggle datasets create -p "$U" --dir-mode zip -q || kaggle datasets version -p "$U" --dir-mode zip -m "$1" -q; } && PUSHED=1
  else
    kaggle datasets version -p "$U" --dir-mode zip -m "$1" -q
  fi
}

finish() {
  status "uploading ($1)"
  if push_results "$1"; then
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
timeout 900 pip install --progress-bar off --timeout 20 --retries 2 --prefer-binary -c /tmp/pins.txt timm kaggle pyarrow scikit-learn pandas 2>&1 \
  | grep --line-buffered -E "^(Collecting|Successfully|ERROR)" || true
python -c "import timm, sklearn, pyarrow, pandas; print('deps ok, timm', timm.__version__)" || finish "pip failed"
# fastMRI (Family B) only: separate, non-fatal so a resolver clash can never break CoAt jobs.
timeout 300 pip install --progress-bar off --prefer-binary -c /tmp/pins.txt h5py 2>&1 | tail -1 || true
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

status "downloading corpus (${CORPUS:-raptor44})"
mkdir -p /kaggle && ln -sfn /workspace/input /kaggle/input
if [ "${CORPUS:-raptor44}" = "dense80" ]; then
  # Own 80-slot submit geometry (~10 GB/shard). Prefer Range-resume fetch over
  # ``kaggle kernels output`` (IncompleteRead near EOF burned a day of GPU).
  mkdir -p /workspace/input/dense80
  status "downloading dense80 (resumeable)"
  python "$B/fetch_dense80.py" || finish "download dense80 failed after resume retries"
elif [ "${CORPUS:-raptor44}" = "dense512" ]; then
  # Native-resolution cache for the region specialists (8 shards, ~11.6 GB each, + geometry).
  mkdir -p /workspace/input/dense512
  status "downloading dense512 (resumeable)"
  DENSE_NAME=dense512 DENSE_SHARDS=8 DENSE80_DIR=/workspace/input/dense512 python "$B/fetch_dense80.py" \
    || finish "download dense512 failed after resume retries"
else
  for ds in knee-raptor-corpus knee-raptor-corpus-ext; do
    kaggle datasets download "dreaddevelopment/$ds" -p "/workspace/input/$ds" --unzip -q \
      || finish "download $ds failed"
  done
fi
# Checkpoints an arm starts from or scores (--init_model): inits.txt = file|dataset|path-in-dataset.
if [ -s "$B/inits.txt" ]; then
  status "downloading init checkpoints"
  mkdir -p /workspace/input/init
  while IFS='|' read -r fname ds path; do
    [ -z "$fname" ] && continue
    for i in 1 2 3; do
      kaggle datasets download "$ds" -f "$path" -p /workspace/input/init -q </dev/null && break
      sleep 20
    done
    (cd /workspace/input/init && for z in *.zip; do [ -f "$z" ] && unzip -o -q "$z" && rm -f "$z"; done)
    [ -f "/workspace/input/init/$fname" ] || finish "init checkpoint $fname download failed"
  done < "$B/inits.txt"
fi
ls -la /workspace/input/*/
cp "$B/train.csv" "$OUT/train.csv"

status "smoke"
# TRAINER: train_knee.py (CoAt members, default) or family_a.py (region-aware specialists).
TRAINER="${TRAINER:-train_knee.py}"
mkdir -p "$OUT/smoke" && cp "$B/$TRAINER" "$OUT/smoke/"
first_labels=$(head -1 "$B/arms.txt" | cut -d'|' -f2)
if [ "$TRAINER" = train_knee.py ]; then
  SMOKE_ARGS="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --norm imagenet"
elif [ "$TRAINER" = region_spec.py ]; then
  SMOKE_ARGS="--workers 4 --region tf"
else
  SMOKE_ARGS="--workers 4"
fi
# shellcheck disable=SC2086
python "$OUT/smoke/$TRAINER" --smoke $SMOKE_ARGS --grad_ckpt --tag smoke --labels "$B/$first_labels" \
  > "$OUT/smoke/train.log" 2>&1 || { tail -40 "$OUT/smoke/train.log"; finish "smoke failed"; }
tail -5 "$OUT/smoke/train.log"

while IFS='|' read -r tag labels extra; do
  [ -z "$tag" ] && continue
  # Per-arm trainer override: extra args starting with "@script.py" run that script instead.
  ARM_TRAINER="$TRAINER"
  case "$extra" in @*) ARM_TRAINER="${extra%% *}"; ARM_TRAINER="${ARM_TRAINER#@}"; extra="${extra#* }";; esac
  status "training $tag"
  mkdir -p "$OUT/$tag" && cp "$B/$ARM_TRAINER" "$OUT/$tag/"
  # shellcheck disable=SC2086
  if ! python "$OUT/$tag/$ARM_TRAINER" --labels "$B/$labels" --tag "$tag" $extra \
    > "$OUT/$tag/train.log" 2>&1; then
    # Host limits (small /dev/shm → DataLoader "Bus error"; mmap faults) killed arms on A40 hosts.
    echo "arm $tag FAILED; retrying once with --workers 2"
    mv "$OUT/$tag/train.log" "$OUT/$tag/train_try1.log"
    extra2=$(echo "$extra" | sed -E 's/--workers [0-9]+/--workers 2/')
    # shellcheck disable=SC2086
    python "$OUT/$tag/$ARM_TRAINER" --labels "$B/$labels" --tag "$tag" $extra2 \
      > "$OUT/$tag/train.log" 2>&1 || echo "arm $tag FAILED twice (continuing)"
  fi
  tail -20 "$OUT/$tag/train.log"
  # Multi-arm jobs: checkpoint finished arms to Kaggle before starting the next one.
  if [ "$(grep -c . "$B/arms.txt")" -gt 1 ]; then
    status "uploading partial after $tag"
    push_results "after $tag" </dev/null || echo "partial upload after $tag failed (continuing)"
  fi
done < "$B/arms.txt"

finish "all arms finished"
