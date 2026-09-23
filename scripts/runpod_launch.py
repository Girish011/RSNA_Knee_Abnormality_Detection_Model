#!/usr/bin/env python3
"""Bundle a RunPod job as a private Kaggle dataset, then start a 4090 pod that runs it.

The pod downloads the bundle and the public corpus with the Kaggle API, runs
``runpod/run_job.sh`` over the arms in ``arms.txt``, uploads results to a private Kaggle
dataset, and terminates itself. Watch it at ``<proxy>/STATUS`` and ``<proxy>/job.log``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from rsna_knee.agent import runpod
from rsna_knee.agent.activity import assert_budget, log
from rsna_knee.constants import LABEL_COLS

REPO = Path(__file__).resolve().parents[1]
USER = "girishbose"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PRICE_HR = 0.74
COATNET = ("--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 16 --bs 8 --k 12 "
           "--k_eval 24 --grad_ckpt --norm imagenet --seed 42 --workers 8")
JOBS = {
    "labels-ab-v1": {
        "arms": [("dread_s42", "labels_dread.parquet", COATNET),
                 ("blendv2_s42", "labels_llm_blend_v2.parquet", COATNET)],
        "hours": 7.0,
    },
}

BOOT = """set -u; mkdir -p /workspace/out ~/.kaggle
cd /workspace/out && nohup python -m http.server 8000 >/dev/null 2>&1 &
printf %s "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
pip install -q kaggle
for i in $(seq 1 20); do
  kaggle datasets download {bundle} -p /workspace/bundle --unzip -q && break
  echo "bundle not ready, retry $i" >> /workspace/out/job.log; sleep 30
done
bash /workspace/bundle/run_job.sh
sleep infinity"""


def build_bundle(job: str) -> Path:
    spec = JOBS[job]
    d = REPO / "outputs" / "runpod_bundle" / job
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(REPO / "runpod" / "train_knee.py", d)
    shutil.copy(REPO / "runpod" / "run_job.sh", d)
    train = pd.read_csv(REPO / "data" / "raw" / "train.csv")
    train[["StudyInstanceUID", *LABEL_COLS]].to_csv(d / "train.csv", index=False)
    for _, labels, _ in spec["arms"]:
        shutil.copy(REPO / "data" / "processed" / labels, d)
    (d / "arms.txt").write_text("".join(f"{t}|{lab}|{extra}\n" for t, lab, extra in spec["arms"]))
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": f"rsna-knee-rp-{job}-bundle", "id": f"{USER}/rsna-knee-rp-{job}-bundle",
        "licenses": [{"name": "CC0-1.0"}]}))
    return d


def upload_bundle(d: Path) -> str:
    slug = json.loads((d / "dataset-metadata.json").read_text())["id"]
    exists = subprocess.run(["kaggle", "datasets", "status", slug], capture_output=True, text=True)
    cmd = (["kaggle", "datasets", "version", "-p", str(d), "-m", "update", "-q"]
           if exists.returncode == 0 and "ready" in exists.stdout.lower()
           else ["kaggle", "datasets", "create", "-p", str(d), "-q"])
    subprocess.run(cmd, check=True)
    return slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = JOBS[a.job]
    est = spec["hours"] * PRICE_HR
    assert_budget(est)

    d = build_bundle(a.job)
    print("bundle:", sorted(p.name for p in d.iterdir()))
    if a.dry_run:
        return
    bundle = upload_bundle(d)
    time.sleep(90)
    result = f"{USER}/rsna-knee-rp-{a.job}-out"
    base = {
        "name": f"rsna-{a.job}", "imageName": IMAGE, "computeType": "GPU", "gpuCount": 1,
        "containerDiskInGb": 30, "volumeInGb": 100, "volumeMountPath": "/workspace",
        "ports": ["8000/http"],
        "allowedCudaVersions": ["13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4"],
        "env": {"KAGGLE_API_TOKEN": Path("~/.kaggle/access_token").expanduser().read_text().strip(),
                "RUNPOD_API_KEY": runpod.secret("RUNPOD_API_KEY"),
                "RESULT_DATASET": result, "RESULT_TITLE": f"rsna-knee-rp-{a.job}-out"},
        "dockerStartCmd": ["bash", "-c", BOOT.replace("{bundle}", bundle)],
    }
    attempts = [
        {"cloudType": "SECURE", "gpuTypeIds": [runpod.GPU_4090]},
        {"cloudType": "COMMUNITY", "gpuTypeIds": [runpod.GPU_4090]},
        {"cloudType": "COMMUNITY", "gpuTypeIds": [runpod.GPU_4090, runpod.GPU_3090]},
    ]
    pod = None
    for extra in attempts:
        try:
            pod = runpod.create_pod({**base, **extra})
            break
        except RuntimeError as e:
            print(f"no capacity for {extra['cloudType']} {extra['gpuTypeIds']}: {str(e)[:120]}")
    if pod is None:
        log("error", f"RunPod job {a.job}: no GPU capacity on any fallback", bundle=bundle)
        raise SystemExit("no RunPod capacity; bundle is uploaded, rerun later")
    pod_id = pod["id"]
    url = runpod.proxy_url(pod_id)
    log("train", f"RunPod job {a.job} launched: {[t for t, _, _ in spec['arms']]}",
        cost_usd=0.0, pod=pod_id, bundle=bundle, result=result, monitor=url,
        est_usd=round(est, 2), price_hr=pod.get("costPerHr"))
    print(json.dumps({"pod": pod_id, "costPerHr": pod.get("costPerHr"), "monitor": url,
                      "bundle": bundle, "result": result}, indent=1))


if __name__ == "__main__":
    main()
