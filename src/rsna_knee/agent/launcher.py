"""Bundle a RunPod job as a private Kaggle dataset, then start a GPU pod that runs it.

The pod downloads the bundle and the public corpus with the Kaggle API, runs
``runpod/run_job.sh`` over the arms in ``arms.txt``, uploads results to a private Kaggle
dataset ``<user>/rsna-knee-rp-<job>-out`` and terminates itself. Progress is served at
``<proxy>/STATUS`` and ``<proxy>/job.log``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from rsna_knee.agent import runpod
from rsna_knee.agent.activity import assert_budget, log
from rsna_knee.constants import LABEL_COLS

REPO = Path(__file__).resolve().parents[3]
USER = "girishbose"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PRICE_HR = 0.74  # worst case (secure 4090) for the budget check
PROTOCOL_FILES = ["folds_nonexpert_5.json", "expert_holdout.json"]

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

Arm = tuple[str, str, str]  # (tag, labels parquet under data/processed, trainer args)


def result_slug(job: str) -> str:
    return f"{USER}/rsna-knee-rp-{job}-out"


def build_bundle(job: str, arms: list[Arm]) -> Path:
    d = REPO / "outputs" / "runpod_bundle" / job
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(REPO / "runpod" / "train_knee.py", d)
    shutil.copy(REPO / "runpod" / "run_job.sh", d)
    for f in PROTOCOL_FILES:
        shutil.copy(REPO / "agent" / "protocol" / f, d)
    train = pd.read_csv(REPO / "data" / "raw" / "train.csv")
    train[["StudyInstanceUID", *LABEL_COLS]].to_csv(d / "train.csv", index=False)
    for _, labels, _ in arms:
        shutil.copy(REPO / "data" / "processed" / labels, d)
    (d / "arms.txt").write_text("".join(f"{t}|{lab}|{extra}\n" for t, lab, extra in arms))
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
    subprocess.run(cmd, check=True, capture_output=True)
    return slug


def launch(job: str, arms: list[Arm], hours: float, *, capacity_rounds: int = 8) -> dict:
    """Upload the bundle and start a pod. Returns {pod, costPerHr, monitor, bundle, result}.

    Retries across community 4090 → 4090/3090 → secure 4090, with backoff, until a host is
    free or ``capacity_rounds`` is exhausted.
    """
    assert_budget(hours * PRICE_HR)
    bundle = upload_bundle(build_bundle(job, arms))
    time.sleep(30)
    result = result_slug(job)
    base = {
        "name": f"rsna-{job}", "imageName": IMAGE, "computeType": "GPU", "gpuCount": 1,
        "containerDiskInGb": 30, "volumeInGb": 100, "volumeMountPath": "/workspace",
        "ports": ["8000/http"],
        "allowedCudaVersions": ["13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4"],
        "env": {"KAGGLE_API_TOKEN": Path("~/.kaggle/access_token").expanduser().read_text().strip(),
                "RUNPOD_API_KEY": runpod.secret("RUNPOD_API_KEY"),
                "RESULT_DATASET": result, "RESULT_TITLE": f"rsna-knee-rp-{job}-out"},
        "dockerStartCmd": ["bash", "-c", BOOT.replace("{bundle}", bundle)],
    }
    attempts = [
        {"cloudType": "COMMUNITY", "gpuTypeIds": [runpod.GPU_4090]},
        {"cloudType": "COMMUNITY", "gpuTypeIds": [runpod.GPU_4090, runpod.GPU_3090]},
        {"cloudType": "SECURE", "gpuTypeIds": [runpod.GPU_4090]},
        {"cloudType": "COMMUNITY", "gpuTypeIds": [runpod.GPU_3090]},
    ]
    pod = None
    last_err = ""
    for round_i in range(capacity_rounds):
        for extra in attempts:
            try:
                pod = runpod.create_pod({**base, **extra})
                break
            except RuntimeError as e:
                last_err = str(e)[:160]
                continue
        if pod is not None:
            break
        wait = min(30 * (round_i + 1), 120)
        log("note", f"RunPod capacity miss round {round_i + 1}/{capacity_rounds}; retry in {wait}s",
            job=job, err=last_err)
        time.sleep(wait)
    if pod is None:
        log("error", f"RunPod job {job}: no GPU capacity after {capacity_rounds} rounds",
            bundle=bundle, err=last_err)
        raise RuntimeError("no RunPod capacity")
    out = {"pod": pod["id"], "costPerHr": pod.get("costPerHr"), "monitor": runpod.proxy_url(pod["id"]),
           "bundle": bundle, "result": result}
    log("train", f"RunPod job {job} launched: {[t for t, _, _ in arms]}", pod=out["pod"],
        bundle=bundle, result=result, monitor=out["monitor"], price_hr=out["costPerHr"],
        est_hours=round(hours, 1))
    return out
