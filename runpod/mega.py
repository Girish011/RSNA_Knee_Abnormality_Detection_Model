#!/usr/bin/env python3
"""Multi-GPU queue worker for one long-lived pod: runs one arm per GPU and keeps taking new arms.

The pod downloads the image cache once and then polls a private Kaggle dataset (QUEUE_DATASET) every few
minutes. That dataset is the remote control: it holds

  arms.jsonl      one JSON object per line
                    training arm : {"job": "r3-effv2s-f1", "tag": "r3effv2s1", "labels": "labels_r3.parquet",
                                    "args": "--arch ... --fold 1 ...", "trainer": "train_knee.py"}
                    shell task   : {"job": "fm-cache", "cmd": "python fastmri_dicom_cache.py", "gpu": 0}
                  optional keys: "gpu" (0 = CPU task, runs beside the GPU arms), "needs" (paths that must exist
                  before the arm may start), "tries" (default 2), "upload" (shell tasks: true to upload the job dir)
  *.py / *.parquet / *.json / *.csv   code, labels, protocol files; copied over the queue dir at each poll, so a
                  trainer fix or a new label file reaches the pod without a restart (running arms keep their copy)
  STOP            when present and nothing is running: the pod deletes itself

Each finished training arm is uploaded as the private dataset <user>/rsna-knee-rp-<job>-out with the usual
<tag>/raptor_* layout, so the collectors and kernel builders work unchanged. State is kept in
/workspace/out/mega_state.json (served on port 8000) and survives a restart of this script.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

Q = Path(os.environ.get("MEGA_QUEUE_DIR", "/workspace/queue"))
OUT = Path(os.environ.get("MEGA_OUT_DIR", "/workspace/out"))
STATE = OUT / "mega_state.json"
QUEUE_DATASET = os.environ.get("QUEUE_DATASET", "")
USER = os.environ.get("KAGGLE_USER", "girishbose")
DRY = os.environ.get("MEGA_DRY") == "1"          # local test: no Kaggle calls, no pod deletion
POLL_S = int(os.environ.get("MEGA_POLL_S", "180"))
TICK_S = float(os.environ.get("MEGA_TICK_S", "20"))


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def n_gpus() -> int:
    if os.environ.get("MEGA_GPUS"):
        return int(os.environ["MEGA_GPUS"])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
        return max(1, sum(1 for ln in out.splitlines() if ln.startswith("GPU ")))
    except Exception:  # noqa: BLE001
        return 1


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATE)


def refresh_queue() -> None:
    """Pull the newest queue dataset version over the queue dir. Files are added or replaced, never removed
    (a running arm may still read them); STOP is the one exception so that a stop can be withdrawn."""
    if DRY or not QUEUE_DATASET:
        return
    tmp = Path("/workspace/queue_dl")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        r = subprocess.run(["kaggle", "datasets", "download", QUEUE_DATASET, "-p", str(tmp), "--unzip", "--force", "-q"],
                           capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        log("queue refresh timed out")
        return
    if r.returncode != 0 or not (tmp / "arms.jsonl").exists():
        log(f"queue refresh failed: {(r.stderr or r.stdout)[-200:]}")
        return
    for f in tmp.iterdir():
        if f.is_file():
            shutil.copy2(f, Q / f.name)
    if not (tmp / "STOP").exists():
        (Q / "STOP").unlink(missing_ok=True)


def read_arms() -> list[dict]:
    arms = []
    try:
        lines = (Q / "arms.jsonl").read_text().splitlines()
    except OSError:
        return arms
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            a = json.loads(ln)
        except ValueError:
            log(f"bad arm line skipped: {ln[:80]}")
            continue
        if a.get("job"):
            arms.append(a)
    return arms


def pid_alive(pid: int) -> bool:
    """For arms inherited from an earlier run of this script (no Popen handle)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][0] != "Z"
    except (OSError, IndexError):
        return False


def arm_dir(arm: dict) -> Path:
    return OUT / arm["job"] / (arm.get("tag") or arm["job"])


def start(arm: dict, gpu: int | None, tries: int) -> subprocess.Popen:
    d = arm_dir(arm)
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    if "cmd" in arm:
        cmd, cwd = arm["cmd"], str(Q)
        env["MEGA_JOB_DIR"] = str(d)
    else:
        trainer = arm.get("trainer", "train_knee.py")
        shutil.copy2(Q / trainer, d / trainer)
        if (Q / "train.csv").exists():      # the trainer looks for train.csv one level above its own dir
            shutil.copy2(Q / "train.csv", OUT / arm["job"] / "train.csv")
        args = arm["args"]
        if tries > 1:                       # small /dev/shm on some hosts kills DataLoader workers
            args = re.sub(r"--workers \d+", "--workers 2", args)
        cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(str(d / trainer))} --labels {shlex.quote(str(Q / arm['labels']))} "
               f"--tag {shlex.quote(arm.get('tag') or arm['job'])} {args}")
        cwd = str(d)
    logp = d / "train.log"
    if logp.exists():
        logp.rename(d / f"train_try{tries - 1}.log")
    return subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=open(logp, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)


def succeeded(arm: dict, rc: int | None) -> bool:
    if "cmd" in arm:
        return rc == 0 or (rc is None and (arm_dir(arm) / "DONE").exists())
    tag = arm.get("tag") or arm["job"]
    try:
        return json.loads((arm_dir(arm) / f"raptor_ft_{tag}.json").read_text()).get("partial") is False
    except (OSError, ValueError):
        return False


def upload(arm: dict) -> bool:
    """Private dataset <user>/rsna-knee-rp-<job>-out: <tag>/raptor_* + logs (same layout as run_job.sh)."""
    if DRY or ("cmd" in arm and not arm.get("upload")):
        return True
    job = arm["job"]
    up = Path("/workspace/upload") / job
    shutil.rmtree(up, ignore_errors=True)
    up.mkdir(parents=True)
    for d in (OUT / job).iterdir():
        if not d.is_dir():
            continue
        (up / d.name).mkdir()
        for f in d.iterdir():
            if f.is_file() and (f.name.startswith("raptor_") or f.suffix in (".log", ".json", ".txt")):
                shutil.copy2(f, up / d.name / f.name)
    (up / "STATUS").write_text(f"done {job} (queue pod)\n")   # a top-level file keeps <tag>/ from being flattened
    (up / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"rsna-knee-rp-{job}-out", "id": f"{USER}/rsna-knee-rp-{job}-out", "licenses": [{"name": "CC0-1.0"}]}))
    for cmd in (["kaggle", "datasets", "create", "-p", str(up), "--dir-mode", "zip", "-q"],
                ["kaggle", "datasets", "version", "-p", str(up), "--dir-mode", "zip", "-m", "mega", "-q"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0:
            shutil.rmtree(up, ignore_errors=True)
            return True
    log(f"upload of {job} failed: {(r.stderr or r.stdout)[-200:]}")
    return False


def delete_pod() -> None:
    if DRY:
        return
    subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {os.environ.get('RUNPOD_API_KEY', '')}",
                    f"https://rest.runpod.io/v1/pods/{os.environ.get('RUNPOD_POD_ID', '')}"], timeout=120)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpus, st, procs, last_poll = n_gpus(), load_state(), {}, 0.0
    log(f"mega worker: {gpus} GPU(s) | queue {QUEUE_DATASET or Q}")
    while True:
        if time.time() - last_poll >= POLL_S:
            refresh_queue()
            last_poll = time.time()
        arms = read_arms()
        by_job = {a["job"]: a for a in arms}
        # 1. finished arms: check, upload, mark
        for job, s in st.items():
            if s["status"] != "running":
                continue
            p = procs.get(job)
            rc = p.poll() if p is not None else None
            if (p is not None and rc is None) or (p is None and pid_alive(s["pid"])):
                continue
            arm = by_job.get(job) or s["arm"]
            procs.pop(job, None)
            if succeeded(arm, rc):
                s["status"] = "uploading"
                save_state(st)
                s["status"] = "done" if upload(arm) else "done_upload_failed"
            else:
                s["status"] = "retry" if s["tries"] < int(arm.get("tries", 2)) else "failed"
            s["t1"] = time.time()
            log(f"{job}: {s['status']} (try {s['tries']}, rc {rc}, {(s['t1'] - s['t0']) / 60:.0f} min)")
            save_state(st)
        # 2. start what fits
        busy = {s["gpu"] for s in st.values() if s["status"] == "running" and s.get("gpu") is not None}
        free = [g for g in range(gpus) if g not in busy]
        for arm in arms:
            s = st.get(arm["job"])
            if s and s["status"] != "retry":
                continue
            if any(not Path(n).exists() for n in arm.get("needs", [])):
                continue
            wants_gpu = int(arm.get("gpu", 1)) > 0
            if wants_gpu and not free:
                continue
            g = free.pop(0) if wants_gpu else None
            tries = (s["tries"] if s else 0) + 1
            try:
                p = start(arm, g, tries)
            except Exception as e:  # noqa: BLE001  (missing trainer / labels file: wait for the next queue version)
                log(f"{arm['job']}: cannot start yet ({type(e).__name__}: {e})")
                if g is not None:
                    free.insert(0, g)
                continue
            procs[arm["job"]] = p
            st[arm["job"]] = {"status": "running", "gpu": g, "pid": p.pid, "tries": tries, "t0": time.time(), "arm": arm}
            log(f"{arm['job']}: started on {'cpu' if g is None else 'gpu ' + str(g)} (try {tries})")
            save_state(st)
        # 3. status line + stop
        counts: dict[str, int] = {}
        for s in st.values():
            counts[s["status"]] = counts.get(s["status"], 0) + 1
        waiting = sum(1 for a in arms if a["job"] not in st)
        (OUT / "STATUS").write_text(f"mega {gpus} gpu | " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                                    + f" | waiting {waiting} | {time.strftime('%FT%TZ', time.gmtime())}\n")
        if (Q / "STOP").exists() and not counts.get("running"):
            log("STOP found and nothing running: deleting the pod")
            (OUT / "STATUS").write_text("mega stopped by STOP\n")
            delete_pod()
            return
        time.sleep(TICK_S)


if __name__ == "__main__":
    main()
