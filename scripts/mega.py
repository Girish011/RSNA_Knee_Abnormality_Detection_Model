#!/usr/bin/env python3
"""Host side of the multi-GPU queue pod (runpod/mega.py): build / push the queue dataset, launch the pod, add arms.

    python scripts/mega.py init                       # (re)build outputs/mega_queue and push it as the queue dataset
    python scripts/mega.py launch --gpus 4 [--volume 800] [--fastmri]
    python scripts/mega.py add --round 2 --labels labels_r1.parquet --families swint224 --folds 1,2,3,4
    python scripts/mega.py task --job fm-download --cmdline "python fastmri_dicom_cache.py download" [--needs PATH]
    python scripts/mega.py push                       # after editing outputs/mega_queue by hand (code hot-fix)
    python scripts/mega.py status
    python scripts/mega.py stop | unstop              # STOP file: the pod deletes itself once nothing is running

The queue dataset girishbose/rsna-knee-mega-queue is the pod's remote control (it is polled every 3 min). Training
arms are registered in agent/runs/label_rounds.json with "mega": true; scripts/collect_round.py fetches them from
their result datasets. FASTMRI_LINKS (the two DICOM links) go to the pod as an env var only, never into the dataset.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
QD = REPO / "outputs" / "mega_queue"
REG = REPO / "agent" / "runs" / "label_rounds.json"
SLUG = "girishbose/rsna-knee-mega-queue"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))
CODE = ["mega.py", "mega_boot.sh", "train_knee.py", "fetch_dense80.py", "fastmri_dicom_cache.py", "ext_pseudo.py", "tta_eval.py"]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
BOOT = """set -u; mkdir -p /workspace/out /workspace/queue ~/.kaggle
cd /workspace/out && nohup python -m http.server 8000 >/dev/null 2>&1 &
printf %s "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
pip install -q kaggle
for i in $(seq 1 40); do
  kaggle datasets download {slug} -p /workspace/queue --unzip --force -q && break
  echo "queue not ready, retry $i" >> /workspace/out/job.log; sleep 30
done
if [ ! -f /workspace/queue/mega_boot.sh ]; then
  echo "QUEUE UNAVAILABLE; deleting pod" > /workspace/out/STATUS; sleep 60
  curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
fi
bash /workspace/queue/mega_boot.sh
sleep infinity"""


def recipes() -> dict:
    spec = importlib.util.spec_from_file_location("launch_round", REPO / "scripts" / "launch_round.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.RECIPES


def sync_code() -> None:
    QD.mkdir(parents=True, exist_ok=True)
    for f in CODE:
        shutil.copy2(REPO / "runpod" / f, QD / f)
    for f in ("folds_nonexpert_5.json", "expert_holdout.json", "knee_sides.json"):
        shutil.copy2(REPO / "agent" / "protocol" / f, QD / f)
    pd.read_csv(REPO / "data" / "raw" / "train.csv")[["StudyInstanceUID", *LAB]].to_csv(QD / "train.csv", index=False)
    (QD / "arms.jsonl").touch()
    (QD / "dataset-metadata.json").write_text(json.dumps({"title": "rsna-knee-mega-queue", "id": SLUG, "licenses": [{"name": "CC0-1.0"}]}))


def push(msg: str) -> None:
    exists = subprocess.run([KAGGLE, "datasets", "status", SLUG], capture_output=True, text=True, timeout=120)
    cmd = ([KAGGLE, "datasets", "version", "-p", str(QD), "-m", msg[:60], "-q"] if exists.returncode == 0 and "ready" in exists.stdout.lower()
           else [KAGGLE, "datasets", "create", "-p", str(QD), "-q"])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print("queue pushed" if r.returncode == 0 else f"PUSH FAILED: {(r.stderr or r.stdout)[-300:]}")
    if r.returncode != 0:
        raise SystemExit(1)


def arms() -> list[dict]:
    return [json.loads(x) for x in (QD / "arms.jsonl").read_text().splitlines() if x.strip() and not x.startswith("#")]


def append(arm: dict) -> bool:
    if any(a["job"] == arm["job"] for a in arms()):
        print(f"{arm['job']}: already queued - skipped")
        return False
    with open(QD / "arms.jsonl", "a") as f:
        f.write(json.dumps(arm) + "\n")
    return True


def pod_id() -> str:
    return json.loads((QD.parent / "mega_pod.json").read_text())["pod"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "launch", "add", "task", "push", "status", "stop", "unstop"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--volume", type=int, default=800)
    ap.add_argument("--fastmri", action="store_true", help="pass the DICOM links to the pod as FASTMRI_LINKS")
    ap.add_argument("--round", type=int)
    ap.add_argument("--labels")
    ap.add_argument("--families", default="")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--job")
    ap.add_argument("--cmdline", "--cmd-line", dest="cmdline")
    ap.add_argument("--needs", action="append", default=[])
    ap.add_argument("--gpu-task", action="store_true", help="task: give the shell task one GPU (default: CPU task)")
    ap.add_argument("--upload", action="store_true", help="task: upload the job dir (raptor_* / logs) as rsna-knee-rp-<job>-out")
    a = ap.parse_args()

    if a.cmd == "init":
        sync_code()
        push("init")
    elif a.cmd == "push":
        sync_code()
        push("update")
    elif a.cmd in ("stop", "unstop"):
        (QD / "STOP").write_text("stop\n") if a.cmd == "stop" else (QD / "STOP").unlink(missing_ok=True)
        push(a.cmd)
    elif a.cmd == "task":
        sync_code()
        if append({"job": a.job, "cmd": a.cmdline, "gpu": int(a.gpu_task), **({"needs": a.needs} if a.needs else {}),
                   **({"upload": True} if a.upload else {})}):
            push(f"task {a.job}")
    elif a.cmd == "add":
        sync_code()
        assert (REPO / "data" / "processed" / a.labels).exists(), a.labels
        shutil.copy2(REPO / "data" / "processed" / a.labels, QD / a.labels)
        rec, reg, new = recipes(), json.loads(REG.read_text()), []
        for fam in a.families.split(","):
            r = rec[fam]
            assert r["trainer"] == "train_knee.py", "the queue pod runs train_knee.py arms only"
            for fold in (int(x) for x in a.folds.split(",")):
                job, tag = f"r{a.round}-{fam}-f{fold}", f"r{a.round}{fam}{fold}"
                if any(j["job"] == job for j in reg["jobs"]):
                    print(f"{job}: already in the registry - skipped")
                    continue
                if append({"job": job, "tag": tag, "labels": a.labels, "args": r["args"].format(fold=fold),
                           **({"needs": a.needs} if a.needs else {})}):
                    new.append({"job": job, "pod": "", "tag": tag, "family": fam, "fold": fold, "round": a.round,
                                "epochs": r["epochs"], "price": 0.49, "launched": time.strftime("%Y-%m-%dT%H:%M:%S"), "mega": True})
                    if fam not in [f["key"] for f in reg["families"]]:
                        reg["families"].append({"key": fam, "name": r["name"], "note": r["note"], "globs": [f"r{a.round}-{fam}-f*/*"]})
        if new:
            push(f"add {len(new)} arms")
            reg = json.loads(REG.read_text())
            reg["jobs"] += new
            REG.write_text(json.dumps(reg, indent=1))
            print("queued:", [j["job"] for j in new])
    elif a.cmd == "launch":
        from rsna_knee.agent import runpod
        env = {"KAGGLE_API_TOKEN": Path("~/.kaggle/access_token").expanduser().read_text().strip(),
               "RUNPOD_API_KEY": runpod.secret("RUNPOD_API_KEY"), "QUEUE_DATASET": SLUG}
        if a.fastmri:
            from urllib.parse import urlparse
            by_name = {}
            for u in Path("~/.rsna_ext/fastmri_links.txt").expanduser().read_text().split():
                name = Path(urlparse(u).path).name
                if u.startswith("http") and "DICOM" in name:
                    by_name[name] = u
            links = list(by_name.values())
            assert len(links) == 2, f"expected the two DICOM batch links, found {sorted(by_name)}"
            env["FASTMRI_LINKS"] = " ".join(links)
        spec = {"name": f"rsna-mega-{a.gpus}gpu", "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                "computeType": "GPU", "gpuCount": a.gpus, "cloudType": "SECURE", "gpuTypeIds": [runpod.GPU_A40],
                "containerDiskInGb": 60, "volumeInGb": a.volume, "volumeMountPath": "/workspace", "ports": ["8000/http"],
                "allowedCudaVersions": ["13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4"],
                "env": env, "dockerStartCmd": ["bash", "-c", BOOT.replace("{slug}", SLUG)]}
        pod = runpod.create_pod(spec)
        time.sleep(20)
        live = runpod.get_pod(pod["id"])
        (QD.parent / "mega_pod.json").write_text(json.dumps({"pod": pod["id"], "gpus": a.gpus, "costPerHr": live.get("costPerHr"),
                                                              "launched": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        print("mega pod", pod["id"], "|", live.get("desiredStatus"), "| $/h", live.get("costPerHr"),
              "| monitor", runpod.proxy_url(pod["id"]))
    elif a.cmd == "status":
        base = f"https://{pod_id()}-8000.proxy.runpod.net"
        for path in ("STATUS", "mega_state.json"):
            try:
                req = urllib.request.Request(f"{base}/{path}", headers={"User-Agent": "curl/8"})
                body = urllib.request.urlopen(req, timeout=30).read().decode()
            except Exception as e:  # noqa: BLE001
                print(path, "unavailable:", type(e).__name__)
                continue
            if path == "STATUS":
                print(body.strip())
            else:
                for job, s in json.loads(body).items():
                    t1 = s.get("t1") or time.time()
                    print(f"  {job:24s} {s['status']:10s} gpu {s.get('gpu')} | try {s['tries']} | {(t1 - s['t0']) / 60:.0f} min")


if __name__ == "__main__":
    main()
