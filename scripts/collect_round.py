#!/usr/bin/env python3
"""Fetch finished label-round jobs (agent/runs/label_rounds.json) into outputs/runpod_results/<job>/<tag>/.

    python scripts/collect_round.py [--wait-min 110]   # polls every 5 min until all jobs are collected or time is up

Takes the OOF / holdout / summary files from the pod's monitor while it is alive, else from the job's
private Kaggle result dataset. Exits 0 when everything is collected, 2 when time ran out, 3 when a pod
vanished without results (relaunch needed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "agent" / "runs" / "label_rounds.json"
RES = REPO / "outputs" / "runpod_results"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))


def get(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except Exception:
        return None


def files(j: dict) -> list[str]:
    t = j["tag"]
    return [f"raptor_oof_{t}_fold{j['fold']}.npz", f"raptor_holdout_{t}.npz", f"raptor_ft_{t}.json"]


def collect(j: dict) -> str:
    d = RES / j["job"] / j["tag"]
    if all((d / f).exists() and (d / f).stat().st_size > 0 for f in files(j)):
        return "collected"
    base = f"https://{j['pod']}-8000.proxy.runpod.net"
    status = (get(f"{base}/STATUS") or b"").decode("utf-8", "replace").strip()
    if status and not status.startswith(("done", "uploading")):
        return "failed: " + status if "fail" in status.lower() else "running"
    d.mkdir(parents=True, exist_ok=True)
    for f in files(j):
        b = get(f"{base}/{j['tag']}/{f}") if status else None
        if b:
            (d / f).write_bytes(b)
            continue
        subprocess.run([KAGGLE, "datasets", "download", f"girishbose/rsna-knee-rp-{j['job']}-out", "-f", f"{j['tag']}/{f}",
                        "-p", str(d), "-q"], capture_output=True, text=True, timeout=600)
        z = d / (f + ".zip")
        if z.exists():
            subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(d)]); z.unlink()
    if all((d / f).exists() and (d / f).stat().st_size > 0 for f in files(j)):
        return "collected"
    return "running" if status else "vanished"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-min", type=float, default=0)
    a = ap.parse_args()
    t0 = time.time()
    seen_gone: dict[str, int] = {}
    while True:
        jobs = [j for j in json.loads(REG.read_text())["jobs"] if not j.get("done_local")]
        st = {j["job"]: collect(j) for j in jobs}
        print(time.strftime("%H:%M"), json.dumps(st), flush=True)
        for k, v in st.items():   # a pod must be missing on 3 polls (15 min) before it counts as vanished
            seen_gone[k] = seen_gone.get(k, 0) + 1 if v == "vanished" else 0
        if all(v == "collected" for v in st.values()):
            sys.exit(0)
        if any(n >= 3 for n in seen_gone.values()) or any(v.startswith("failed") for v in st.values()):
            sys.exit(3)
        if time.time() - t0 > a.wait_min * 60:
            sys.exit(2)
        time.sleep(300)


if __name__ == "__main__":
    main()
