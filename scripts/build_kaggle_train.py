#!/usr/bin/env python3
"""Kaggle T4 x 2 training kernel: one train_knee.py arm per GPU (the RunPod pods are closed from 7 Oct).

    python scripts/build_kaggle_train.py --name ktrain-a --labels labels_r4.parquet \
        --arm "r4-kcnxtseq-f0|r4kcnxtseq0|<train_knee args>" --arm "r4-kcnxtmil-f0|r4kcnxtmil0|<args>"
    kaggle kernels push -p outputs/kernels/ktrain-a
    python scripts/build_kaggle_train.py --name ktrain-a --collect      # when the run is complete

Inputs: the dense80 cache (outputs of girishbose/rsna-knee-dense80-s0..3), the queue dataset
girishbose/rsna-knee-mega-queue (train_knee.py, label files, folds, expert holdout; refresh it with
`python scripts/mega.py push`) and girishbose/rsna-knee-timm-pretrained (<arch>.pth ImageNet weights, from
outputs/timm_pretrained). Internet stays OFF: a kernel with internet on and kernel-output sources dies at start with
an empty log (6 Oct), and the 40 GB of shard outputs take ~10 min to mount before the script starts. Arms run with `--amp fp16` (a T4 has no native bfloat16) and `--workers 2` (4 vCPUs).
Results land in /kaggle/working/<tag>/ and are collected into outputs/runpod_results/<job>/<tag>/, where
oof_ruler.py / family_ruler.py read them like pod results.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "girishbose"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))
SRC = r'''
import os, shutil, subprocess, sys, time
W = "/kaggle/working"
ARMS = __ARMS__            # [(job, tag, args)]
LABELS = __LABELS__


def find(name):
    for d, _, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(d, name)
    raise SystemExit(name + " not found under /kaggle/input")


subprocess.call("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
trainer = find("train_knee.py")
proto = f"--labels {find(LABELS)} --fold_file {find('folds_nonexpert_5.json')} --holdout {find('expert_holdout.json')}"
procs = []
for g, (job, tag, args) in enumerate(ARMS):
    d = f"{W}/{tag}"
    os.makedirs(d, exist_ok=True)
    shutil.copy2(trainer, d + "/train_knee.py")
    arch = args.split("--arch ")[1].split()[0]
    cmd = f"{sys.executable} -u train_knee.py --tag {tag} {proto} {args} --bb_file {arch}.pth --amp fp16 --workers 2"
    print(f"[ktrain] gpu {g} | {job} | {cmd}", flush=True)
    procs.append((tag, subprocess.Popen(cmd, shell=True, cwd=d, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                                        stdout=open(d + "/train.log", "w"), stderr=subprocess.STDOUT)))
seen = {tag: 0 for tag, _ in procs}
t0 = time.time()
while True:
    alive = False
    for tag, p in procs:
        alive |= p.poll() is None
        try:
            lines = open(f"{W}/{tag}/train.log").read().splitlines()
        except OSError:
            lines = []
        for ln in lines[seen[tag]:]:
            if "Warning" not in ln and "warn" not in ln and ln.strip():
                print(f"[{tag}] {ln}", flush=True)
        seen[tag] = len(lines)
    if not alive:
        break
    time.sleep(60)
for tag, p in procs:
    print(f"[ktrain] {tag} exit {p.returncode} | {(time.time() - t0) / 60:.0f} min", flush=True)
    os.remove(f"{W}/{tag}/train_knee.py")
if any(p.returncode for _, p in procs):
    raise SystemExit("an arm failed")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", default="labels_r4.parquet")
    ap.add_argument("--arm", action="append", default=[], help="job|tag|train_knee args (max two: one per T4)")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    d = REPO / "outputs" / "kernels" / a.name
    if a.collect:
        arms = json.loads((d / "arms.json").read_text())
        tmp = d / "_out"
        shutil.rmtree(tmp, ignore_errors=True)
        r = subprocess.run([KAGGLE, "kernels", "output", f"{USER}/{a.name}", "-p", str(tmp)], capture_output=True, text=True)
        print((r.stdout or r.stderr)[-300:])
        for job, tag, _ in arms:
            dst = REPO / "outputs" / "runpod_results" / job / tag
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in (tmp / tag).glob("*"):
                if f.suffix != ".pt":          # weights stay on Kaggle (the kernel output is the weight source)
                    shutil.copy2(f, dst / f.name); n += 1
            print(f"{job}: {n} files -> {dst}")
        return
    assert 1 <= len(a.arm) <= 2, "one arm per T4"
    arms = [tuple(x.split("|", 2)) for x in a.arm]
    d.mkdir(parents=True, exist_ok=True)
    code = a.name.replace("-", "_") + ".py"
    src = SRC.replace("__ARMS__", repr(arms)).replace("__LABELS__", repr(a.labels))
    compile(src, code, "exec")
    (d / code).write_text(src)
    (d / "arms.json").write_text(json.dumps(arms, indent=1))
    meta = {"id": f"{USER}/{a.name}", "title": a.name, "code_file": code, "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
            "dataset_sources": [f"{USER}/rsna-knee-mega-queue", f"{USER}/rsna-knee-timm-pretrained"],
            "kernel_sources": [f"{USER}/rsna-knee-dense80-s{i}" for i in range(4)],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {d} | {len(arms)} arms | push: kaggle kernels push -p {d}")


if __name__ == "__main__":
    main()
