#!/usr/bin/env python3
"""Score the external (fastMRI) dense80 cache with finished checkpoints: the teacher side of the pseudo-labels.

    python ext_pseudo.py --name effv2s --ckpts r2-effv2s336-f0:r2effv2s3360,r2-effv2s336-f1:r2effv2s3361,...

Each job:tag is fetched from the private result dataset <user>/rsna-knee-rp-<job>-out (<tag>/raptor_ft_<tag>.pt) and
run over every external study with the trainer's own evaluation windows (k_eval windows, shifted passes), at the
checkpoint's resolution. Writes $MEGA_JOB_DIR/raptor_ext_<name>.npz: ids, pred = mean over checkpoints (N, 12),
per_ckpt (C, N, 12) float16. Predictions only - no image leaves the pod. The host turns them into training targets
on the scale of the competition labels (scripts/make_ext_labels.py).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_knee as tk  # noqa: E402

USER = os.environ.get("KAGGLE_USER", "girishbose")
CK = Path("/workspace/ckpt")
JOB = Path(os.environ.get("MEGA_JOB_DIR", "."))


def fetch(job: str, tag: str) -> Path:
    local = Path(os.environ.get("MEGA_OUT_DIR", "/workspace/out")) / job / tag / f"raptor_ft_{tag}.pt"
    if local.exists():                  # trained on this pod: no download
        return local
    d = CK / job
    dest = d / f"raptor_ft_{tag}.pt"
    if dest.exists():
        return dest
    d.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        path = f"{tag}/raptor_ft_{tag}.pt" if i % 2 == 0 else f"raptor_ft_{tag}.pt"   # flattened single-folder datasets
        subprocess.run(["kaggle", "datasets", "download", f"{USER}/rsna-knee-rp-{job}-out", "-f", path,
                        "-p", str(d), "-q"], capture_output=True, text=True, timeout=1800)
        for z in d.glob("*.zip"):
            with zipfile.ZipFile(z) as f:
                f.extractall(d)
            z.unlink()
        if dest.exists():
            return dest
        time.sleep(30)
    raise SystemExit(f"checkpoint {job}:{tag} could not be downloaded")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpts", required=True, help="comma list of job:tag")
    ap.add_argument("--ext_dir", default="/workspace/fastmri/dense80")
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--norm", default="imagenet")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ["KNEE_EXT_DIR"] = a.ext_dir
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = tk._open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    ext = [u for u in ids if u.startswith("fm_")]
    if a.limit:
        ext = ext[:a.limit]
    labels = {u: np.zeros(12, np.float32) for u in ext}
    print(f"{len(ext)} external studies | {len(ids)} rows in the corpus | device {dev}", flush=True)
    per = []
    for spec in a.ckpts.split(","):
        job, tag = spec.split(":")
        ck = torch.load(fetch(job, tag), map_location="cpu", weights_only=False)
        bb = tk.build_backbone(ck["arch"], pretrained=False)
        model = tk.RaptorClassifier(bb, F_dim=bb.num_features)
        model.load_state_dict(ck["model"], strict=True)
        model.eval().to(dev)
        res, t0, acc = int(ck["res"]), time.time(), None
        for p in range(a.passes):
            ds = tk.StudyWindows(tk.HERE, ext, id2row, labels, res, a.k_eval, train=False, norm=a.norm,
                                 eval_phase=0.0 if a.passes == 1 else p / a.passes)
            dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=tk.collate)
            out = []
            with torch.no_grad():
                for x, _y, _w, *_ in dl:
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        out.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            out = np.concatenate(out)
            acc = out if acc is None else acc + out
        per.append((acc / a.passes).astype(np.float16))
        print(f"{spec}: {ck['arch']} @{res} | {time.time() - t0:.0f}s | mean prob {per[-1].astype(np.float32).mean(0).round(2).tolist()}", flush=True)
        del model, ck
        if dev == "cuda":
            torch.cuda.empty_cache()
    per = np.stack(per)
    JOB.mkdir(parents=True, exist_ok=True)
    np.savez(JOB / f"raptor_ext_{a.name}.npz", ids=np.array(ext), pred=per.astype(np.float32).mean(0), per_ckpt=per,
             ckpts=np.array(a.ckpts.split(",")), lab=np.array(tk.LAB))
    print(f"wrote raptor_ext_{a.name}.npz: {per.shape}", flush=True)


if __name__ == "__main__":
    main()
