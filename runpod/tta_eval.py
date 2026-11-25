#!/usr/bin/env python3
"""Test-time variants of finished checkpoints, each on its own held-out fold of the competition studies.

    python tta_eval.py --name tta1 --ckpts r2-cnxs336-f0:r2cnxs3360:0,r2-effv2s336-f1:r2effv2s3361:1,...

For every job:tag:fold it scores the fold's studies once per (window phase, mirror) pair and keeps each pass apart:
$MEGA_JOB_DIR/raptor_tta_<tag>.npz = ids, phases, flips, pred (P, N, 12) float16. The host then composes any plan
from those passes (the leg's two plain passes, mirror on the second pass, one pass, four passes ...) and reads
them on the report ruler (scripts/tta_ruler.py). The mirror is train_knee._mirror applied before normalisation,
exactly as in training with --aug_flip.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_pseudo as ep  # noqa: E402
import train_knee as tk  # noqa: E402

JOB = Path(os.environ.get("MEGA_JOB_DIR", "."))


class EvalWindows(tk.StudyWindows):
    """Deterministic eval windows with an optional left-right mirror of the whole study."""

    def __init__(self, *a, flip=False, out_norm="imagenet", **k):
        super().__init__(*a, norm="none", **k)
        self.flip, self.out_norm = flip, out_norm

    def _centers(self, valid, count):
        self._cs = super()._centers(valid, count)
        return self._cs

    def __getitem__(self, i):
        x, y, w = super().__getitem__(i)
        if self.flip:
            x = tk._mirror(x, self._cs)
        if self.out_norm == "imagenet":
            x = (x - self._mean) / self._std
        return x, y, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpts", required=True, help="comma list of job:tag:fold")
    ap.add_argument("--fold_file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "folds_nonexpert_5.json"))
    ap.add_argument("--phases", default="0,0.25,0.5,0.75")
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--norm", default="imagenet")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ.pop("KNEE_EXT_DIR", None)          # competition corpus only
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = tk._open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    fmap = json.load(open(a.fold_file))["folds"]
    phases = [float(x) for x in a.phases.split(",")]
    plan = [(ph, fl) for fl in (False, True) for ph in phases]
    JOB.mkdir(parents=True, exist_ok=True)
    for spec in a.ckpts.split(","):
        job, tag, fold = spec.split(":")
        sub = [u for u in ids if fmap.get(u, -1) == int(fold)]
        if a.limit:
            sub = sub[:a.limit]
        labels = {u: np.zeros(12, np.float32) for u in sub}
        ck = torch.load(ep.fetch(job, tag), map_location="cpu", weights_only=False)
        bb = tk.build_backbone(ck["arch"], pretrained=False)
        model = tk.RaptorClassifier(bb, F_dim=bb.num_features)
        model.load_state_dict(ck["model"], strict=True)
        model.eval().to(dev)
        res, t0, per = int(ck["res"]), time.time(), []
        for ph, fl in plan:
            ds = EvalWindows(tk.HERE, sub, id2row, labels, res, a.k_eval, train=False, eval_phase=ph, flip=fl,
                             out_norm=a.norm)
            dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=tk.collate)
            out = []
            with torch.no_grad():
                for x, _y, _w, *_ in dl:
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        out.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            per.append(np.concatenate(out).astype(np.float16))
        np.savez(JOB / f"raptor_tta_{tag}.npz", ids=np.array(sub), pred=np.stack(per), fold=int(fold),
                 phases=np.array([p for p, _ in plan], np.float32), flips=np.array([f for _, f in plan]),
                 lab=np.array(tk.LAB))
        print(f"{spec}: {ck['arch']} @{res} | {len(sub)} studies x {len(plan)} passes | {time.time() - t0:.0f}s", flush=True)
        del model, ck
        if dev == "cuda":
            torch.cuda.empty_cache()
    print(f"tta_eval {a.name}: done", flush=True)


if __name__ == "__main__":
    main()
