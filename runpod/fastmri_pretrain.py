#!/usr/bin/env python3
"""Family B: localized pretraining of the Family A tile encoder on fastMRI knee + fastMRI+ boxes.

Stage 1 (prep, idempotent): download the NYU single-coil knee tarballs (train + val, ~107 GB) with
Range resume, stream-extract each .h5, keep only ``reconstruction_esc`` (S,320,320), flip up/down
into the fastMRI+ annotation frame (fastMRI+ README: pixel arrays were flipped up/down for labelling),
normalise per volume (2-98 %), resize 320 -> 336 (both are ~140 mm FOV, = dense80 coronal geometry)
and store uint8 volumes under /workspace/fastmri/vols (OUTSIDE the uploaded out dir: the DSA forbids
redistributing the data; only encoder weights leave the pod). Tarballs are deleted after extraction.

Links come from env FASTMRI_LINKS (newline-separated) and are never printed; errors are reported
by type / HTTP status only (the pod log is served over the proxy).

Stage 2 (pretrain): 2.5D coronal tiles cut exactly like Family A's (full 320 crop, or a medial/lateral
half 384x288), ConvNeXt feature map -> 1x1 conv -> per-class heatmap (stride 32), BCE vs rasterised
boxes + tile-level BCE on the log-sum-exp pooled logit. Output: raptor_fmri_<tag>.pt (backbone
state_dict, timm keys) + raptor_ft_<tag>.json (val AUC per class).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
FM = os.environ.get("FASTMRI_DIR", "/workspace/fastmri")
IMG = 336
CLASSES = {
    "meniscus": ["Meniscus Tear", "Displaced Meniscal Tissue"],
    "cart_partial": ["Cartilage - Partial Thickness loss/defect"],
    "cart_full": ["Cartilage - Full Thickness loss/defect"],
    "effusion": ["Joint Effusion"],
    "bone_edema": ["Bone- Subchondral edema", "Bone-Fracture/Contusion/dislocation"],
    "cyst": ["Periarticular cysts"],
    "acl": ["Ligament - ACL Low Grade sprain", "Ligament - ACL High Grade Sprain"],
    "ligament": ["Ligament - MCL Low-Mod Grade Sprain", "Ligament - MCL High Grade sprain",
                 "LCL Complex - Low-Mod Grade Sprain", "LCL Complex- High Grade Sprain",
                 "Ligament - PCL Low-Mod grade sprain", "Ligament - PCL High Grade",
                 "Patellar Retinaculum - High grade sprain"],
    "other": ["Bone - Lesion", "Soft Tissue Lesion", "Muscle Strain", "Joint Bodies"],
}
CNAMES = list(CLASSES)
LABEL2C = {lab: i for i, c in enumerate(CNAMES) for lab in CLASSES[c]}


def _find(name):
    for root in (HERE, "/workspace/bundle", "/kaggle/input", "/workspace/input"):
        for d, _, fs in os.walk(root) if os.path.isdir(root) else []:
            if name in fs:
                return os.path.join(d, name)
    return None


# ------------------------------------------------------------------ stage 1: prep
def _links() -> list[str]:
    raw = os.environ.get("FASTMRI_LINKS", "")
    links = [s.strip() for s in raw.splitlines() if s.strip().startswith("http")]
    if not links:
        raise SystemExit("FASTMRI_LINKS env missing (links are never stored in the bundle)")
    return links


def _name(url: str) -> str:
    return os.path.basename(url.split("?", 1)[0])


def download(url: str, dest: str, tries: int = 12) -> None:
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total = None
    for t in range(1, tries + 1):
        pos = os.path.getsize(dest) if os.path.exists(dest) else 0
        if total is not None and pos >= total:
            return
        try:
            with requests.get(url, headers={"Range": f"bytes={pos}-"} if pos else {}, stream=True,
                              timeout=600) as r:
                if r.status_code == 416:            # already complete
                    return
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code == 200:
                    pos = 0
                n = int(r.headers.get("Content-Length", 0) or 0)
                total = pos + n if n else None
                with open(dest, "ab" if pos else "wb") as f:
                    for chunk in r.iter_content(16 << 20):
                        f.write(chunk)
            size = os.path.getsize(dest)
            print(f"  {os.path.basename(dest)}: {size / 1e9:.1f} GB", flush=True)
            if total is None or size >= total:
                return
        except Exception as e:  # never echo the exception text: it can contain the signed URL
            code = getattr(getattr(e, "response", None), "status_code", "")
            print(f"  try {t}/{tries} {os.path.basename(dest)}: {type(e).__name__} {code}", flush=True)
            time.sleep(min(60, 5 * t))
    raise SystemExit(f"download failed: {os.path.basename(dest)}")


def to_vol(recon: np.ndarray) -> np.ndarray:
    v = np.flip(recon.astype(np.float32), axis=1)          # up/down flip -> fastMRI+ label frame
    lo, hi = np.percentile(v, [2.0, 98.0])
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    t = F.interpolate(torch.from_numpy(np.ascontiguousarray(v))[:, None], size=(IMG, IMG),
                      mode="bilinear", align_corners=False)[:, 0]
    return (t.clamp(0, 1).numpy() * 255).astype(np.uint8)


def extract(tar_path: str, vol_dir: str) -> int:
    """Stream the tarball through xz; keep reconstruction_esc only. Returns #volumes written."""
    import h5py
    os.makedirs(vol_dir, exist_ok=True)
    import shutil
    xz = tar_path.endswith(".xz")
    proc = subprocess.Popen(["xz", "-dc", "-T0", tar_path], stdout=subprocess.PIPE) \
        if xz and shutil.which("xz") else None
    src = proc.stdout if proc else open(tar_path, "rb")
    n = 0
    with tarfile.open(fileobj=src, mode="r|" if (proc or not xz) else "r|xz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".h5"):
                continue
            stem = os.path.basename(m.name)[:-3]
            dest = os.path.join(vol_dir, stem + ".npy")
            buf = tf.extractfile(m).read()
            if os.path.exists(dest):
                continue
            with h5py.File(io.BytesIO(buf), "r") as h:
                if "reconstruction_esc" not in h:
                    continue
                np.save(dest + ".tmp.npy", to_vol(h["reconstruction_esc"][()]))
            os.replace(dest + ".tmp.npy", dest)
            n += 1
            if n % 50 == 0:
                print(f"  extracted {n} volumes from {os.path.basename(tar_path)}", flush=True)
    if proc:
        proc.wait()
        if proc.returncode:
            raise SystemExit(f"xz failed on {os.path.basename(tar_path)} ({proc.returncode})")
    return n


def prep() -> str:
    vol_dir = os.path.join(FM, "vols")
    for url in _links():
        name = _name(url)
        done = os.path.join(FM, f"{name}.done")
        if os.path.exists(done):
            print(f"[prep] {name} already extracted", flush=True)
            continue
        tar = os.path.join(FM, "raw", name)
        print(f"[prep] downloading {name}", flush=True)
        t0 = time.time()
        download(url, tar)
        print(f"[prep] extracting {name} ({time.time() - t0:.0f}s download)", flush=True)
        n = extract(tar, vol_dir)
        open(done, "w").write(str(n))
        os.remove(tar)                                     # free the volume; data stays private
        print(f"[prep] {name}: {n} volumes in {time.time() - t0:.0f}s", flush=True)
    return vol_dir


# ------------------------------------------------------------------ stage 2: data
def load_boxes(csv_path: str, files: set[str]) -> dict:
    d = pd.read_csv(csv_path)
    d = d[d.label.isin(LABEL2C) & d.x.notna()]
    s = IMG / 320.0
    boxes: dict = {}
    for r in d.itertuples():
        if r.file in files:
            boxes.setdefault((r.file, int(r.slice)), []).append(
                (LABEL2C[r.label], r.x * s, r.y * s, (r.x + r.width) * s, (r.y + r.height) * s))
    return boxes


class Tiles(Dataset):
    """(volume, centre slice) -> tile (3,h,w) + heatmap target (K,h/32,w/32).

    mode "full": central 320 crop (Family A sag/ax/full geometry); "half": central 76% rows ×
    57% width, side a/b, resized to 384x288 (Family A coronal geometry).
    """

    def __init__(self, vol_dir, items, boxes, mode, train, full=320, half_hw=(384, 288)):
        self.vol_dir, self.items, self.boxes = vol_dir, items, boxes
        self.mode, self.train, self.full, self.half_hw = mode, train, full, half_hw
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def _vol(self, f):
        if f not in self.cache:
            self.cache[f] = np.load(os.path.join(self.vol_dir, f + ".npy"), mmap_mode="r")
        return self.cache[f]

    def __getitem__(self, i):
        f, c, side = self.items[i]
        v = self._vol(f)
        c = int(np.clip(c, 1, len(v) - 2))
        tri = np.asarray(v[c - 1:c + 2], np.float32) / 255.0
        H = W = IMG
        j = (lambda s: random.randint(-s, s)) if self.train else (lambda s: 0)
        if self.mode == "full":
            y0 = int(np.clip((H - self.full) // 2 + j(8), 0, H - self.full))
            x0 = int(np.clip((W - self.full) // 2 + j(8), 0, W - self.full))
            y1, x1, out_hw = y0 + self.full, x0 + self.full, (self.full, self.full)
        else:
            hh, ww = int(round(H * 0.76)), int(round(W * 0.57))
            y0 = int(np.clip((H - hh) // 2 + j(10), 0, H - hh))
            x0 = int(np.clip((0 if side == 0 else W - ww) + j(8), 0, W - ww))
            y1, x1, out_hw = y0 + hh, x0 + ww, self.half_hw
        t = torch.from_numpy(np.ascontiguousarray(tri[:, y0:y1, x0:x1]))
        t = F.interpolate(t[None], size=out_hw, mode="bilinear", align_corners=False)[0]
        sy, sx = out_hw[0] / (y1 - y0), out_hw[1] / (x1 - x0)
        gh, gw = out_hw[0] // 32, out_hw[1] // 32
        tgt = np.zeros((len(CNAMES), gh, gw), np.float32)
        flip = self.train and random.random() < 0.5
        for k, bx0, by0, bx1, by1 in self.boxes.get((f, c), []):
            ax0, ax1 = (bx0 - x0) * sx, (bx1 - x0) * sx
            ay0, ay1 = (by0 - y0) * sy, (by1 - y0) * sy
            if flip:
                ax0, ax1 = out_hw[1] - ax1, out_hw[1] - ax0
            cx0, cx1 = int(np.floor(max(ax0, 0) / 32)), int(np.ceil(min(ax1, out_hw[1]) / 32))
            cy0, cy1 = int(np.floor(max(ay0, 0) / 32)), int(np.ceil(min(ay1, out_hw[0]) / 32))
            if cx1 > cx0 and cy1 > cy0:
                tgt[k, cy0:cy1, cx0:cx1] = 1.0
        if flip:
            t = t.flip(-1)
        if self.train:
            t = (t * (1 + (random.random() - 0.5) * 0.25) + (random.random() - 0.5) * 0.06).clamp(0, 1)
        return t, torch.from_numpy(tgt)


def make_items(vol_dir, files, boxes, per_vol_neg=6, seed=0):
    """All annotated (file, slice) pairs ×2 sides + random negative slices per volume."""
    rng = random.Random(seed)
    items = []
    pos = {}
    for (f, s) in boxes:
        pos.setdefault(f, set()).add(s)
    for f in files:
        n = np.load(os.path.join(vol_dir, f + ".npy"), mmap_mode="r").shape[0]
        ps = sorted(s for s in pos.get(f, ()) if 1 <= s <= n - 2)
        neg = [s for s in range(2, n - 2) if s not in pos.get(f, ())]
        rng.shuffle(neg)
        for s in ps + neg[:per_vol_neg]:
            items += [(f, s, 0), (f, s, 1)]
    return items


class HeatNet(nn.Module):
    def __init__(self, arch, pretrained=True, grad_ckpt=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
        if grad_ckpt:
            self.backbone.set_grad_checkpointing(True)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        self.head = nn.Conv2d(self.backbone.num_features, len(CNAMES), 1)

    def forward(self, x):
        fm = self.backbone.forward_features((x - self.mean) / self.std)
        if fm.ndim == 4 and fm.shape[1] != self.backbone.num_features:   # NHWC backbones
            fm = fm.permute(0, 3, 1, 2)
        return self.head(fm)                                          # (B,K,h,w)


def lse_pool(m, r=4.0):
    B, K = m.shape[:2]
    return torch.logsumexp(r * m.flatten(2), -1) / r - np.log(m.shape[2] * m.shape[3]) / r


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="ignored (run_job.sh passes it to every arm)")
    ap.add_argument("--tag", default="fmri")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--vol_dir", default=None, help="skip prep; use these extracted volumes")
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--arch", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--pos_w", type=float, default=4.0)
    ap.add_argument("--per_vol_neg", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    vol_dir = a.vol_dir or prep()
    files = sorted(f[:-4] for f in os.listdir(vol_dir) if f.endswith(".npy") and ".tmp" not in f)
    boxes = load_boxes(a.boxes or _find("knee.csv"), set(files))
    rng = random.Random(a.seed)
    shuf = files[:]; rng.shuffle(shuf)
    nv = max(1, int(len(shuf) * a.val_frac))
    val_f, tr_f = shuf[:nv], shuf[nv:]
    if a.smoke:
        tr_f, val_f, a.epochs = tr_f[:20], val_f[:6], 1
    tr_items = make_items(vol_dir, tr_f, boxes, a.per_vol_neg, a.seed)
    va_items = make_items(vol_dir, val_f, boxes, a.per_vol_neg, a.seed + 1)
    print(f"device {dev} | volumes {len(files)} (train {len(tr_f)}, val {len(val_f)}) | boxed slices "
          f"{len(boxes)} | train tiles {len(tr_items)} | val tiles {len(va_items)}", flush=True)

    model = HeatNet(a.arch, pretrained=not a.no_pretrained, grad_ckpt=a.grad_ckpt).to(dev)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.lr * 5}], weight_decay=a.wd)
    loaders = {}
    for mode in ("full", "half"):
        ds = Tiles(vol_dir, tr_items, boxes, mode, True)
        loaders[mode] = DataLoader(ds, batch_size=a.bs, shuffle=True, drop_last=True,
                                   num_workers=a.workers, persistent_workers=a.workers > 0)
    steps = a.epochs * sum(len(l) for l in loaders.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 5], total_steps=max(steps, 1),
                                                pct_start=0.08)
    pw = torch.tensor(a.pos_w, device=dev)

    def evaluate():
        from sklearn.metrics import roc_auc_score
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for mode in ("full", "half"):
                dl = DataLoader(Tiles(vol_dir, va_items, boxes, mode, False), batch_size=a.bs,
                                num_workers=a.workers)
                for x, t in dl:
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        m = model(x.to(dev)).float()
                    P.append(torch.sigmoid(lse_pool(m)).cpu().numpy())
                    Y.append(t.flatten(2).amax(-1).numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        out = {c: float(roc_auc_score(Y[:, k], P[:, k])) for k, c in enumerate(CNAMES)
               if 0 < Y[:, k].sum() < len(Y)}
        return out

    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        its = {m: iter(l) for m, l in loaders.items()}
        order = [m for m, l in loaders.items() for _ in range(len(l))]
        random.shuffle(order)
        tot = 0.0
        for i, mode in enumerate(order):
            x, t = next(its[mode])
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                m = model(x)
            m = m.float()
            if m.shape[-2:] != t.shape[-2:]:
                m = F.interpolate(m, size=t.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(m, t, pos_weight=pw) + \
                F.binary_cross_entropy_with_logits(lse_pool(m), t.flatten(2).amax(-1), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step(); tot += loss.item()
            if i % 200 == 0:
                print(f"  ep{ep} it{i}/{len(order)} loss {tot / (i + 1):.4f} | {time.time() - t0:.0f}s", flush=True)
        au = evaluate()
        rec = {"ep": ep, "loss": tot / max(1, len(order)), "sec": round(time.time() - t0),
               "val_macro": float(np.mean(list(au.values()))) if au else None,
               "val": {k: round(v, 4) for k, v in au.items()}}
        hist.append(rec)
        print(f"ep{ep} {json.dumps(rec)}", flush=True)
        torch.save({"backbone": {k: v.cpu() for k, v in model.backbone.state_dict().items()},
                    "arch": a.arch, "classes": CNAMES, "epoch": ep},
                   os.path.join(a.out, f"raptor_fmri_{a.tag}.pt"))
    json.dump({"tag": a.tag, "args": vars(a), "hist": hist, "n_volumes": len(files),
               "classes": CNAMES}, open(os.path.join(a.out, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
