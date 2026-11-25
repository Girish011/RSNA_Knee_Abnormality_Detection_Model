#!/usr/bin/env python3
"""Report-supervised MRI reader. Fixed EMA deployment; no image-teacher weights.

Inputs: dense512 shards (including geometry and series JSON), definition_v1.npz,
train.csv and train_series.csv. The nonexpert fold metric measures REPORT agreement,
not expert-image accuracy. --smoke --dummy-encoder creates a tiny synthetic corpus
when --data-root is omitted. --predict uses the checkpoint's exact view protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import definition_geometry as geometry

LAB = geometry.LAB
BLOCKS = list(geometry.BLOCKS.values())
PLANES = {"sag": 0, "cor": 1, "ax": 2}
PROTOCOL_VERSION = 1
STOP = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_torch(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def valid_centres(mask, lo, hi):
    """No triplet may cross a sequence boundary or include a missing slice."""
    c = np.arange(lo + 1, hi - 1)
    return c[mask[c - 1] & mask[c] & mask[c + 1]]


def choose_centres(candidates, n, *, rng=None, phase=0, phases=1):
    """Stratified random training; distinct interleaved deterministic eval grids.

    With sufficient candidates, passes are disjoint. Short stacks are sampled at
    most once per pass; missing slots are masked padding, never duplicate tiles.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if not len(candidates):
        return candidates
    if rng is not None:
        n = min(n, len(candidates))
        bounds = np.linspace(0, len(candidates), n + 1).astype(int)
        return candidates[[rng.integers(bounds[i], bounds[i + 1]) for i in range(n)]]
    total = min(n * phases, len(candidates))
    ix = np.floor((np.arange(total) + .5) * len(candidates) / total).astype(int)
    # When a sequence is shorter than the number of passes each pass still sees it.
    take = ix[phase::phases]
    if not len(take):
        take = ix[:1]
    return candidates[take]


class Corpus(geometry.Corpus):
    """Geometry helper plus the ACTUAL sequence attributes from competition CSV."""
    def __init__(self, root, n_shards=8, series_csv=None):
        super().__init__(n_shards, str(root))
        # Field-centre estimates may only use valid geometry. This is a geometric
        # prior, not a learned joint/patella localizer.
        center = geometry.slice_centres(self.geom)
        good = self.masks & (self.geom[..., 19] > .5) & np.isfinite(center).all(-1)
        for row in range(len(self.ids)):
            fallback = center[row, good[row]].mean(0) if good[row].any() else np.zeros(3)
            for axis, blocks in enumerate(((0, 1), (2, 3), (4,))):
                ix = np.concatenate([np.arange(BLOCKS[b][0], BLOCKS[b][1]) for b in blocks])
                ix = ix[good[row, ix]]
                self.P[row, axis] = center[row, ix, axis].mean() if len(ix) else fallback[axis]
        rows = {}
        if series_csv:
            df = pd.read_csv(series_csv, dtype={"SeriesInstanceUID": str})
            rows = df.set_index("SeriesInstanceUID").to_dict("index")
        self.sequence = {}
        for i in range(n_shards):
            info = json.loads(Path(self._p(f"series_s{i}.json")).read_text())
            for uid, series in info.items():
                self.sequence[uid] = {}
                sides = {str(s.get("Laterality") or s.get("ImageLaterality") or "")[:1].upper() for s in series} & {"L", "R"}
                if len(sides) == 1:
                    self.tag_left[uid] = float(next(iter(sides)) == "L")
                for s in series:
                    r = rows.get(str(s.get("series", "")), {})
                    # The cache JSON's fluid field is the requested slot, not the
                    # actual acquired series type. Never use it as ground truth.
                    vals = []
                    for key in ("Fluid_Sensitive", "Fat_Suppression"):
                        value = r.get(key, -1)
                        vals.append(float(value) if pd.notna(value) and value in (0, 1) else -1.)
                    self.sequence[uid][int(s.get("slot", -1))] = vals


def bounded_crop(image, g, anchor, eh, ev, extent, size):
    """Physical square projected into a source slice; shrink/shift to retain pixels.

    The fixed dense512 source has already lost anything outside its 140mm field.
    This function never invents those pixels: a requested box that does not fit
    is translated/scaled inside the available field. Return its retained scale.
    """
    corners = np.array([anchor - eh * extent / 2 - ev * extent / 2,
                        anchor + eh * extent / 2 - ev * extent / 2,
                        anchor - eh * extent / 2 + ev * extent / 2])
    src = geometry.to_pixel(g, corners) * (image.shape[-1] / geometry.IMG)
    fourth = src[1] + src[2] - src[0]
    allp = np.vstack([src, fourth])
    center = allp.mean(0)
    width = np.ptp(allp, axis=0)
    if not np.isfinite(src).all() or min(width) < 1e-5:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), 0.
    scale = min(1., float(((np.array(image.shape[::-1]) - 2) / np.maximum(width, 1e-6)).min()))
    src = center + (src - center) * scale
    allp = np.vstack([src, src[1] + src[2] - src[0]])
    shift = np.maximum(1 - allp.min(0), 0) - np.maximum(allp.max(0) - (np.array(image.shape[::-1]) - 1), 0)
    src += shift
    matrix = cv2.getAffineTransform(src.astype(np.float32), np.array([[0, 0], [size - 1, 0], [0, size - 1]], np.float32))
    tile = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return tile, scale


def views_for(plane, mode):
    # (view id, horizontal mm, vertical mm, extent mm). Global always survives.
    out = [(0, 0., 0., 140.)]
    if mode == "global-local":
        if plane == "cor":
            out += [(2, -28., 0., 90.), (3, 28., 0., 90.)]
        elif plane == "sag":
            out += [(1, 0., 0., 100.), (4, -30., -20., 90.)]
        else:
            out += [(1, 0., 0., 100.), (4, 0., -30., 90.)]
    return out


class Windows(Dataset):
    def __init__(self, corpus, ids, *, size=224, windows=2, views="global-local", train=False,
                 seed=11, epoch=0, phase=0, phases=1):
        self.c, self.ids = corpus, list(ids)
        self.size, self.windows, self.views, self.train = size, windows, views, train
        self.seed, self.epoch, self.phase, self.phases = seed, epoch, phase, phases

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        row = self.c.row[uid]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, row]))
        vol = self.c.study(row)
        geom, mask = self.c.geom[row], self.c.masks[row]
        tiles, meta = [], []
        p = self.c.P[row]
        tagged = self.c.tag_left.get(uid)
        left = tagged if tagged is not None else float(p[0] > 0)
        # Unknown image tags remain visible as confidence 0 in model metadata.
        lat = 1. if left >= .5 else -1.
        E = np.diag([lat, -1., 1.])
        for lo, hi, plane in BLOCKS:
            cand = valid_centres(mask, lo, hi)
            cs = choose_centres(cand, self.windows, rng=rng if self.train else None,
                                phase=self.phase, phases=self.phases)
            eh, ev = (np.asarray(a, float) @ E for a in geometry.AXES[plane])
            normal = np.asarray(geometry.NORMAL[plane], float) @ E
            centers = geometry.slice_centres(geom[lo:hi])
            vals = self.c.sequence.get(uid, {}).get(lo, [-1., -1.])
            for c in cs:
                ok = bool(np.all(geom[c - 1:c + 2, 19] > .5) and np.all(geom[c - 1:c + 2, :2] > 0))
                spacing = abs(float((centers[c - lo + 1] - centers[c - lo - 1]) @ normal)) / 2 if ok else 0.
                rel = float((centers[c - lo] - p) @ normal / 70.) if ok else (c - lo) / max(hi - lo - 1, 1)
                ordered = [c - 1, c, c + 1]
                if ok and (centers[c - lo + 1] - centers[c - lo - 1]) @ normal < 0:
                    ordered.reverse()
                for view, dh, dv, extent in views_for(plane, self.views):
                    if not ok and view != 0:
                        continue
                    anchor = (centers[c - lo] if view == 0 else p) + dh * eh + dv * ev
                    if self.train and view:
                        anchor = anchor + rng.uniform(-3, 3) * eh + rng.uniform(-3, 3) * ev
                    rgb, scales = [], []
                    for k in ordered:
                        if ok:
                            tile, scale = bounded_crop(np.ascontiguousarray(vol[k]), geom[k], anchor, eh, ev, extent, self.size)
                        else:
                            tile, scale = cv2.resize(vol[k], (self.size, self.size)), 0.
                        rgb.append(tile); scales.append(scale)
                    tiles.append(np.stack(rgb))
                    meta.append([PLANES[plane], view, rel, math.log1p(spacing), extent * min(scales) / 140.,
                                 *vals, float(ok), float(tagged is not None)])
        if not tiles:  # Emit a masked fallback; prediction runner treats this as a data error.
            return torch.zeros((1, 3, self.size, self.size), dtype=torch.uint8), torch.zeros((1, 9)), torch.zeros(1, dtype=torch.bool), i
        x = np.stack(tiles)
        if self.train:
            gain, bias = rng.uniform(.9, 1.1), rng.uniform(-5, 5)
            x = np.clip(x.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return torch.from_numpy(x), torch.tensor(meta, dtype=torch.float32), torch.ones(len(x), dtype=torch.bool), i


def collate(batch):
    n = max(len(x[0]) for x in batch)
    shape = (len(batch), n) + tuple(batch[0][0].shape[1:])
    x = torch.zeros(shape, dtype=torch.uint8)
    m = torch.zeros((len(batch), n, 9))
    valid = torch.zeros((len(batch), n), dtype=torch.bool)
    for i, (a, b, v, _) in enumerate(batch):
        x[i, :len(a)], m[i, :len(a)], valid[i, :len(a)] = a, b, v
    return x, m, valid, torch.tensor([b[3] for b in batch])


class TinyEncoder(nn.Module):
    """CPU wiring tests only; never a competitive training default."""
    num_features = 16
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=4, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    def forward(self, x):
        return self.net(x)


class Reader(nn.Module):
    def __init__(self, arch="convnext_tiny.fb_in1k", *, pretrained=False, bb_file=None,
                 hidden=192, dummy=False, grad_checkpoint=False):
        super().__init__()
        if dummy:
            self.encoder = TinyEncoder()
        else:
            import timm
            self.encoder = timm.create_model(arch, pretrained=pretrained and not bb_file, num_classes=0, global_pool="avg")
            if bb_file:
                state = torch.load(bb_file, map_location="cpu", weights_only=False)
                self.encoder.load_state_dict(state, strict=True)
            if grad_checkpoint:
                self.encoder.set_grad_checkpointing(True)
        self.proj = nn.Sequential(nn.Linear(self.encoder.num_features, hidden), nn.LayerNorm(hidden))
        self.plane, self.view = nn.Embedding(3, hidden), nn.Embedding(5, hidden)
        self.geom = nn.Linear(7, hidden, bias=False)
        nn.init.zeros_(self.plane.weight); nn.init.zeros_(self.view.weight); nn.init.zeros_(self.geom.weight)
        self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 12))
        self.cls_w = nn.Parameter(torch.randn(12, hidden) * .02)
        self.cls_b = nn.Parameter(torch.zeros(12))
        self.grade_w = nn.Parameter(torch.randn(12, 3, hidden) * .02)
        self.grade_b = nn.Parameter(torch.zeros(12, 3))
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x, meta, valid):
        if not valid.any(1).all():
            raise ValueError("study without usable within-series triplets")
        pixels = (x[valid].float() / 255. - self.mean) / self.std
        h = self.proj(self.encoder(pixels).float())
        tokens = h.new_zeros((*valid.shape, h.shape[-1]))
        tokens[valid] = h
        tokens = tokens + self.plane(meta[..., 0].long()) + self.view(meta[..., 1].long()) + self.geom(meta[..., 2:])
        a = self.att(tokens).float().masked_fill(~valid[..., None], -torch.inf).softmax(1)
        study = torch.einsum("bwc,bwh->bch", a, tokens)
        return (study * self.cls_w).sum(-1) + self.cls_b, torch.einsum("bch,ckh->bck", study, self.grade_w) + self.grade_b


def masked_loss(logits, targets, mask, confidence=None):
    weight = mask.float() if confidence is None else mask.float() * confidence
    # Each finding contributes equally among those with supervision in this batch.
    # Confidence reduces absolute evidence strength, even for a lone weak label.
    # Normalizing by confidence-weighted counts would cancel that reduction.
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    dims = (0,)
    den = mask.float().sum(dims)
    keep = den > 0
    return ((loss * weight).sum(dims)[keep] / den[keep]).mean() if keep.any() else logits.sum() * 0.


def load_labels(path, train_csv, mode):
    z = dict(np.load(path, allow_pickle=False))
    assert list(z["labels"].astype(str)) == LAB, "label ordering mismatch"
    ids = z["ids"].astype(str)
    assert len(set(ids)) == len(ids), "duplicate label IDs"
    tr = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    gold = set(tr.loc[tr[LAB].notna().all(axis=1), "StudyInstanceUID"])
    if set(ids) & gold:
        raise ValueError("expert gold studies must not enter training label bundle")
    prefix = "control_" if mode == "control" else ""
    out = {"ids": ids, "folds": np.asarray(z["folds"])}
    for key in ("target", "mask", "confidence"):
        v = np.asarray(z[prefix + key])
        assert v.shape == (len(ids), 12) and np.isfinite(v).all(), key
        assert ((v >= 0) & (v <= 1)).all(), key
        out[key] = torch.tensor(v, dtype=torch.float32)
    for key in ("ordinal_target", "ordinal_mask"):
        v = np.asarray(z[key])
        assert v.shape == (len(ids), 12, 3) and np.isfinite(v).all(), key
        assert ((v >= 0) & (v <= 1)).all(), key
        out[key] = torch.tensor(v, dtype=torch.float32)
    oc = np.asarray(z.get("ordinal_confidence", np.ones((len(ids), 12, 3), np.float32)))
    assert oc.shape == (len(ids), 12, 3) and np.isfinite(oc).all() and ((oc >= 0) & (oc <= 1)).all()
    out["ordinal_confidence"] = torch.tensor(oc, dtype=torch.float32)
    for key in ("control_target", "control_mask"):
        v = np.asarray(z[key])
        assert v.shape == (len(ids), 12) and np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), key
        out[key] = torch.tensor(v, dtype=torch.float32)
    return out


def protocol(a):
    return {"version": PROTOCOL_VERSION, "arch": a.arch, "dummy": a.dummy_encoder, "size": a.size,
            "windows": a.windows, "views": a.views, "eval_passes": a.eval_passes, "hidden": a.hidden,
            "norm": "imagenet", "sampler": "valid_series_interleaved", "checkpoint_policy": "fixed_ema"}


def snapshot_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def diagnostic_auc(pred, labels, mask):
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, name in enumerate(LAB):
        keep = mask[:, j] > 0
        y = labels[keep, j] >= .5
        out[name] = float(roc_auc_score(y, pred[keep, j])) if len(np.unique(y)) == 2 else None
    finite = [x for x in out.values() if x is not None]
    return {"meaning": "thresholded report-label agreement; NOT expert-image AUC", "per_finding": out,
            "macro_available": float(np.mean(finite)) if finite else None}


@torch.no_grad()
def predict(model, corpus, ids, cfg, *, bs=2, workers=0, device="cpu", max_seconds=None):
    model.eval()
    deadline = time.monotonic() + max_seconds if max_seconds is not None else math.inf
    result = np.zeros((len(ids), 12), np.float64)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.bfloat16
    for phase in range(cfg["eval_passes"]):
        ds = Windows(corpus, ids, size=cfg["size"], windows=cfg["windows"], views=cfg["views"],
                     phase=phase, phases=cfg["eval_passes"])
        loader = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, collate_fn=collate,
                            generator=torch.Generator().manual_seed(123))
        for x, m, v, ix in loader:
            if time.monotonic() >= deadline:
                raise TimeoutError("prediction budget exceeded; no partial predictions published")
            with torch.autocast(str(device).split(":")[0], dtype=dtype, enabled=str(device).startswith("cuda")):
                logits, _ = model(x.to(device), m.to(device), v.to(device))
            result[ix.numpy()] += logits.float().sigmoid().cpu().numpy()
    return (result / cfg["eval_passes"]).astype(np.float32)


def synthetic(out):
    """Tiny on-disk geometry corpus, with one absent non-fluid sequence."""
    root = Path(out) / "synthetic"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "labels.npz").exists():
        return root
    ids = np.array([f"synthetic-{i}" for i in range(10)])
    rng = np.random.default_rng(1)
    vols = rng.integers(1, 255, (10, 80, 32, 32), dtype=np.uint8)
    masks = np.ones((10, 80), np.uint8)
    masks[0, 22:40] = 0; vols[0, 22:40] = 0
    geoms = np.zeros((10, 80, 20), np.float32)
    info, series = {}, []
    for uid in ids:
        info[uid] = []
        for bi, (lo, hi, plane) in enumerate(BLOCKS):
            sid = f"{uid}-{bi}"
            info[uid].append({"slot": lo, "series": sid, "Laterality": "L"})
            series.append({"SeriesInstanceUID": sid, "Fluid_Sensitive": bi % 2, "Fat_Suppression": bi % 2})
    for lo, hi, plane in BLOCKS:
        for k, pos in enumerate(np.linspace(-40, 40, hi - lo)):
            g = np.zeros(20, np.float32)
            g[:2] = 140 / 32; g[13:16] = 32; g[19] = 1
            if plane == "sag":
                g[2:5] = [80 + pos, -70, 70]; g[5:8] = [0, 1, 0]; g[8:11] = [0, 0, -1]
            elif plane == "cor":
                g[2:5] = [10, pos, 70]; g[5:8] = [1, 0, 0]; g[8:11] = [0, 0, -1]
            else:
                g[2:5] = [10, -70, pos]; g[5:8] = [1, 0, 0]; g[8:11] = [0, 1, 0]
            geoms[:, lo + k] = g
    for name, val in (("ids", ids), ("vols", vols), ("masks", masks), ("geom", geoms)):
        np.save(root / f"{name}_s0.npy", val)
    (root / "series_s0.json").write_text(json.dumps(info))
    pd.DataFrame(series).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids, **{n: [np.nan] * len(ids) for n in LAB}}).to_csv(root / "train.csv", index=False)
    y = rng.integers(0, 2, (len(ids), 12)).astype(np.float32)
    np.savez(root / "labels.npz", ids=ids, labels=np.array(LAB), folds=np.arange(len(ids)) % 2,
             target=y, mask=np.ones_like(y), confidence=np.ones_like(y), control_target=y,
             control_mask=np.ones_like(y), control_confidence=np.ones_like(y),
             ordinal_target=np.repeat(y[..., None], 3, -1), ordinal_mask=np.ones((len(ids), 12, 3)))
    return root


def main(argv=None):
    global STOP
    STOP = False
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root"); ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--labels"); ap.add_argument("--train-csv"); ap.add_argument("--series-csv")
    ap.add_argument("--out", default="."); ap.add_argument("--tag", default="pilot")
    ap.add_argument("--target-mode", choices=["control", "definition"], default="definition")
    ap.add_argument("--views", choices=["global", "global-local"], default="global-local")
    ap.add_argument("--arch", default="convnext_tiny.fb_in1k")
    ap.add_argument("--bb-file"); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--dummy-encoder", action="store_true"); ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--size", type=int, default=224); ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=192); ap.add_argument("--eval-passes", type=int, default=2)
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=.05); ap.add_argument("--ema", type=float, default=.99)
    ap.add_argument("--aux-weight", type=float, default=.2); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--max-seconds", type=float, default=10800)
    ap.add_argument("--eval-max-seconds", type=float, default=900)
    ap.add_argument("--resume"); ap.add_argument("--predict"); ap.add_argument("--prediction-out")
    ap.add_argument("--predict-ids", help="optional Unicode .npy or one study UID per line; preserves this order")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.epochs, a.size, a.windows, a.workers, a.hidden, a.eval_passes = 1, 32, 1, 0, 32, 1
        a.bs, a.accum = 2, 2
        torch.set_num_threads(1)
        if not a.data_root:
            if not a.dummy_encoder:
                raise SystemExit("synthetic smoke requires --dummy-encoder")
            root = synthetic(out)
            a.data_root, a.n_shards = str(root), 1
            a.labels, a.train_csv, a.series_csv = [str(root / n) for n in ("labels.npz", "train.csv", "train_series.csv")]
    if not a.data_root or min(a.bs, a.accum, a.windows, a.eval_passes, a.epochs) < 1:
        raise SystemExit("provide corpus and positive batching/epoch arguments")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cv2.setNumThreads(1)
    corpus = Corpus(a.data_root, a.n_shards, a.series_csv)
    cfg = protocol(a)
    if a.predict:
        ck = torch.load(a.predict, map_location="cpu", weights_only=False)
        cfg = ck["protocol"]
        if cfg["version"] != PROTOCOL_VERSION:
            raise ValueError("unsupported prediction protocol")
        model = Reader(cfg["arch"], hidden=cfg["hidden"], dummy=cfg["dummy"]).to(device)
        model.load_state_dict(ck["model"], strict=True)
        predict_ids = list(corpus.ids)
        if a.predict_ids:
            pids = Path(a.predict_ids)
            predict_ids = (np.load(pids, allow_pickle=False).astype(str).tolist() if pids.suffix == ".npy"
                           else [s.strip() for s in pids.read_text().splitlines() if s.strip()])
            if not predict_ids or len(set(predict_ids)) != len(predict_ids):
                raise ValueError("--predict-ids must be nonempty and unique")
            if set(predict_ids) - set(corpus.ids):
                raise ValueError("--predict-ids contains studies absent from the corpus")
        pred = predict(model, corpus, predict_ids, cfg, bs=a.bs, workers=a.workers, device=device, max_seconds=a.eval_max_seconds)
        dest = a.prediction_out or str(out / f"definition_{a.tag}_pred.npz")
        np.savez(dest, ids=np.array(predict_ids), labels=np.array(LAB), pred=pred,
                 protocol=json.dumps(cfg), metadata=json.dumps(ck.get("contract", {})),
                 checkpoint_sha256=sha256(a.predict))
        print(f"predictions {dest}: {len(pred)} studies", flush=True)
        return 0
    if not a.labels or not a.train_csv:
        raise SystemExit("--labels and --train-csv required; no gold image may train")
    targets = load_labels(a.labels, a.train_csv, a.target_mode)
    ids = targets["ids"]
    have = np.array([u in corpus.row for u in ids])
    if not have.all():
        raise ValueError(f"{int((~have).sum())} labeled studies missing from corpus")
    train_ix = np.flatnonzero(targets["folds"] != a.fold)
    val_ix = np.flatnonzero(targets["folds"] == a.fold)
    if not len(train_ix) or not len(val_ix):
        raise ValueError("training and held-out report fold must both be nonempty")
    model = Reader(a.arch, pretrained=not a.no_pretrained and not a.resume, bb_file=a.bb_file,
                   hidden=a.hidden, dummy=a.dummy_encoder, grad_checkpoint=a.grad_checkpoint).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": a.lr},
                            {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": a.head_lr}], weight_decay=a.wd)
    per_epoch = math.ceil(math.ceil(len(train_ix) / a.bs) / a.accum)
    total_steps = a.epochs * per_epoch
    def factor(s):
        warm = max(1, int(total_steps * .1))
        return min(1., (s + 1) / warm) * .5 * (1 + math.cos(math.pi * min(s / max(total_steps, 1), 1.)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    use_cuda = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    label_sha = sha256(a.labels)
    epoch, batch_cursor, step, history = 0, 0, 0, []
    contract = {"protocol": cfg, "labels_sha256": label_sha, "target_mode": a.target_mode, "fold": a.fold,
                "seed": a.seed, "bs": a.bs, "accum": a.accum, "epochs": a.epochs,
                "aux_weight": a.aux_weight, "lr": a.lr, "head_lr": a.head_lr, "wd": a.wd, "ema": a.ema,
                "train_csv_sha256": sha256(a.train_csv), "series_csv_sha256": sha256(a.series_csv) if a.series_csv else None,
                "corpus_geometry_sha256": hashlib.sha256(corpus.geom.tobytes() + corpus.masks.tobytes()).hexdigest(),
                "code_sha256": sha256(__file__), "reader_sha256": sha256(__file__),
                "geometry_helper_sha256": sha256(geometry.__file__),
                "init_sha256": sha256(a.bb_file) if a.bb_file else None,
                "init_source": Path(a.bb_file).name if a.bb_file else ("random" if a.no_pretrained or a.dummy_encoder else "timm:" + a.arch)}
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        if ck["contract"] != contract:
            raise ValueError("resume contract differs (data/protocol/loss/schedule)")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        ema = {k: v.to(device) for k, v in ck["ema"].items()}
        epoch, batch_cursor, step, history = ck["epoch"], ck["batch_cursor"], ck["step"], ck["history"]
        restore_rng(ck["rng"])
    started = time.monotonic()
    def stopping():
        return STOP or time.monotonic() - started >= a.max_seconds
    def handler(_sig, _frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, handler); signal.signal(signal.SIGINT, handler)
    def save(ep, cursor, reason):
        atomic_torch({"model": model.state_dict(), "ema": ema, "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "rng": snapshot_rng(),
                      "epoch": ep, "dataset_epoch": ep, "batch_cursor": cursor, "step": step, "history": history,
                      "contract": contract}, out / f"definition_{a.tag}_resume.pt")
        atomic_torch({"model": ema, "protocol": cfg, "contract": contract, "step": step, "epoch": ep,
                      "checkpoint_policy": "fixed_ema"}, out / f"definition_{a.tag}_ema.pt")
        atomic_json({"epoch": ep, "next_batch": cursor, "step": step, "reason": reason,
                     "elapsed_seconds": round(time.monotonic() - started, 2)}, out / "progress.json")
    interrupted = False
    while epoch < a.epochs:
        ds = Windows(corpus, ids[train_ix], size=a.size, windows=a.windows, views=a.views,
                     train=True, seed=a.seed, epoch=epoch)
        generator = torch.Generator().manual_seed(a.seed + epoch)
        loader = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                            generator=generator, pin_memory=use_cuda)
        model.train(); opt.zero_grad(set_to_none=True)
        running, batches = 0., 0
        for bi, (x, m, v, ix) in enumerate(loader):
            if bi < batch_cursor:
                continue
            if stopping() and bi % a.accum == 0:
                save(epoch, bi, "time_or_signal")
                interrupted = True
                break
            actual = train_ix[ix.numpy()]
            values = {k: targets[k][actual].to(device) for k in ("target", "mask", "confidence", "ordinal_target", "ordinal_mask", "ordinal_confidence")}
            with torch.autocast(str(device).split(":")[0], dtype=torch.float16, enabled=use_cuda):
                logit, ordinal = model(x.to(device), m.to(device), v.to(device))
                loss = masked_loss(logit, values["target"], values["mask"], values["confidence"])
                loss = loss + a.aux_weight * masked_loss(ordinal, values["ordinal_target"], values["ordinal_mask"], values["ordinal_confidence"])
            # Flush even a short final accumulation group; no gradients disappear at epoch boundaries.
            group_start = (bi // a.accum) * a.accum
            group_size = min(a.accum, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            boundary = (bi + 1) % a.accum == 0 or bi + 1 == len(loader)
            if boundary:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                old_scale = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= old_scale:
                    sched.step(); step += 1
                    with torch.no_grad():
                        decay = min(a.ema, (1 + step) / (10 + step))
                        for k, value in model.state_dict().items():
                            if value.is_floating_point():
                                ema[k].mul_(decay).add_(value.detach(), alpha=1 - decay)
                            else:
                                ema[k].copy_(value)
                opt.zero_grad(set_to_none=True)
            running += float(loss.detach()); batches += 1
            # Mid-epoch checkpoints occur only at an optimizer boundary so resume is exact.
            if boundary and stopping():
                save(epoch, bi + 1, "time_or_signal"); interrupted = True; break
        if interrupted:
            break
        history.append({"epoch": epoch, "loss": running / max(1, batches), "steps": step})
        epoch += 1; batch_cursor = 0
        save(epoch, 0, "epoch_complete")
        print(json.dumps(history[-1]), flush=True)
    if not interrupted:
        model.load_state_dict(ema)
        try:
            pred = predict(model, corpus, ids[val_ix], cfg, bs=a.bs, workers=a.workers, device=device,
                           max_seconds=a.eval_max_seconds)
        except TimeoutError:
            atomic_json({"contract": contract, "history": history, "status": "training_complete_evaluation_timeout",
                         "train_studies": len(train_ix), "validation_studies": len(val_ix)}, out / f"definition_{a.tag}_report.json")
            return 0
        diag = {"fixed_control_report": diagnostic_auc(pred, targets["control_target"][val_ix].numpy(), targets["control_mask"][val_ix].numpy()),
                "training_target_report": diagnostic_auc(pred, targets["target"][val_ix].numpy(), targets["mask"][val_ix].numpy())}
        np.savez(out / f"definition_{a.tag}_oof.npz", ids=ids[val_ix], labels=np.array(LAB), pred=pred,
                 target=targets["target"][val_ix].numpy(), mask=targets["mask"][val_ix].numpy(),
                 control_target=targets["control_target"][val_ix].numpy(), control_mask=targets["control_mask"][val_ix].numpy(),
                 protocol=json.dumps(cfg), metadata=json.dumps(contract),
                 checkpoint_sha256=sha256(out / f"definition_{a.tag}_ema.pt"))
        atomic_json({"contract": contract, "history": history, "diagnostic": diag, "status": "complete",
                     "train_studies": len(train_ix), "validation_studies": len(val_ix),
                     "checkpoint_sha256": sha256(out / f"definition_{a.tag}_ema.pt")}, out / f"definition_{a.tag}_report.json")
        print(json.dumps(diag), flush=True)
    else:
        atomic_json({"contract": contract, "history": history, "status": "interrupted_resumable",
                     "train_studies": len(train_ix), "validation_studies": len(val_ix)}, out / f"definition_{a.tag}_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
