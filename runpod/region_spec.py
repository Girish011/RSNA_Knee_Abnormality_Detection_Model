#!/usr/bin/env python3
"""Native-resolution region specialists on the dense512 cache (PLAN_TO_TOP 2026-10-03).

Stage 1 (localiser, no training): the three stacks are prescribed on the joint, so the joint centre is
    P = (x of the sagittal stack centre, y of the coronal stack centre, z of the axial stack centre)
in patient mm (LPS), read from the per-slot DICOM geometry in geom_s*.npy. Knee side (which patient
direction is lateral) = DICOM Laterality, else the "side" region model (arm `--region side`), else
sign(P.x). Canonical knee axes: A = lateral, B = anterior, C = superior.

Stage 2 (specialists): a region is a set of tile groups; every tile is a 2.5D window (3 adjacent slots)
warped from the 512 px slice to a box given in canonical mm around P, at ~native resolution, mirrored
so that all knees look alike. `tf` sees ONE tibiofemoral compartment per sample: every study gives a
lateral and a (mirrored) medial sample with the meniscus / OA labels of that compartment.
Tiles -> shared encoder -> tokens (+group, +slice-position) -> transformer mix -> one query per output.

Protocol = family_a.py (folds_nonexpert_5, dev-29 at gold weight, holdout-29 never trained, EMA + last
members, no epoch selection). Outputs: raptor_rs_<tag>.pt/_last.pt, raptor_oof_<tag>_fold<f>.npz and
raptor_holdout_<tag>.npz (pred has all 12 columns, NaN where the region has no output),
raptor_ft_<tag>.json; `--region side` writes raptor_side_<tag>.npz (ids, p_left) for every study.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
L = {f: i for i, f in enumerate(LAB)}
WEAK5 = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis", "Contusion"]
BLOCKS = {"SagF": (0, 22, "sag"), "SagNF": (22, 40, "sag"), "CorF": (40, 55, "cor"),
          "CorNF": (55, 65, "cor"), "Ax": (65, 80, "ax")}
IMG = 512
N_POS = 8
PARTNER = {"SagF": "SagNF", "CorF": "CorNF"}     # --chan seqfuse: fluid-sensitive block -> its non-fluid partner
AUX_FINDS = ("Medial OA", "Lateral OA", "PF OA")  # --aux_grades: findings with a report severity grade (0-4) as side target
AUX_THR = (2, 3, 4)                               # ordinal side outputs: grade >= 2 (any), >= 3 (moderate), >= 4 (severe)

# In-plane image axes as canonical (A lateral, B anterior, C superior) unit vectors:
# coronal: right = lateral, down = inferior | sagittal: right = posterior, down = inferior |
# axial: right = lateral, down = posterior.  NORMAL = canonical axis that orders the slices.
AXES = {"cor": ((1, 0, 0), (0, 0, -1)), "sag": ((0, -1, 0), (0, 0, -1)), "ax": ((1, 0, 0), (0, -1, 0))}
NORMAL = {"cor": (0, 1, 0), "sag": (1, 0, 0), "ax": (0, 0, 1)}


def grp(block, hbox, vbox, out_wh, sel, n_train, n_eval):
    """hbox/vbox: mm along the image axes (AXES) relative to P; sel: mm range along NORMAL (or None)."""
    return dict(block=block, hbox=hbox, vbox=vbox, out=out_wh, sel=sel, n=(n_train, n_eval))


REGIONS = {
    # one tibiofemoral compartment; variant = (compartment sign along A, findings)
    "tf": dict(
        variants=[(+1, ["Lateral Meniscus", "Lateral OA"]), (-1, ["Medial Meniscus", "Medial OA"])],
        groups=[grp("CorF", (-12, 56), (-40, 40), (256, 304), (-35, 35), 5, 8),
                grp("CorNF", (-12, 56), (-40, 40), (256, 304), (-35, 35), 3, 5),
                grp("SagF", (-52, 52), (-40, 40), (384, 304), (4, 52), 4, 6),
                grp("SagNF", (-52, 52), (-40, 40), (384, 304), (4, 52), 3, 5)]),
    "pf": dict(
        variants=[(+1, ["PF OA"])],
        groups=[grp("Ax", (-50, 50), (-82, 6), (368, 320), (-15, 50), 5, 8),
                grp("SagF", (-86, 0), (-60, 40), (304, 352), (-26, 26), 4, 6),
                grp("SagNF", (-86, 0), (-60, 40), (304, 352), (-26, 26), 3, 5)]),
    "rec": dict(
        variants=[(+1, ["Effusion", "Synovitis", "Baker's"])],
        groups=[grp("Ax", (-70, 70), (-70, 70), (448, 448), None, 5, 8),
                grp("SagF", (-70, 70), (-75, 45), (448, 384), None, 5, 8),
                grp("SagNF", (-70, 70), (-75, 45), (448, 384), None, 2, 4)]),
    "bm": dict(
        variants=[(+1, ["Contusion", "Fracture"])],
        groups=[grp("CorF", (-62, 62), (-55, 55), (448, 400), None, 5, 8),
                grp("SagF", (-62, 62), (-55, 55), (448, 400), None, 5, 8),
                grp("Ax", (-62, 62), (-62, 62), (400, 400), None, 3, 5),
                grp("CorNF", (-62, 62), (-55, 55), (448, 400), None, 3, 5)]),
    # knee side: coronal field incl. the fibula in PATIENT orientation (A := +x), target = left knee
    "side": dict(
        variants=[(+1, ["__left__"])],
        groups=[grp("CorF", (-70, 70), (-45, 70), (224, 184), None, 3, 4),
                grp("CorNF", (-70, 70), (-45, 70), (224, 184), None, 2, 3)]),
}


# ------------------------------------------------------------------ corpus + localiser
def _find(name, roots=(HERE, "/workspace/bundle", "/workspace/out", "/kaggle/input", "/workspace/input")):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d, _, fs in os.walk(root):
            if name in fs:
                return os.path.join(d, name)
    return None


def slice_centres(g):
    """(...,GEOM) -> (...,3) patient mm of the slice centres."""
    return (g[..., 2:5] + g[..., 5:8] * (g[..., 1:2] * g[..., 15:16] / 2)
            + g[..., 8:11] * (g[..., 0:1] * g[..., 14:15] / 2))


class Corpus:
    """Memory-mapped dense512 shards + per-slot geometry; joint centre P and tag laterality per study."""

    def __init__(self, n_shards=8, root=None):
        self.n_shards, self.root, self.vols = n_shards, root, None
        ids, masks, geom, tag = [], [], [], {}
        for i in range(n_shards):
            ids.append(np.load(self._p(f"ids_s{i}.npy"), allow_pickle=True).astype(str))
            masks.append(np.load(self._p(f"masks_s{i}.npy")))
            geom.append(np.load(self._p(f"geom_s{i}.npy")))
            for u, infos in json.load(open(self._p(f"series_s{i}.json"))).items():
                s = {x.get("Laterality", "")[:1].upper() for x in infos} & {"L", "R"}
                if len(s) == 1:
                    tag[str(u)] = 1.0 if s.pop() == "L" else 0.0
        self.ids = np.concatenate(ids)
        self.masks = np.concatenate(masks).astype(bool)
        self.geom = np.concatenate(geom).astype(np.float64)
        self.offsets = np.cumsum([0] + [len(x) for x in ids])
        self.row = {u: i for i, u in enumerate(self.ids)}
        self.tag_left = tag
        cen = slice_centres(self.geom)
        self.P = np.zeros((len(self.ids), 3))
        for r in range(len(self.ids)):
            allc = cen[r][self.masks[r]]
            fall = allc.mean(0) if len(allc) else np.zeros(3)
            for ax, blocks in enumerate((("SagF", "SagNF"), ("CorF", "CorNF"), ("Ax",))):
                k = np.concatenate([np.arange(*BLOCKS[b][:2]) for b in blocks])
                k = k[self.masks[r, k]]
                self.P[r, ax] = cen[r, k, ax].mean() if len(k) else fall[ax]

    def _p(self, name):
        p = os.path.join(self.root, name) if self.root else _find(name)
        if not p or not os.path.exists(p):
            raise SystemExit(f"dense512 file {name} not found (DENSE_NAME=dense512 fetch_dense80.py)")
        return p

    def study(self, row):
        if self.vols is None:
            self.vols = [np.load(self._p(f"vols_s{i}.npy"), mmap_mode="r") for i in range(self.n_shards)]
        s = int(np.searchsorted(self.offsets, row, side="right") - 1)
        return self.vols[s][row - self.offsets[s]]

    def left_prob(self, side_file=None):
        """P(left knee) per study: Laterality tag > side model > sign of P.x."""
        p = {u: (1.0 if self.P[i, 0] > 0 else 0.0) for u, i in self.row.items()}
        src = dict.fromkeys(p, "x")
        if side_file:
            z = np.load(side_file, allow_pickle=True)
            for u, v in zip(z["ids"].astype(str), z["p_left"]):
                p[u] = float(v); src[u] = "model"
        for u, v in self.tag_left.items():
            p[u] = v; src[u] = "tag"
        return p, src


def to_pixel(g, X):
    """Patient mm (k,3) -> (u,v) px in the 512 slice with geometry row g."""
    d = X - g[2:5]
    x = d @ g[5:8] / g[1]
    y = d @ g[8:11] / g[0]
    return np.stack([(x - g[12]) * IMG / g[13], (y - g[11]) * IMG / g[13]], 1)


# ------------------------------------------------------------------ data
class RegionTiles(Dataset):
    """Item = (study, variant) -> {group: (n,3,H,W)} tiles + pos/valid vectors, target, weight."""

    def __init__(self, corpus, region, ids, labels, weights, left, train, aug=1.0, eval_phase=0.0,
                 side_mode=False, chan="adj", aux=None, variants=None, few=False):
        self.c, self.reg, self.train = corpus, REGIONS[region], train
        self.aux = aux        # {uid: (targets (12, 3), weight (12,))} from load_aux, or None: 3 extra targets per item
        self.chan = chan      # "seqfuse": tiles of a fluid-sensitive group = [fluid, non-fluid, difference] (see PARTNER)
        self.labels, self.weights, self.left = labels, weights, left
        self.items = [(u, v) for u in ids for v in (variants or range(len(self.reg["variants"])))]
        self.aug = aug if train else 0.0
        self.eval_phase, self.side_mode = eval_phase, side_mode
        self.few = few        # evaluation with the TRAINING number of tiles per group (evenly spaced): ~1.6x cheaper

    def __len__(self):
        return len(self.items)

    def _pick(self, cand, n):
        if self.train:
            return np.sort(np.random.choice(cand, n, replace=len(cand) < n))
        frac = (np.linspace(0.0, 1.0, n) + self.eval_phase) % 1.0 if n > 1 else np.array([0.5])
        return cand[np.round(frac * (len(cand) - 1)).astype(int)]

    def __getitem__(self, i):
        uid, vi = self.items[i]
        sgn, finds = self.reg["variants"][vi]
        row = self.c.row[uid]
        vol, msk, geom, P = self.c.study(row), self.c.masks[row], self.c.geom[row], self.c.P[row].copy()
        if self.side_mode:       # patient orientation; random mirror flips the target
            lat = random.choice((1.0, -1.0)) if self.train else 1.0
            t = self.labels.get(uid, 0.0)                 # untagged studies are only predicted
            y = np.array([t if lat > 0 else 1.0 - t], np.float32)
            w = np.array([self.weights.get(uid, 0.0)], np.float32)
        else:                    # A points lateral for the lateral variant, medial for the mirrored one
            lat = (1.0 if self.left[uid] >= 0.5 else -1.0) * sgn
            idx = [L[f] for f in finds]
            y, w = self.labels[uid][idx].astype(np.float32), self.weights[uid][idx].astype(np.float32)
            if self.aux is not None:     # severity grade of this variant's OA finding; weight 0 = not mentioned / no report
                k = next(L[f] for f in finds if f in AUX_FINDS)
                ay, aw = self.aux.get(uid, (np.zeros((len(LAB), len(AUX_THR)), np.float32), np.zeros(len(LAB), np.float32)))
                y = np.concatenate([y, ay[k]])
                w = np.concatenate([w, np.full(len(AUX_THR), aw[k], np.float32)])
        E = np.array([[lat, 0, 0], [0, -1, 0], [0, 0, 1]], float)   # canonical A,B,C in patient LPS
        rot = math.radians(random.uniform(-8, 8)) * self.aug
        sc = 1.0 + random.uniform(-0.08, 0.08) * self.aug
        P = P + np.random.uniform(-5, 5, 3) * self.aug
        cr, sr = math.cos(rot), math.sin(rot)
        out = {}
        for gi, gdef in enumerate(self.reg["groups"]):
            lo, hi, plane = BLOCKS[gdef["block"]]
            n = gdef["n"][0 if (self.train or self.few) else 1]
            W, H = gdef["out"]
            eh, ev = (np.array(a, float) @ E for a in AXES[plane])
            nrm = np.array(NORMAL[plane], float) @ E
            valid = np.where(msk[lo:hi])[0]
            tiles = np.zeros((n, 3, H, W), np.float32)
            pos = np.zeros(n, np.int64)
            ok = 0
            if len(valid) >= 3:
                coord = (slice_centres(geom[lo:hi]) - P) @ nrm
                cand = valid[(valid > valid.min()) & (valid < valid.max())]
                rng = gdef["sel"] or (coord[valid].min(), coord[valid].max())
                inr = cand[(coord[cand] >= rng[0]) & (coord[cand] <= rng[1])]
                if len(inr) == 0:
                    inr = cand[np.argsort(np.abs(coord[cand] - np.mean(rng)))[:1]]
                order = inr[np.argsort(coord[inr])]
                ok = 1
                (h0, h1), (v0, v1) = gdef["hbox"], gdef["vbox"]
                hc, vc, hh, vh = (h0 + h1) / 2, (v0 + v1) / 2, (h1 - h0) / 2 * sc, (v1 - v0) / 2 * sc
                dst = np.float32([[0, 0], [W, 0], [0, H]])
                # --chan seqfuse: the non-fluid series of the same plane, matched in PATIENT mm (each slice is warped
                # with its own geometry, so the two sequences are registered in the tile)
                plo, phi = BLOCKS[PARTNER[gdef["block"]]][:2] if (self.chan == "seqfuse" and gdef["block"] in PARTNER) else (0, 0)
                pvalid = np.where(msk[plo:phi])[0]
                pcoord = (slice_centres(geom[plo:phi]) - P) @ nrm if len(pvalid) else None
                for t, c in enumerate(self._pick(order, n)):
                    c = int(c)
                    pos[t] = int(np.clip((coord[c] - rng[0]) / max(rng[1] - rng[0], 1e-3) * N_POS, 0, N_POS - 1))
                    corners = []
                    for a, b in ((-hh, -vh), (hh, -vh), (-hh, vh)):
                        a, b = a * cr - b * sr, a * sr + b * cr
                        corners.append(P + eh * (hc + a) + ev * (vc + b))
                    corners = np.array(corners)
                    M = cv2.getAffineTransform(to_pixel(geom[lo + c], corners).astype(np.float32), dst)
                    if len(pvalid):
                        k2 = int(pvalid[np.argmin(np.abs(pcoord[pvalid] - coord[c]))])
                        M2 = cv2.getAffineTransform(to_pixel(geom[plo + k2], corners).astype(np.float32), dst)
                        fl = cv2.warpAffine(np.ascontiguousarray(vol[lo + c]), M, (W, H), flags=cv2.INTER_LINEAR) / 255.0
                        nf = cv2.warpAffine(np.ascontiguousarray(vol[plo + k2]), M2, (W, H), flags=cv2.INTER_LINEAR) / 255.0
                        tiles[t, 0], tiles[t, 1], tiles[t, 2] = fl, nf, np.clip(0.5 + (fl - nf) / 2, 0, 1)
                        continue
                    # neighbours ordered along the canonical normal so mirrored knees stay consistent
                    nb = (c - 1, c, c + 1) if coord[c + 1] >= coord[c - 1] else (c + 1, c, c - 1)
                    for ch, k in enumerate(nb):
                        tiles[t, ch] = cv2.warpAffine(np.ascontiguousarray(vol[lo + k]), M, (W, H),
                                                      flags=cv2.INTER_LINEAR) / 255.0
            if self.aug:
                tiles = np.clip(tiles * (1.0 + (random.random() - 0.5) * 0.25) + (random.random() - 0.5) * 0.06, 0, 1)
            out[f"x{gi}"] = torch.from_numpy(tiles)
            out[f"p{gi}"] = torch.from_numpy(pos)
            out[f"v{gi}"] = torch.tensor(ok)
        out["var"] = torch.tensor(vi)
        return out, torch.from_numpy(y), torch.from_numpy(w)


def collate(batch):
    x = {k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0]}
    return x, torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch])


# ------------------------------------------------------------------ model
class RegionNet(nn.Module):
    def __init__(self, arch, n_groups, n_var, n_out, pretrained=True, d=384, heads=6, mix_layers=2,
                 drop=0.1, grad_ckpt=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
        if grad_ckpt:
            self.backbone.set_grad_checkpointing(True)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        self.proj = nn.Sequential(nn.Linear(self.backbone.num_features, d), nn.LayerNorm(d))
        self.gemb = nn.Embedding(n_groups, d)
        self.pemb = nn.Embedding(N_POS, d)
        self.vemb = nn.Embedding(n_var, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, mix_layers, enable_nested_tensor=False)
        self.q = nn.Parameter(torch.randn(n_out, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.cls_w = nn.Parameter(torch.randn(n_out, d) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        self.n_groups = n_groups

    def forward(self, x):
        toks, valid = [], []
        for g in range(self.n_groups):
            t = x[f"x{g}"]
            B, N = t.shape[:2]
            f = self.backbone((t.flatten(0, 1) - self.mean) / self.std).view(B, N, -1)
            toks.append(self.proj(f.float()) + self.gemb.weight[g] + self.pemb(x[f"p{g}"]))
            valid.append(x[f"v{g}"].bool()[:, None].expand(B, N))
        h, valid = torch.cat(toks, 1), torch.cat(valid, 1)
        valid = valid | (~valid.any(1, keepdim=True))
        h = self.mix(h + self.vemb(x["var"])[:, None], src_key_padding_mask=~valid)
        q = self.q[None].expand(h.shape[0], -1, -1) + self.vemb(x["var"])[:, None]
        o, _ = self.xattn(q, h, h, key_padding_mask=~valid)
        o = self.norm(o + q)
        return (o * self.cls_w).sum(-1) + self.cls_b


# ------------------------------------------------------------------ labels
def load_targets(a, corpus):
    tr = pd.read_csv(a.train_csv or _find("train.csv"))
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr[tr[LAB].notna().all(1)].set_index("StudyInstanceUID")[LAB].astype(np.float32)
    hold = set(json.load(open(a.holdout)))
    lab = pd.read_parquet(a.labels) if a.labels.endswith(".parquet") else pd.read_csv(a.labels)
    lab["StudyInstanceUID"] = lab["StudyInstanceUID"].astype(str)
    lab = lab.set_index("StudyInstanceUID")[LAB].astype(np.float32)
    have = set(corpus.ids)
    labels, weights = {}, {}
    for u, r in lab.iterrows():
        if u in have and u not in gold.index:
            labels[u] = r.values
            weights[u] = 0.25 + 0.75 * np.clip(np.abs(2 * r.values - 1), 0, 1)
    for u, r in gold.iterrows():
        if u in have:
            labels[u] = r.values
            weights[u] = np.full(len(LAB), a.gold_w, np.float32)
    holdout = [u for u in gold.index if u in hold and u in have]
    dev = [u for u in gold.index if u not in hold and u in have]
    folds = json.load(open(a.fold_file))["folds"]
    nonexp = [u for u in lab.index if u in have and u not in gold.index]
    oof = [u for u in nonexp if folds.get(u, -1) == a.fold] if a.fold >= 0 else []
    oofset = set(oof)
    return labels, weights, [u for u in nonexp if u not in oofset] + dev, oof, holdout


def load_aux(path, w):
    """Report severity grades (scripts/graded_labels.py: 0 explicitly normal, 1 not mentioned, 2 low-grade, 3 moderate,
    4 severe) -> {uid: (ordinal targets (12, 3) = grade >= AUX_THR, weight (12,))}. A finding that the report does not
    mention is unknown (weight 0), not negative. `w` is the summed weight of the three side outputs of a known finding."""
    g = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    g["StudyInstanceUID"] = g["StudyInstanceUID"].astype(str)
    G = g.set_index("StudyInstanceUID")[LAB].to_numpy(np.float32)
    thr = np.array(AUX_THR, np.float32)
    out = {}
    for u, r in zip(g["StudyInstanceUID"], G):
        known = np.isfinite(r) & (r != 1)
        out[u] = ((np.nan_to_num(r)[:, None] >= thr).astype(np.float32), known.astype(np.float32) * w / len(AUX_THR))
    return out


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = (np.asarray(y) >= 0.5).astype(int)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--region", required=True, choices=sorted(REGIONS))
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--fold_file", default=None)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--side_file", default=None, help="raptor_side_*.npz (default: search)")
    ap.add_argument("--fold", type=int, default=0, help="-1 = train on all non-holdout studies")
    ap.add_argument("--arch", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--bb_lr", type=float, default=6e-5)
    ap.add_argument("--head_lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--gold_w", type=float, default=3.0)
    ap.add_argument("--eval_passes", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rs")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dump", default=None, help="write a jpg of the first samples' tiles and exit")
    ap.add_argument("--chan", default="adj", choices=["adj", "seqfuse"],
                    help="tile channels: adj = 3 adjacent slices; seqfuse = fluid-sensitive slice, the non-fluid slice of the "
                         "same plane nearest in patient mm, and their difference (fluid-sensitive groups only)")
    ap.add_argument("--aux_grades", default=None,
                    help="report severity grades (labels_graded_v1.parquet): three extra ordinal outputs for the OA finding "
                         "of each variant (tf, pf), trained with a masked loss and ignored at test time")
    ap.add_argument("--aux_w", type=float, default=0.5, help="summed loss weight of the three side outputs")
    ap.add_argument("--predict", default=None,
                    help="comma list of raptor_rs_*.pt checkpoints of ONE region (paths or names under the search "
                         "roots): score every study of the corpus without labels, mean over the checkpoints, and write "
                         "raptor_rspred_<tag>.npz (ids, pred (n, 12), NaN outside the region) or, for the side region, "
                         "raptor_side_<tag>.npz (ids, p_left). Used by the test-time leg (ours_leg_rs.py)")
    ap.add_argument("--only_findings", default=None,
                    help="--predict: semicolon list of findings; only the variants that hold one of them are scored "
                         "(tf: 'Lateral OA' = the lateral compartment alone, half the cost); the others stay NaN")
    ap.add_argument("--channels_last", action="store_true", help="--predict: backbone in channels-last memory format")
    ap.add_argument("--threads", type=int, default=0, help="--predict: torch CPU threads (0 = torch's default)")
    ap.add_argument("--eval_tiles", default="eval", choices=["eval", "train"],
                    help="--predict: tiles per group = the evaluation count (default) or the smaller training count")
    a = ap.parse_args()
    if a.predict:
        return predict_only(a)
    a.fold_file = a.fold_file or _find("folds_nonexpert_5.json")
    a.holdout = a.holdout or _find("expert_holdout.json")
    if not a.fold_file or not a.holdout:
        raise SystemExit("folds_nonexpert_5.json / expert_holdout.json not found")
    if a.smoke:
        a.epochs, a.limit, a.eval_passes, a.eval_every = 1, 24, 1, 1
        a.workers = min(a.workers, 2)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    reg = REGIONS[a.region]
    side_mode = a.region == "side"

    corpus = Corpus(a.n_shards, a.data_root)
    labels, weights, train_ids, oof_ids, hold_ids = load_targets(a, corpus)
    a.side_file = a.side_file or (None if side_mode else _find_side())
    left, src = corpus.left_prob(a.side_file)
    cols = [[L.get(f, -1) for f in finds] for _, finds in reg["variants"]]
    aux = None
    if a.aux_grades:
        if not all(any(f in AUX_FINDS for f in finds) for _, finds in reg["variants"]):
            raise SystemExit(f"--aux_grades: region {a.region} has a variant without an OA finding")
        path = a.aux_grades if os.path.exists(a.aux_grades) else _find(os.path.basename(a.aux_grades))
        if not path:
            raise SystemExit(f"--aux_grades: {a.aux_grades} not found")
        aux = load_aux(path, a.aux_w)
        n_known = {f: int(sum(aux[u][1][L[f]] > 0 for u in train_ids if u in aux)) for _, fs in reg["variants"]
                   for f in fs if f in AUX_FINDS}
        print(f"aux grades {os.path.basename(path)} | train studies with a known grade {n_known} | weight {a.aux_w}", flush=True)
    n_aux = len(AUX_THR) if aux is not None else 0
    if side_mode:                 # target = Laterality tag; only tagged studies train / score
        tagged = corpus.tag_left
        labels = {u: tagged[u] for u in tagged}
        weights = dict.fromkeys(tagged, 1.0)
        train_ids = [u for u in train_ids if u in tagged]
        oof_ids = [u for u in oof_ids + hold_ids if u in tagged]
        hold_ids = []
    if a.limit:
        train_ids = train_ids[: a.limit]; oof_ids = oof_ids[: max(8, a.limit // 2)]
    n_src = {s: sum(v == s for v in src.values()) for s in ("tag", "model", "x")}
    print(f"device {dev} | region {a.region} | arch {a.arch} | fold {a.fold} | train {len(train_ids)} | "
          f"oof {len(oof_ids)} | holdout {len(hold_ids)} | side source {n_src}", flush=True)

    mk = lambda ids, train, phase=0.0: RegionTiles(corpus, a.region, ids, labels, weights, left, train,
                                                    eval_phase=phase, side_mode=side_mode, chan=a.chan, aux=aux)
    if a.dump:
        ds = mk((train_ids + oof_ids)[:4], False)
        rows = []
        for i in range(len(ds)):
            x, y, _ = ds[i]
            for g in range(len(reg["groups"])):
                t = (x[f"x{g}"][:, 1].numpy() * 255).astype(np.uint8)
                rows.append(cv2.resize(np.hstack(list(t[:4])), None, fx=0.6, fy=0.6))
        w = max(r.shape[1] for r in rows)
        cv2.imwrite(a.dump, np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]))) for r in rows]))
        print("dumped", a.dump, [ds.items[i] for i in range(len(ds))], [left[u] for u, _ in ds.items])
        return 0

    tl = DataLoader(mk(train_ids, True), batch_size=a.bs, shuffle=True, drop_last=True,
                    num_workers=a.workers, collate_fn=collate, pin_memory=dev == "cuda",
                    persistent_workers=a.workers > 0,
                    worker_init_fn=lambda w: np.random.seed((a.seed * 1000 + w + int(time.time())) % 2**31))
    n_main = len(cols[0])
    n_out = n_main + n_aux        # outputs: the region's findings, then the ordinal side outputs (training only)
    model = RegionNet(a.arch, len(reg["groups"]), len(reg["variants"]), n_out,
                      pretrained=not a.no_pretrained, grad_ckpt=a.grad_ckpt).to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": hd, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(1, a.epochs * (len(tl) // a.accum))
    warm = max(1, int(a.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    if side_mode:
        pw = torch.ones(1, len(cols), n_out, device=dev)
    else:
        Y = np.stack([labels[u] for u in train_ids])
        prev = np.clip(np.stack([Y[:, c].mean(0) for c in cols]), 0.03, 0.7)        # (variants, n_out)
        pw = torch.tensor(np.clip((1 - prev) / prev, 1, 6), dtype=torch.float32, device=dev)[None]
        pw = torch.cat([pw, torch.ones(1, len(cols), n_aux, device=dev)], 2)           # side outputs: no class weight
    ema = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    def fwd(x):
        x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            return model(x).float(), x["var"]

    def predict(ids, passes):
        """-> (len(ids), 12) with NaN for findings outside the region (side: column 0 = p_left)."""
        model.eval()
        acc = 0
        for p in range(passes):
            ds = mk(ids, False, 0.0 if passes == 1 else p / passes)
            dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=collate)
            out = []
            with torch.no_grad():
                for x, _, _ in dl:
                    out.append(torch.sigmoid(fwd(x)[0]).cpu().numpy())
            acc = acc + np.concatenate(out)
        acc = (acc / passes).reshape(len(ids), len(cols), n_out)
        P = np.full((len(ids), len(LAB)), np.nan, np.float32)
        for v, c in enumerate(cols):
            for j, k in enumerate(c):
                P[:, k if k >= 0 else 0] = acc[:, v, j]
        return P

    def score(P, ids):
        if side_mode:
            y = np.array([labels[u] for u in ids])
            return {"auc": auc(y, P[:, 0]), "acc": float(((P[:, 0] >= 0.5) == (y >= 0.5)).mean())}
        Yt = np.stack([labels[u] for u in ids])
        return {LAB[k]: auc(Yt[:, k], P[:, k]) for c in cols for k in c}

    def swap_in(sd):
        cur = model.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                cur[k].copy_(v.to(cur[k].dtype))

    hist, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        model.train(); tot = 0.0; n = 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y, w) in enumerate(tl):
            logit, var = fwd(x)
            y, w = y.to(dev), w.to(dev)
            per = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw[0][var], reduction="none") * w
            loss = per[:, :n_main].mean() + per[:, n_main:].sum(1).mean()   # the finding loss keeps its scale; side term added
            (loss / a.accum).backward()
            if (it + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                with torch.no_grad():
                    d = min(a.ema, (1 + step) / (10 + step))
                    for k, v in model.state_dict().items():
                        if k in ema:
                            ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            tot += loss.item(); n += 1
            if it % 100 == 0:
                print(f"  ep{ep} it{it}/{len(tl)} loss {tot / n:.4f} | {time.time() - t0:.0f}s", flush=True)
        rec = {"ep": ep, "loss": tot / max(n, 1), "sec": round(time.time() - t0)}
        if oof_ids and ((ep + 1) % a.eval_every == 0 or ep == a.epochs - 1):
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            swap_in(ema)
            rec["oof_ema"] = {k: round(v, 4) for k, v in score(predict(oof_ids, 1), oof_ids).items()}
            swap_in(live)
        hist.append(rec)
        print(f"ep{ep} {json.dumps(rec)}", flush=True)

    last = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    meta_ck = {"arch": a.arch, "region": a.region, "spec": reg, "lab": LAB, "fold": a.fold, "seed": a.seed, "chan": a.chan,
               "n_aux": n_aux}
    torch.save({"model": last, **meta_ck}, os.path.join(a.out, f"raptor_rs_{a.tag}_last.pt"))
    res = {"tag": a.tag, "args": vars(a), "hist": hist, "n_train": len(train_ids), "n_oof": len(oof_ids),
           "n_holdout": len(hold_ids), "members": ["last", "ema"], "side_source": n_src}
    all_ids = list(corpus.ids) if side_mode else []
    po, ph, pa = [], [], []
    for name, sd in (("last", None), ("ema", ema)):
        if sd is not None:
            swap_in(sd)
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **meta_ck},
                       os.path.join(a.out, f"raptor_rs_{a.tag}.pt"))
        if oof_ids:
            po.append(predict(oof_ids, a.eval_passes))
        if hold_ids:
            ph.append(predict(hold_ids, a.eval_passes))
        if all_ids and not a.smoke:
            pa.append(predict(all_ids, 1))
    if oof_ids:
        Po = np.mean(po, 0)
        res["oof"] = score(Po, oof_ids)
        if not side_mode:
            np.savez(os.path.join(a.out, f"raptor_oof_{a.tag}_fold{a.fold}.npz"), pred=Po,
                     truth=np.stack([labels[u] for u in oof_ids]), ids=np.array(oof_ids), fold=a.fold,
                     pred_last=po[0], pred_ema=po[1])
    if hold_ids:
        Ph = np.mean(ph, 0)
        res["holdout29"] = score(Ph, hold_ids)
        np.savez(os.path.join(a.out, f"raptor_holdout_{a.tag}.npz"), pred=Ph,
                 truth=np.stack([labels[u] for u in hold_ids]), ids=np.array(hold_ids),
                 pred_last=ph[0], pred_ema=ph[1])
    if pa:
        p_left = np.mean(pa, 0)[:, 0]
        np.savez(os.path.join(a.out, f"raptor_side_{a.tag}.npz"), ids=np.array(all_ids), p_left=p_left)
        x_left = (corpus.P[:, 0] > 0)
        res["side_all"] = {"n": len(all_ids), "frac_left": float((p_left >= 0.5).mean()),
                           "agree_sign_x": float(((p_left >= 0.5) == x_left).mean()),
                           "unsure": int(((p_left > 0.1) & (p_left < 0.9)).sum())}
    json.dump(res, open(os.path.join(a.out, f"raptor_ft_{a.tag}.json"), "w"), indent=1, default=str)
    print("RESULT " + json.dumps({k: res[k] for k in ("oof", "holdout29", "side_all") if k in res}), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


def predict_only(a):
    """Labels-free scoring of a whole corpus with trained checkpoints (same tiles and passes as training-time predict)."""
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    paths = [x if os.path.exists(x) else _find(os.path.basename(x)) for x in a.predict.split(",") if x]
    if not paths or not all(paths):
        raise SystemExit(f"--predict: checkpoint not found among {a.predict}")
    cks = [torch.load(x, map_location="cpu", weights_only=False) for x in paths]
    region = cks[0]["region"]
    assert all(c["region"] == region and c["arch"] == cks[0]["arch"] for c in cks), "one region / arch per call"
    reg, side_mode = REGIONS[region], region == "side"
    corpus = Corpus(a.n_shards, a.data_root)
    ids = list(corpus.ids)
    left, src = corpus.left_prob(None if side_mode else a.side_file)
    cols = [[L.get(f, -1) for f in finds] for _, finds in reg["variants"]]
    vsel = list(range(len(cols)))
    if a.only_findings:
        want = {f.strip() for f in a.only_findings.split(";") if f.strip()}
        vsel = [v for v, (_, finds) in enumerate(reg["variants"]) if want & set(finds)]
        if not vsel:
            raise SystemExit(f"--only_findings {sorted(want)}: none of them belongs to region {region}")
    n_out = len(cols[0]) + cks[0].get("n_aux", 0)     # side outputs of --aux_grades models are scored and dropped below
    assert all(c.get("n_aux", 0) == cks[0].get("n_aux", 0) for c in cks), "checkpoints with and without side outputs: call twice"
    zero = np.zeros(len(LAB), np.float32)
    labels, weights = ({}, {}) if side_mode else (dict.fromkeys(ids, zero), dict.fromkeys(ids, zero))
    model = RegionNet(cks[0]["arch"], len(reg["groups"]), len(reg["variants"]), n_out, pretrained=False).to(dev)
    model.eval()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.channels_last:
        model.backbone.to(memory_format=torch.channels_last)
        model.backbone.register_forward_pre_hook(lambda m, x: (x[0].contiguous(memory_format=torch.channels_last),))
    n_src = {k: sum(v == k for v in src.values()) for k in ("tag", "model", "x")}
    print(f"predict | device {dev} | region {region} | {len(paths)} checkpoint(s) | studies {len(ids)} | "
          f"passes {a.eval_passes} | variants {vsel} | tiles {a.eval_tiles} | threads {torch.get_num_threads()} | "
          f"side source {n_src}", flush=True)
    acc = 0
    for ci, ck in enumerate(cks):
        model.load_state_dict(ck["model"], strict=True)
        for p in range(a.eval_passes):
            ds = RegionTiles(corpus, region, ids, labels, weights, left, False,
                             eval_phase=0.0 if a.eval_passes == 1 else p / a.eval_passes, side_mode=side_mode,
                             chan=ck.get("chan", "adj"), variants=vsel, few=a.eval_tiles == "train")
            dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=collate)
            out = []
            with torch.no_grad():
                for x, _, _ in dl:
                    x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
                    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):   # T4: no native bfloat16
                        out.append(torch.sigmoid(model(x).float()).cpu().numpy())
            acc = acc + np.concatenate(out)
            print(f"  checkpoint {os.path.basename(paths[ci])} pass {p} | {time.time() - t0:.0f}s", flush=True)
    acc = (acc / (len(cks) * a.eval_passes)).reshape(len(ids), len(vsel), n_out)
    P = np.full((len(ids), len(LAB)), np.nan, np.float32)
    for vi, v in enumerate(vsel):
        for j, k in enumerate(cols[v]):
            P[:, k if k >= 0 else 0] = acc[:, vi, j]
    if side_mode:
        np.savez(os.path.join(a.out, f"raptor_side_{a.tag}.npz"), ids=np.array(ids), p_left=P[:, 0])
    else:
        np.savez(os.path.join(a.out, f"raptor_rspred_{a.tag}.npz"), ids=np.array(ids), pred=P)
    print(f"predict done | {len(ids)} studies | {time.time() - t0:.0f}s", flush=True)
    return 0


def _find_side():
    for root in ("/workspace/out", HERE, "/workspace/bundle"):
        if os.path.isdir(root):
            for d, _, fs in os.walk(root):
                for f in fs:
                    if f.startswith("raptor_side_") and f.endswith(".npz") and "smoke" not in f:
                        return os.path.join(d, f)
    return None


if __name__ == "__main__":
    sys.exit(main())
