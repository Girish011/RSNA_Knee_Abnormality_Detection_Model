#!/usr/bin/env python3
"""Family A: region-aware 2.5D specialist on the dense80 cache (weak five focus, all 12 trained).

Input: dense80 shards (vols_s{0..3}.npy (n,80,336,336) u8, masks, ids), slot layout
    SagF 0-21 | SagNF 22-39 | CorF 40-54 | CorNF 55-64 | Ax 65-79   (dense80_cache_template.py)

Model ("RegionMIL"):
  * every tile = 3 consecutive slots of one series (2.5D RGB); sagittal/axial tiles are the full
    140 mm field, coronal tiles are split into two overlapping halves (medial/lateral compartments)
    upsampled ×1.5, i.e. higher effective resolution where the menisci and tibiofemoral cartilage live;
  * a shared ConvNeXt/EfficientNet encoder (ImageNet-21k, NOT CoAt → diversity vs the stack) pools
    each tile to one vector; tokens get a group embedding (series × half) + slice-position embedding;
  * a small transformer mixes tokens across planes (lets the model tell medial from lateral from
    anatomy, e.g. the fibula), then 12 finding queries cross-attend over the tokens with a routing
    mask (each finding only reads the series that show it, e.g. PF OA ← Ax + Sag).
  * 3D mirror augmentation (a right knee mirrored is a valid left knee): coronal/axial images are
    flipped left-right and the sagittal slot order is reversed, consistently for the whole study.

Protocol (matches our CoAt 5-fold, so OOF is comparable): folds = folds_nonexpert_5.json, labels =
fused (soft, certainty-weighted), expert dev-29 trained at weight 3 with gold labels, expert
holdout-29 never trained on. No epoch selection (no leak): the EMA and last-epoch weights are the
members; OOF/holdout predictions are their mean. Outputs (raptor_* names so run_job.sh uploads them):
    raptor_fa_<tag>.pt (ema), raptor_fa_<tag>_last.pt, raptor_oof_<tag>_fold<f>.npz,
    raptor_holdout_<tag>.npz, raptor_ft_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEAK5 = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis", "Contusion"]

# (name, slot lo, slot hi, kind)  kind: "full" = one tile per window, "halves" = two half tiles
SERIES = [("SagF", 0, 22, "full"), ("SagNF", 22, 40, "full"), ("CorF", 40, 55, "halves"),
          ("CorNF", 55, 65, "halves"), ("Ax", 65, 80, "full")]
# token groups (series × half) -> index for the group embedding / routing
GROUPS = ["SagF", "SagNF", "CorF-a", "CorF-b", "CorNF-a", "CorNF-b", "Ax"]
G = {g: i for i, g in enumerate(GROUPS)}
_COR = ["CorF-a", "CorF-b", "CorNF-a", "CorNF-b"]
ROUTES = {  # finding -> groups its query may attend to
    "ACL": ["SagF", "SagNF", *_COR, "Ax"],
    "MCL": [*_COR, "Ax"],
    "Medial Meniscus": ["SagF", "SagNF", *_COR],
    "Lateral Meniscus": ["SagF", "SagNF", *_COR],
    "Medial OA": ["SagF", "SagNF", *_COR],
    "Lateral OA": ["SagF", "SagNF", *_COR],
    "PF OA": ["Ax", "SagF", "SagNF"],
    "Effusion": ["Ax", "SagF", "SagNF"],
    "Synovitis": ["Ax", "SagF", "SagNF"],
    "Baker's": ["Ax", "SagF", "SagNF"],
    "Contusion": ["SagF", "CorF-a", "CorF-b", "Ax"],
    "Fracture": ["SagF", "SagNF", *_COR, "Ax"],
}
N_POS = 8  # slice-position buckets within a series


# ------------------------------------------------------------------ corpus
def _find(name, roots=(HERE, "/workspace/bundle", "/workspace/out", "/kaggle/input", "/workspace/input")):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d, _, fs in os.walk(root):
            if name in fs:
                return os.path.join(d, name)
    return None


class Corpus:
    """Memory-mapped dense80 shards (opened lazily per worker)."""

    def __init__(self, n_shards=4, root=None):
        self.n_shards, self.root = n_shards, root
        self.vols = None
        ids, masks = [], []
        for i in range(n_shards):
            ids.append(np.load(self._p(f"ids_s{i}.npy"), allow_pickle=True).astype(str))
            masks.append(np.load(self._p(f"masks_s{i}.npy")))
        self.ids = np.concatenate(ids)
        self.masks = np.concatenate(masks).astype(bool)
        self.sizes = [len(x) for x in ids]
        self.offsets = np.cumsum([0] + self.sizes)
        self.row = {u: i for i, u in enumerate(self.ids)}

    def _p(self, name):
        p = os.path.join(self.root, name) if self.root else _find(name)
        if not p or not os.path.exists(p):
            raise SystemExit(f"dense80 file {name} not found (fetch with runpod/fetch_dense80.py)")
        return p

    def study(self, row):
        if self.vols is None:
            self.vols = [np.load(self._p(f"vols_s{i}.npy"), mmap_mode="r") for i in range(self.n_shards)]
        s = int(np.searchsorted(self.offsets, row, side="right") - 1)
        return self.vols[s][row - self.offsets[s]]


# ------------------------------------------------------------------ data
class StudyTiles(Dataset):
    """One study -> full tiles (Nf,3,R,R) + half tiles (Nh,3,Hh,Wh) with group/pos/valid vectors.

    n_win: windows per series (train: random centres, eval: evenly spaced). Tile counts are fixed
    so batches stack; windows of a missing series are zero tiles with valid=0.
    """

    def __init__(self, corpus, ids, labels, weights, n_win, train, res_full=320,
                 half_hw=(384, 288), mirror=0.5, eval_phase=0.0):
        self.c, self.ids, self.labels, self.weights = corpus, list(ids), labels, weights
        self.n_win, self.train = n_win, train
        self.res_full, self.half_hw = res_full, tuple(half_hw)
        self.mirror = mirror if train else 0.0
        self.eval_phase = eval_phase

    def __len__(self):
        return len(self.ids)

    def _centres(self, valid, n):
        cs = valid[(valid - 1 >= valid.min()) & (valid + 1 <= valid.max())]
        if len(cs) == 0:
            cs = valid[len(valid) // 2: len(valid) // 2 + 1]
        if self.train:
            return np.sort(np.random.choice(cs, n, replace=len(cs) < n))
        frac = (np.linspace(0.0, 1.0, n) + self.eval_phase) % 1.0
        return cs[np.round(frac * (len(cs) - 1)).astype(int)]

    def _jitter(self, span):
        return random.randint(-span, span) if self.train else 0

    def __getitem__(self, i):
        uid = self.ids[i]
        row = self.c.row[uid]
        vol = self.c.study(row)                      # (80,H,W) u8 memmap view
        msk = self.c.masks[row]
        H, W = vol.shape[1:]
        flip = self.train and random.random() < self.mirror
        full, half, gf, gh, pf, ph, vf, vh = [], [], [], [], [], [], [], []
        for name, lo, hi, kind in SERIES:
            n = self.n_win[name]
            valid = np.where(msk[lo:hi])[0]
            if len(valid) < 3:                        # missing series -> padded, masked tokens
                cs, ok = np.zeros(n, int), 0
            else:
                cs, ok = self._centres(valid, n), 1
            for c in cs:
                c = int(np.clip(c, 1, hi - lo - 2))
                pos = min(N_POS - 1, int(N_POS * c / max(1, hi - lo)))
                if flip and name.startswith("Sag"):   # mirror: reverse the medial-lateral order
                    pos = N_POS - 1 - pos
                tri = np.asarray(vol[lo + c - 1: lo + c + 2], dtype=np.float32) / 255.0  # (3,H,W)
                if flip and not name.startswith("Sag"):
                    tri = tri[:, :, ::-1]
                if kind == "full":
                    m = (H - min(H, self.res_full)) // 2
                    y0 = int(np.clip(m + self._jitter(8), 0, H - min(H, self.res_full)))
                    x0 = int(np.clip(m + self._jitter(8), 0, W - min(W, self.res_full)))
                    t = tri[:, y0:y0 + self.res_full, x0:x0 + self.res_full]
                    full.append(self._resize(t, (self.res_full, self.res_full)))
                    gf.append(G[name]); pf.append(pos); vf.append(ok)
                else:
                    # central 76% of rows (joint line) × 57% of width, two overlapping halves
                    hh, ww = int(round(H * 0.76)), int(round(W * 0.57))
                    y0 = int(np.clip((H - hh) // 2 + self._jitter(10), 0, H - hh))
                    for side, xs in (("a", 0), ("b", W - ww)):
                        x0 = int(np.clip(xs + self._jitter(8), 0, W - ww))
                        half.append(self._resize(tri[:, y0:y0 + hh, x0:x0 + ww], self.half_hw))
                        gh.append(G[f"{name}-{side}"]); ph.append(pos); vh.append(ok)
        x_full = torch.from_numpy(np.stack(full))
        x_half = torch.from_numpy(np.stack(half))
        if self.train:
            g = 1.0 + (random.random() - 0.5) * 0.25
            b = (random.random() - 0.5) * 0.06
            x_full = (x_full * g + b).clamp(0, 1)
            x_half = (x_half * g + b).clamp(0, 1)
        meta = {k: torch.tensor(v, dtype=torch.long) for k, v in
                (("gf", gf), ("gh", gh), ("pf", pf), ("ph", ph), ("vf", vf), ("vh", vh))}
        y = torch.from_numpy(self.labels[uid].astype(np.float32))
        w = torch.from_numpy(self.weights[uid].astype(np.float32))
        return x_full, x_half, meta, y, w

    @staticmethod
    def _resize(t, hw):
        t = torch.from_numpy(np.ascontiguousarray(t))
        if tuple(t.shape[-2:]) != tuple(hw):
            t = F.interpolate(t[None], size=hw, mode="bilinear", align_corners=False)[0]
        return t.numpy()


def collate(batch):
    xf = torch.stack([b[0] for b in batch])
    xh = torch.stack([b[1] for b in batch])
    meta = {k: torch.stack([b[2][k] for b in batch]) for k in batch[0][2]}
    return xf, xh, meta, torch.stack([b[3] for b in batch]), torch.stack([b[4] for b in batch])


# ------------------------------------------------------------------ model
class RegionMIL(nn.Module):
    def __init__(self, arch, pretrained=True, d=384, heads=6, mix_layers=2, drop=0.1, grad_ckpt=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
        if grad_ckpt:
            self.backbone.set_grad_checkpointing(True)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        self.proj = nn.Sequential(nn.Linear(self.backbone.num_features, d), nn.LayerNorm(d))
        self.gemb = nn.Embedding(len(GROUPS), d)
        self.pemb = nn.Embedding(N_POS, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, mix_layers, enable_nested_tensor=False)
        self.q = nn.Parameter(torch.randn(len(LAB), d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.cls_w = nn.Parameter(torch.randn(len(LAB), d) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LAB)))
        route = torch.zeros(len(LAB), len(GROUPS), dtype=torch.bool)
        for j, f in enumerate(LAB):
            for g in ROUTES[f]:
                route[j, G[g]] = True
        self.register_buffer("route", route)
        self.heads = heads

    def _enc(self, x):
        B, N = x.shape[:2]
        f = self.backbone(((x.flatten(0, 1)) - self.mean) / self.std)
        return f.view(B, N, -1)

    def forward(self, xf, xh, meta):
        tok = torch.cat([self._enc(xf), self._enc(xh)], 1)             # (B,T,F)
        grp = torch.cat([meta["gf"], meta["gh"]], 1)                    # (B,T)
        pos = torch.cat([meta["pf"], meta["ph"]], 1)
        valid = torch.cat([meta["vf"], meta["vh"]], 1).bool()
        valid = valid | (~valid.any(1, keepdim=True))                   # all-missing study: keep all
        h = self.proj(tok.float()) + self.gemb(grp) + self.pemb(pos)
        h = self.mix(h, src_key_padding_mask=~valid)
        allow = self.route[:, grp].permute(1, 0, 2) & valid[:, None, :]  # (B,12,T)
        allow = allow | (~allow.any(-1, keepdim=True) & valid[:, None, :])  # routed series missing
        q = self.q[None].expand(h.shape[0], -1, -1)
        mask = (~allow).repeat_interleave(self.heads, 0)                 # (B*heads,12,T) True=block
        o, _ = self.xattn(q, h, h, attn_mask=mask)
        o = self.norm(o + q)
        return (o * self.cls_w).sum(-1) + self.cls_b


# ------------------------------------------------------------------ labels
def load_targets(a, corpus):
    tcsv = a.train_csv or _find("train.csv")
    tr = pd.read_csv(tcsv)
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr[tr[LAB].notna().all(1)].set_index("StudyInstanceUID")[LAB].astype(np.float32)
    hold = set(json.load(open(a.holdout))) if a.holdout else set()
    lab = (pd.read_parquet(a.labels) if a.labels.endswith(".parquet") else pd.read_csv(a.labels))
    lab["StudyInstanceUID"] = lab["StudyInstanceUID"].astype(str)
    lab = lab.set_index("StudyInstanceUID")[LAB].astype(np.float32)
    have = set(corpus.ids)
    lw = np.array([a.weak_w if f in WEAK5 else 1.0 for f in LAB], np.float32)
    lw = lw / lw.mean()
    labels, weights = {}, {}
    for u, r in lab.iterrows():
        if u in have and u not in gold.index:
            y = r.values
            labels[u] = y
            weights[u] = (0.25 + 0.75 * np.clip(np.abs(2 * y - 1), 0, 1)) * lw
    for u, r in gold.iterrows():
        if u in have:
            labels[u] = r.values
            weights[u] = np.full(len(LAB), a.gold_w, np.float32) * lw
    holdout = [u for u in gold.index if u in hold and u in have]
    dev = [u for u in gold.index if u not in hold and u in have]
    folds = json.load(open(a.fold_file))["folds"]
    nonexp = [u for u in lab.index if u in have and u not in gold.index]
    oof = [u for u in nonexp if folds.get(u, -1) == a.fold] if a.fold >= 0 else []
    oofset = set(oof)
    train = [u for u in nonexp if u not in oofset] + dev
    return labels, weights, train, oof, holdout


def auc_table(P, Y):
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, f in enumerate(LAB):
        y = (Y[:, j] >= 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out[f] = float(roc_auc_score(y, P[:, j]))
    return out


def summary(t):
    w5 = [t[f] for f in WEAK5 if f in t]
    return {"macro": float(np.mean(list(t.values()))) if t else float("nan"),
            "weak5": float(np.mean(w5)) if w5 else float("nan")}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--fold_file", default=None)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--fold", type=int, default=0, help="-1 = train on all non-holdout studies")
    ap.add_argument("--arch", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--res_full", type=int, default=320)
    ap.add_argument("--half_h", type=int, default=384)
    ap.add_argument("--half_w", type=int, default=288)
    ap.add_argument("--win", default="SagF:6,SagNF:3,CorF:5,CorNF:3,Ax:5",
                    help="train windows per series (coronal windows give 2 tiles each)")
    ap.add_argument("--win_eval", default="SagF:10,SagNF:5,CorF:7,CorNF:4,Ax:7")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--bb_lr", type=float, default=6e-5)
    ap.add_argument("--head_lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--weak_w", type=float, default=2.0, help="loss multiplier on the weak five")
    ap.add_argument("--gold_w", type=float, default=3.0)
    ap.add_argument("--mirror", type=float, default=0.5)
    ap.add_argument("--eval_passes", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="fa")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.fold_file = a.fold_file or _find("folds_nonexpert_5.json")
    a.holdout = a.holdout or _find("expert_holdout.json")
    if not a.fold_file or not a.holdout:
        raise SystemExit("folds_nonexpert_5.json / expert_holdout.json not found")
    if a.smoke:
        a.epochs, a.limit, a.eval_passes, a.eval_every = 1, 24, 1, 1
        a.workers = min(a.workers, 2)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    parse = lambda s: {k: int(v) for k, v in (p.split(":") for p in s.split(","))}
    win, win_eval = parse(a.win), parse(a.win_eval)
    os.makedirs(a.out, exist_ok=True)

    corpus = Corpus(a.n_shards, a.data_root)
    labels, weights, train_ids, oof_ids, hold_ids = load_targets(a, corpus)
    if a.limit:
        train_ids = train_ids[: a.limit]; oof_ids = oof_ids[: max(8, a.limit // 2)]
    print(f"device {dev} | arch {a.arch} | fold {a.fold} | train {len(train_ids)} | oof {len(oof_ids)} "
          f"| holdout {len(hold_ids)} | win {win} | eval {win_eval}", flush=True)

    kw = dict(res_full=a.res_full, half_hw=(a.half_h, a.half_w))
    tds = StudyTiles(corpus, train_ids, labels, weights, win, True, mirror=a.mirror, **kw)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers,
                    collate_fn=collate, pin_memory=dev == "cuda", persistent_workers=a.workers > 0,
                    worker_init_fn=lambda w: np.random.seed((a.seed * 1000 + w + int(time.time())) % 2**31))
    model = RegionMIL(a.arch, pretrained=not a.no_pretrained, grad_ckpt=a.grad_ckpt).to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": hd, "lr": a.head_lr}],
                            weight_decay=a.wd)
    steps = max(1, a.epochs * (len(tl) // a.accum))
    warm = max(1, int(a.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    prev = np.clip(np.mean([labels[u] for u in train_ids], 0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 6), dtype=torch.float32, device=dev)
    ema = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    def predict(ids, passes):
        model.eval()
        acc = 0
        for p in range(passes):
            ds = StudyTiles(corpus, ids, labels, weights, win_eval, False,
                            eval_phase=0.0 if passes == 1 else p / passes, **kw)
            dl = DataLoader(ds, batch_size=max(1, a.bs // 2), shuffle=False, num_workers=a.workers,
                            collate_fn=collate)
            out = []
            with torch.no_grad():
                for xf, xh, meta, _, _ in dl:
                    meta = {k: v.to(dev) for k, v in meta.items()}
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        out.append(torch.sigmoid(model(xf.to(dev), xh.to(dev), meta).float()).cpu().numpy())
            acc = acc + np.concatenate(out)
        return acc / passes

    def swap_in(sd):
        cur = model.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                cur[k].copy_(v.to(cur[k].dtype))

    hist, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        model.train(); tot = 0.0; n = 0
        opt.zero_grad(set_to_none=True)
        for it, (xf, xh, meta, y, w) in enumerate(tl):
            meta = {k: v.to(dev, non_blocking=True) for k, v in meta.items()}
            y, w = y.to(dev), w.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logit = model(xf.to(dev, non_blocking=True), xh.to(dev, non_blocking=True), meta)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pw,
                                                       reduction="none") * w).mean()
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
            t = auc_table(predict(oof_ids, 1), np.stack([labels[u] for u in oof_ids]))
            swap_in(live)
            rec.update({f"oof_ema_{k}": v for k, v in summary(t).items()})
            rec["oof_ema_weak"] = {f: round(t[f], 4) for f in WEAK5 if f in t}
        hist.append(rec)
        print(f"ep{ep} {json.dumps(rec)}", flush=True)

    # ---- members: EMA + last; predictions = mean over members ----
    last = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    meta_ck = {"arch": a.arch, "lab": LAB, "res_full": a.res_full, "half_hw": [a.half_h, a.half_w],
               "win_eval": win_eval, "groups": GROUPS, "routes": ROUTES, "fold": a.fold, "seed": a.seed}
    torch.save({"model": last, **meta_ck}, os.path.join(a.out, f"raptor_fa_{a.tag}_last.pt"))
    preds_oof, preds_hold = [], []
    for name, sd in (("last", None), ("ema", ema)):
        if sd is not None:
            swap_in(sd)
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **meta_ck},
                       os.path.join(a.out, f"raptor_fa_{a.tag}.pt"))
        if oof_ids:
            preds_oof.append(predict(oof_ids, a.eval_passes))
        if hold_ids:
            preds_hold.append(predict(hold_ids, a.eval_passes))
    res = {"tag": a.tag, "args": vars(a), "hist": hist, "n_train": len(train_ids),
           "n_oof": len(oof_ids), "n_holdout": len(hold_ids), "members": ["last", "ema"]}
    if oof_ids:
        Yo = np.stack([labels[u] for u in oof_ids])
        Po = np.mean(preds_oof, 0)
        np.savez(os.path.join(a.out, f"raptor_oof_{a.tag}_fold{a.fold}.npz"), pred=Po, truth=Yo,
                 ids=np.array(oof_ids), fold=a.fold, members=np.array(["last", "ema"]),
                 pred_last=preds_oof[0], pred_ema=preds_oof[1])
        t = auc_table(Po, Yo)
        res["oof_vs_fused"] = {**summary(t), "per": t}
    if hold_ids:
        Yh = np.stack([labels[u] for u in hold_ids])
        Ph = np.mean(preds_hold, 0)
        np.savez(os.path.join(a.out, f"raptor_holdout_{a.tag}.npz"), pred=Ph, truth=Yh,
                 ids=np.array(hold_ids), pred_last=preds_hold[0], pred_ema=preds_hold[1])
        t = auc_table(Ph, Yh)
        res["holdout29"] = {**summary(t), "per": t}
    json.dump(res, open(os.path.join(a.out, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
    print("RESULT " + json.dumps({k: res[k] for k in ("oof_vs_fused", "holdout29") if k in res}), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
