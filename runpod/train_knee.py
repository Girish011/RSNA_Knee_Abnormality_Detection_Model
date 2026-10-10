# %% [code]
# %% [code]
#!/usr/bin/env python3
"""Knee MRI: training the twelve-finding model

This is the training half of the model behind the public 0.924 inference notebook. It reads the
precomputed slice stacks, trains a CoAtNet backbone with a per-finding attention pooling head, and
writes a checkpoint you can drop straight into that notebook.

WHAT YOU NEED ATTACHED

  1. The preprocessed corpus, both parts:
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus          (3,200 studies)
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext      (1,207 studies)
     Every study is already reduced to a fixed 44 x 336 x 336 uint8 stack, so no DICOM reading
     happens here. The two parts concatenate in order.

  2. The competition data, for train.csv.

  3. Training labels, as a parquet with a StudyInstanceUID column and the twelve finding columns.
     THIS IS NOT PROVIDED, and it is the one thing you have to bring yourself. See below.

THE LABEL PROBLEM, WHICH IS THE REAL PROBLEM

The competition gives you 4,407 studies and structured labels for only 58 of them. Every other
study carries a free-text radiology report and nothing else. So before any of this trains, you need
to turn 4,349 reports into twelve numbers each.

The approach behind the published weights was to read each report with a language model and emit
twelve probabilities rather than twelve yes or no answers: a report that hedges, saying a tear is
suspected, becomes something near 0.8 rather than a 1. Soft targets are far more forgiving than
forcing every hedged sentence into a hard label, and the loss here expects them. The 58 studies
that come with real labels are held out and used only for validation, never trained on.

Point --labels at your own parquet built that way. The format is one row per study:
a StudyInstanceUID column plus the twelve finding columns, values between 0 and 1.

WHAT THE MODEL DOES

Three neighbouring slices are stacked into the three channels of one image, so the network sees a
little of what lies above and below the middle slice: most of the benefit of a 3D model at the cost
of a 2D one. Each of these three-slice windows goes through the backbone, and the windows are then
pooled by an attention layer that has separate weights for each of the twelve findings. That last
part matters more than anything else here. A cruciate tear may be visible on two slices while
osteoarthritis spreads across many, and one shared pooling weight forces those to compete; giving
each finding its own attention lets each draw on the slices that actually show it.

Training samples k windows per study at random and evaluates on k_eval windows spread evenly, so
each epoch sees a different view of the same study. Nothing else is augmented.

At the end it keeps the best epoch by validation macro-AUC, and also writes a checkpoint that
averages the weights of the best three epochs. Weight averaging costs nothing at inference, unlike
averaging predictions from three models, and it usually gives a small gain.

TYPICAL RUN

  python train_knee.py --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 16
      --bs 8 --k 12 --k_eval 24 --grad_ckpt --tag mymodel --labels /kaggle/input/YOURS/labels.parquet

About three hours on one 4090 for 16 epochs at 384. --grad_ckpt trades a little speed for a lot of
memory and is what makes bs 8 fit on a 24 GB card. Use --smoke for a fast wiring check.

MEMORY, MEASURED

The CoAtNet at 384 with --bs 8 and --grad_ckpt sits at about 23.7 GB, so it needs a 24 GB card.
On 16 GB use --bs 2 or 4. Separately from the GPU, this script keeps the best checkpoint plus the
top --topk epochs' weights in system RAM so it can average them at the end; that is about 87 MB a
copy for the small ViT and 293 MB for the CoAtNet, and the list fills over the first few epochs.
If you run out of host memory a few epochs in, set --topk 1 and lower --workers.

The checkpoint it writes is a dict with keys model, arch, res and lab, which is exactly what the
inference notebook expects.
"""
import os, sys, time, json, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

# --- ROI localizer (anatomical joint crop). Optional so the no-ROI path is untouched. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from roi_localize import square_box as _roi_square_box, compartments as _roi_compartments
except Exception:
    _roi_square_box = _roi_compartments = None

HERE = os.path.dirname(os.path.abspath(__file__))
RSNA = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Input discovery. On Kaggle the corpus arrives as two read-only datasets and the
# competition data as a third, so nothing lives beside this script. Everything in
# this block is discovery only - the training code further down is unchanged.
# ---------------------------------------------------------------------------
def _find(*names, root="/kaggle/input"):
    """First path under root whose basename matches one of names."""
    for d, _, fs in os.walk(root):
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None


class _MultiPartVols:
    """Memory-mapped concat of corpus parts (raptor two-part or dense80 four shards).

    Row order is part0 then part1 then ...; never sort parts independently of ids/masks.
    """
    def __init__(self, parts):
        self.parts = list(parts)
        self.offsets = []
        n = 0
        for p in self.parts:
            self.offsets.append(n)
            n += p.shape[0]
        self.shape = (n,) + tuple(self.parts[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        for part, off in zip(self.parts, self.offsets):
            if row < off + part.shape[0]:
                return part[row - off]
        raise IndexError(row)


class _TwoPartVols(_MultiPartVols):
    """Back-compat wrapper for the public 44-slot two-part corpus."""
    def __init__(self, a, b):
        super().__init__([a, b])
        self.a, self.b, self.n_a = a, b, a.shape[0]


def _ext_parts(kind):
    """Sorted <kind>_e<k>.npy files of the external cache (env KNEE_EXT_DIR, set by --ext_dir), or []."""
    d = os.environ.get("KNEE_EXT_DIR", "")
    if not d or not os.path.isdir(d):
        return []
    fs = [f for f in os.listdir(d) if f.startswith(kind + "_e") and f.endswith(".npy")]
    return [os.path.join(d, f) for f in sorted(fs, key=lambda f: int(f[len(kind) + 2:-4]))]


def _open_corpus():
    """Return (vols, masks), from local file, dense80 shards, or the public two-part corpus."""
    local_v = os.path.join(HERE, "all_vols.npy")
    if os.path.exists(local_v):
        return (np.load(local_v, mmap_mode="r"),
                np.load(os.path.join(HERE, "all_masks.npy")))
    # Own dense80 cache: vols_s{0..3}.npy + masks_s{0..3}.npy (80 slots, submit geometry).
    shard_vs = [_find(f"vols_s{i}.npy") for i in range(4)]
    shard_ms = [_find(f"masks_s{i}.npy") for i in range(4)]
    if all(shard_vs) and all(shard_ms):
        vols = _MultiPartVols([np.load(p, mmap_mode="r") for p in shard_vs + _ext_parts("vols")])
        masks = np.concatenate([np.load(p) for p in shard_ms + _ext_parts("masks")], axis=0)
        return vols, masks
    av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
    am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
    if not all((av, bv, am, bm)):
        raise SystemExit("Could not find the corpus. Attach both parts: "
                         "dreaddevelopment/knee-raptor-corpus and "
                         "dreaddevelopment/knee-raptor-corpus-ext "
                         "(or dense80 vols_s0..s3 / masks_s0..s3)")
    vols = _TwoPartVols(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    return vols, masks


def _open_ids():
    local = os.path.join(HERE, "all_ids.npy")
    if os.path.exists(local):
        return np.load(local, allow_pickle=True).astype(str)
    shard_ids = [_find(f"ids_s{i}.npy") for i in range(4)]
    if all(shard_ids):
        return np.concatenate([np.load(p, allow_pickle=True).astype(str) for p in shard_ids + _ext_parts("ids")])
    a, b = _find("all_ids.npy"), _find("extra_ids.npy")
    if not (a and b):
        raise SystemExit("Could not find all_ids.npy / extra_ids.npy - attach both corpus parts "
                         "(or dense80 ids_s0..s3).")
    return np.concatenate([np.load(a, allow_pickle=True).astype(str),
                           np.load(b, allow_pickle=True).astype(str)])
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]


_NO_LABELS = """
No training labels found, so there is nothing to train against.

The competition labels only 58 of the 4,407 studies. The other 4,349 carry a free-text
radiology report instead, so before this can train you have to turn those reports into
twelve probabilities per study and pass the result with --labels.

Expected format: a parquet with a StudyInstanceUID column plus the columns
  {cols}
with values between 0 and 1. Soft values work better than hard 0/1 here: the loss is built
for them, and hedged reports are common.

Everything else in this notebook is ready to run once that file exists.
"""


# ------------------------------- data ----------------------------------------
class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from all_vols.npy (memmap).
    Each window = 3 physically-consecutive slices -> RGB, resized to `res`, in [0,1]
    (matches the SSL input pipeline: ToTensor, no ImageNet norm)."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none", aug_geo=0.0,
                 aug_flip=0.0, aug_erase=0.0, ext_weight=1.0, roi=False, roi_mode="tight", roi_pad=0.06, roi_overlap=0.12,
                 roi_sbox=None, roi_cen=None, eval_phase=0.0, sample="rand", sides=None):
        self.root = root
        self.sides = sides                # uid -> True for a left knee: shown mirrored, so every knee reads as a right knee
        self.sample = sample              # "rand" = any slot of the volume; "strat" = ordered windows from every series
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.aug_geo = float(aug_geo)     # 0 = off; 1 = +-12 deg, +-12 % scale, +-7 % shift per window, gamma per study
        self.aug_flip = float(aug_flip)   # probability of a left-right mirror of the whole study (dense80 only)
        self.aug_erase = float(aug_erase) # per-window probability of blanking one random rectangle
        self.ext_weight = float(ext_weight)   # loss weight of external studies (ids starting with "fm_")
        self.norm = norm                  # "none"=[0,1] (Raptor SSL); "imagenet"=DINOv2 stats
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
        # --- eval TTA: shift the deterministic eval window grid by this fraction of the span ---
        self.eval_phase = float(eval_phase)
        # --- ROI anatomical joint-crop config ---
        self.roi = bool(roi)
        self.roi_mode, self.roi_pad, self.roi_overlap = roi_mode, roi_pad, roi_overlap
        self.roi_sbox = roi_sbox          # (N,D,4) int16 per-slice tissue bbox, or None
        self.roi_cen = roi_cen            # (N,D,2) int16 per-slice joint centroid, or None
        if self.roi and (_roi_square_box is None or roi_sbox is None):
            raise RuntimeError("roi=True but roi_localize or roi_boxes not available")

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = _open_corpus()   # (N,D,H,W) view, (N,D) u8

    def _centers(self, valid, count):
        # valid slice indices; window centers must have both neighbors valid & in-range
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return pool[:count]
        # eval: evenly spaced deterministic. eval_phase>0 shifts the grid, so successive TTA
        # passes see *different* (overlapping) subsets of the slices instead of the same ones.
        frac = (np.linspace(0.0, 1.0, count) + self.eval_phase) % 1.0
        idx = np.round(frac * (len(cs) - 1)).astype(int)
        return [cs[i] for i in idx]

    def _centers_strat(self, m, count):
        """Series-aware window centres: every filled series block gets its share of the windows, spread in slice
        order over the block (jittered when training), and no window straddles two series."""
        spans = []
        for s, e in BLOCKS:
            v = np.where(m[s:e] > 0)[0]
            if len(v) == 0:
                continue
            lo, hi = s + int(v.min()) + 1, s + int(v.max()) - 1
            if hi < lo:
                lo = hi = max(s + 1, min(s + int(v[len(v) // 2]), e - 2))
            spans.append((lo, hi))
        if not spans:
            return [1] * count
        cs = []
        for (lo, hi), n in zip(spans, _alloc([hi - lo + 1 for lo, hi in spans], count)):
            for j in range(n):
                u = (j + random.random()) / n if self.train else ((j + 0.5 + self.eval_phase) / n) % 1.0
                cs.append(lo + int(round(u * (hi - lo))))
        return cs

    def _resize(self, tri):
        """tri: (3,h,w) float32 [0,1] -> (3,res,res) float32."""
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        # compartment mode emits 2 crops/center -> sample ceil(k/2) centers to keep #windows==k
        compart = self.roi and self.roi_mode == "compartment"
        n_centers = (self.k + 1) // 2 if compart else self.k
        strat = self.sample == "strat" and self._D == 80 and not self.roi
        cs = self._centers_strat(m, n_centers) if strat else self._centers(valid, n_centers)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        tiles = []            # list of (3,res,res) float32
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            H, W = tri.shape[-2], tri.shape[-1]
            if not self.roi:
                tiles.append(self._resize(tri))
                continue
            # one joint box from the CENTER slice, applied to all 3 slices (keeps RGB registered)
            sq_mode = "tight" if compart else self.roi_mode
            sq = _roi_square_box(tuple(int(v) for v in self.roi_sbox[row, c]), W=W, H=H,
                                 pad=self.roi_pad, mode=sq_mode,
                                 centroid=tuple(float(v) for v in self.roi_cen[row, c]))
            if compart:
                left, right = _roi_compartments(sq, overlap=self.roi_overlap)
                for box in (left, right):
                    x0, y0, x1, y1 = box
                    tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
            else:
                x0, y0, x1, y1 = sq
                tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
        if len(tiles) > self.k:
            tiles = tiles[:self.k]
        wins = np.stack(tiles, 0)          # (K,3,res,res)
        x = torch.from_numpy(wins)
        pos = _slot_pos(cs[:len(tiles)], self._D)   # (K,2): series block, position inside the block (0..1)
        if self.sides is not None and self.sides.get(uid, False) and self._D == 80 and not self.roi:
            x = _mirror(x, cs)                  # canonical side, train and eval alike (use with --aug_flip 0)
            sag = pos[:, 0] < 2
            pos[sag, 1] = 1.0 - pos[sag, 1]
        if self.train and self.aug:
            # light medical-safe aug: NO flips (laterality is signal); mild intensity jitter
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
            if self.aug_flip > 0 and not self.roi and self._D == 80 and random.random() < self.aug_flip:
                x = _mirror(x, cs)
                sag = pos[:, 0] < 2             # a mirrored knee runs through its sagittal slices the other way
                pos[sag, 1] = 1.0 - pos[sag, 1]
            if self.aug_geo > 0:
                x = _geo_aug(x, self.aug_geo)
            if self.aug_erase > 0:
                x = _erase(x, self.aug_erase)
        if self.norm == "imagenet":        # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        return x, y, (self.ext_weight if uid.startswith("fm_") else 1.0), pos


def _geo_aug(x, s):
    """Spatial augmentation the base recipe lacks: one random affine per window (rotation, isotropic scale,
    shift; still NO flips), then one gamma for the study. x: (K,3,H,W) in [0,1]; s scales every range."""
    K = x.shape[0]
    ang = (torch.rand(K) * 2 - 1) * math.radians(12.0 * s)
    sc = 1.0 + (torch.rand(K) * 2 - 1) * 0.12 * s
    tx = (torch.rand(K) * 2 - 1) * 0.14 * s      # affine_grid units span [-1, 1]: 0.14 = 7 % of the width
    ty = (torch.rand(K) * 2 - 1) * 0.14 * s
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)   # (K,2,3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gam = math.exp((random.random() * 2 - 1) * 0.25 * s)
    return x.clamp(0, 1) ** gam


SAG_SLOTS = 40   # dense80 layout: slots 0-39 sagittal (22 fluid + 18 non-fluid), 40-64 coronal, 65-79 axial
BLOCKS = [(0, 22), (22, 40), (40, 55), (55, 65), (65, 80)]   # sag fluid, sag other, cor fluid, cor other, axial


def _alloc(sizes, count):
    """Split `count` windows over blocks in proportion to their sizes (largest remainder; every block >= 1 if it can)."""
    tot = float(sum(sizes)) or 1.0
    raw = [count * z / tot for z in sizes]
    n = [int(r) for r in raw]
    for i in sorted(range(len(sizes)), key=lambda i: raw[i] - n[i], reverse=True)[:count - sum(n)]:
        n[i] += 1
    for i in range(len(n)):
        if n[i] == 0 and max(n) > 1:
            n[n.index(max(n))] -= 1; n[i] = 1
    return n


def _slot_pos(cs, D):
    """(K,2) float32 per window: index of its series block and its position inside that block in [0,1]."""
    out = np.zeros((len(cs), 2), np.float32)
    for i, c in enumerate(cs):
        if D != 80:
            out[i] = (0.0, c / max(D - 1, 1)); continue
        for b, (s, e) in enumerate(BLOCKS):
            if s <= c < e:
                out[i] = (b, (c - s) / (e - s - 1.0)); break
    return torch.from_numpy(out)


def _fourier(rel):
    """(...,) position in [0,1] -> (...,12) sin / cos features at six frequencies."""
    f = rel[..., None] * (math.pi * (2.0 ** torch.arange(6, device=rel.device, dtype=rel.dtype)))
    return torch.cat([torch.sin(f), torch.cos(f)], -1)


def _mirror(x, cs):
    """Left-right mirror of the whole knee = a valid opposite-side knee with the same medial / lateral labels.
    Coronal and axial windows flip in-plane; sagittal windows reverse their slice (= channel) order."""
    out = x.clone()
    for i, c in enumerate(cs[:x.shape[0]]):
        out[i] = x[i].flip(0) if c < SAG_SLOTS else x[i].flip(-1)
    return out


def _erase(x, p):
    """Coarse dropout: with probability p per window, blank one rectangle of 10-30 % of each side."""
    K, _, H, W = x.shape
    for i in range(K):
        if random.random() < p:
            h, w = int(H * random.uniform(0.1, 0.3)), int(W * random.uniform(0.1, 0.3))
            y0, x0 = random.randint(0, H - h), random.randint(0, W - w)
            x[i, :, y0:y0 + h, x0:x0 + w] = 0
    return x


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,K,3,res,res)
    ys = torch.stack([b[1] for b in batch])   # (B,12)
    ws = torch.tensor([b[2] for b in batch], dtype=torch.float32)   # (B,) per-study loss weight
    ps = torch.stack([b[3] for b in batch])   # (B,K,2) slot info per window
    return xs, ys, ws, ps


# ------------------------------- model ---------------------------------------
def build_backbone(arch="vit_small_patch16_224", pretrained=False, drop_path=0.0):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    if drop_path > 0:   # stochastic depth (training only; adds no weights, so checkpoints load unchanged)
        kw.update(drop_path_rate=drop_path)
    return timm.create_model(arch, **kw)


def load_raptor(bb, ckpt_path):
    if ckpt_path in ("timm", "pretrained"):
        return "timm-pretrained"
    if ckpt_path in ("", "none", "None"):
        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)
        return "random-init"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["student"] if "student" in ck else ck
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"


class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x, pos=None):
        return self.head(self.encode(x))


class SeqClassifier(nn.Module):
    """RaptorClassifier that knows where each window comes from. Every added part is gated at zero, so training starts
    from the plain attention-MIL head and only moves away from it where that lowers the loss:
      slot : series block + position inside it, added to the input of the per-finding attention (always on);
      tr   : a small transformer over the study's windows (with slot embeddings), added to the window features -
             lets a window be read in the context of the others (which side of the knee, which sequence);
      sp   : per-finding spatial attention over the backbone's feature map instead of its global average."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2, use_tr=True, use_sp=False, D=384, layers=2, heads=6):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n, self.use_tr, self.use_sp = n, bool(use_tr), bool(use_sp)
        self.blk_att = nn.Embedding(len(BLOCKS), F_dim)
        self.pos_att = nn.Linear(12, F_dim)
        for t in (self.blk_att.weight, self.pos_att.weight, self.pos_att.bias):
            nn.init.zeros_(t)
        if self.use_tr:
            self.proj_in = nn.Linear(F_dim, D)
            self.blk_tr = nn.Embedding(len(BLOCKS), D)
            self.pos_tr = nn.Linear(12, D)
            layer = nn.TransformerEncoderLayer(D, heads, 2 * D, dropout=0.1, batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.proj_out = nn.Linear(D, F_dim)
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.use_sp:
            self.sp_norm = nn.LayerNorm(F_dim)
            self.sp = nn.Linear(F_dim, n)
            nn.init.zeros_(self.sp.weight); nn.init.zeros_(self.sp.bias)

    def forward(self, x, pos=None):
        B, K = x.shape[:2]
        if self.use_sp:
            fm = self.backbone.forward_features(x.flatten(0, 1))          # (BK,C,h,w)
            t = fm.flatten(2).transpose(1, 2)                              # (BK,hw,C)
            s = torch.softmax(self.sp(self.sp_norm(t)).float(), dim=1).to(t.dtype)   # (BK,hw,n); zero init = uniform
            g = torch.einsum("thn,thc->tnc", s, t)                         # (BK,n,C) one pooled vector per finding
            f = self.backbone.forward_head(g.reshape(-1, g.shape[-1], 1, 1)).view(B, K, self.n, -1)
        else:
            f = self.backbone(x.flatten(0, 1)).view(B, K, 1, -1)
        e_att = 0.0
        if pos is not None:
            blk, ff = pos[..., 0].long(), _fourier(pos[..., 1].float())
            e_att = (self.blk_att(blk) + self.pos_att(ff))[:, :, None, :]
            if self.use_tr:
                z = self.proj_in(self.norm(f.mean(2))) + self.blk_tr(blk) + self.pos_tr(ff)
                f = f + self.gamma * self.proj_out(self.tr(z))[:, :, None, :]
        h = self.norm(f)                                                   # (B,K,n or 1,F)
        a = self.att(h + e_att)                                            # (B,K,n or 1,n)
        a = torch.diagonal(a, dim1=2, dim2=3) if self.use_sp else a[:, :, 0]
        a = torch.softmax(a.float(), dim=1).to(h.dtype)                    # (B,K,n) over the windows
        pooled = torch.einsum("bkn,bknf->bnf", a, h.expand(-1, -1, self.n, -1))
        return (pooled * self.clsW).sum(-1) + self.clsb


# ------------------------------- train ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Defaults to ImageNet-pretrained weights, which is what every released arm was trained from.
    # The previous default pointed at a self-supervised checkpoint that is not distributed here,
    # so following the example run above failed on a missing file.
    ap.add_argument("--ckpt", default="timm",
                    help="'timm' for ImageNet-pretrained weights, or a path to your own "
                         "self-supervised checkpoint")
    ap.add_argument("--init_model", default=None,
                    help="warm-start the WHOLE classifier (backbone + attention head) from a "
                         "raptor_ft_*.pt of the same arch, e.g. the public v5 full SWA")
    ap.add_argument("--eval_only", action="store_true",
                    help="score --init_model on gold-dev + holdout and exit (no training)")
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze_blocks", type=int, default=0)
    # Each kept epoch is a full copy of the weights in system RAM (~87 MB small ViT, ~293 MB
    # CoAtNet). Lower to 1 on a memory-tight machine; at 1 no averaged checkpoint is written.
    ap.add_argument("--topk", type=int, default=3,
                    help="top epochs kept for weight averaging (1 disables averaging)")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--labels", default=None,
                    help="parquet of training labels: StudyInstanceUID + the twelve finding "
                         "columns, values 0..1. Not provided with this notebook - see the header.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # ---- CV fold hook ----
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--fold_file", default=None)
    # ---- weak-label robustness knobs (both are inference-reproducible) ----
    ap.add_argument("--ema", type=float, default=0.999,
                    help="decay for the exponential moving average of the weights "
                         "(0 disables; the EMA checkpoint is saved as raptor_ft_<tag>_ema.pt)")
    ap.add_argument("--label_weights", default=None,
                    help="12 comma-separated loss multipliers in LAB order (rescaled to mean 1); "
                         "up-weights chosen findings in the BCE. Default: all 1")
    ap.add_argument("--aug_geo", type=float, default=0.0,
                    help="strength of the per-window random affine + gamma augmentation (0 = off, the base recipe)")
    ap.add_argument("--drop_path", type=float, default=0.0, help="stochastic-depth rate of the backbone (0 = off)")
    ap.add_argument("--aug_flip", type=float, default=0.0,
                    help="probability of mirroring the whole study left-right during training (0 = off; dense80 only)")
    ap.add_argument("--aug_erase", type=float, default=0.0,
                    help="per-window probability of blanking one random rectangle during training (0 = off)")
    ap.add_argument("--pos_weight_cap", type=float, default=10.0,
                    help="upper clip of the per-finding positive weight (1-prev)/prev in the BCE; 1 = no "
                         "up-weighting of positives (report YES is the noisy side). Default 10 = the base recipe")
    ap.add_argument("--eval_passes", type=int, default=2,
                    help="TTA passes at eval: averages the sigmoid over N shifted deterministic "
                         "window grids (1 = old behaviour, same windows every time)")
    # ---- anatomical ROI joint-crop (A/B lever). Default OFF -> identical to baseline. ----
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--roi_mode", default="compartment", choices=["tight", "safe", "compartment"])
    ap.add_argument("--roi_pad", type=float, default=0.06)
    ap.add_argument("--roi_overlap", type=float, default=0.12)
    ap.add_argument("--roi_boxes", default=os.path.join(HERE, "roi_boxes.npz"))
    # ---- expert holdout: these gold studies are neither trained on nor used for epoch selection;
    # they are predicted once at the end with the selected weights (raptor_holdout_<tag>.npz).
    ap.add_argument("--holdout", default=None, help="json list of expert StudyInstanceUIDs")
    # ---- external studies (fastMRI cache built on the pod): extra training rows, never held out or evaluated
    ap.add_argument("--ext_dir", default=None,
                    help="directory with vols_e<k>.npy / masks_e<k>.npy / ids_e<k>.npy (ids start with fm_); their "
                         "pseudo-labels must be rows of --labels")
    ap.add_argument("--ext_weight", type=float, default=1.0, help="loss weight of external studies (1 = like the rest)")
    # ---- slot-aware head (default "mil" = the base head; checkpoints of the base recipe are unchanged)
    ap.add_argument("--head", default="mil", choices=["mil", "seq", "tok", "seqtok"],
                    help="mil = attention-MIL over unordered windows; seq = + slot embeddings and a transformer over the "
                         "windows; tok = + slot embeddings and per-finding spatial attention; seqtok = both")
    ap.add_argument("--bb_file", default=None,
                    help="file name (searched under /kaggle/input) or path of a timm state dict for --arch, saved from "
                         "create_model(arch, pretrained=True, num_classes=0): the ImageNet weights without internet")
    ap.add_argument("--canon", default=None,
                    help="json {StudyInstanceUID: 'L' | 'R'} (file name searched under /kaggle/input, or a path): left "
                         "knees are mirrored at train and eval, so medial / lateral sit on the same image side and at "
                         "the same end of the sagittal stack in every study. Use with --aug_flip 0")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16"],
                    help="autocast precision: bf16 (A40 / 4090 pods, the base recipe) or fp16 with a gradient scaler "
                         "(Kaggle T4, which has no native bfloat16)")
    ap.add_argument("--sample", default="rand", choices=["rand", "strat"],
                    help="rand = window centres anywhere in the volume; strat = ordered windows from every series")
    a = ap.parse_args()
    if a.ext_dir:                       # read by _open_corpus / _open_ids, also in the DataLoader workers
        os.environ["KNEE_EXT_DIR"] = a.ext_dir
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
        a.eval_passes = 1
    print(f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag} | "
          f"roi={a.roi}({a.roi_mode}) | ema={a.ema} | eval_passes={a.eval_passes}", flush=True)

    # ---- ids / labels ----
    ids = _open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    _tcsv = os.path.join(RSNA, "train.csv")
    if not os.path.exists(_tcsv):
        _tcsv = _find("train.csv")
    if not _tcsv:
        raise SystemExit("Could not find train.csv - attach the competition data.")
    tr = pd.read_csv(_tcsv); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    holdout_ids = []
    if a.holdout:
        hset = set(json.load(open(a.holdout)))
        holdout_ids = [u for u in gold_ids if u in hset]
        gold_ids = [u for u in gold_ids if u not in hset]
        print(f"[holdout] {len(holdout_ids)} experts held out; gold-val {len(gold_ids)}", flush=True)
    _lab = a.labels or _find("labels_llm_soft.parquet")
    if not _lab or not os.path.exists(_lab):
        # Exit cleanly rather than as a failure: running this notebook as published, with no
        # labels attached, is the expected path and should read as an explanation, not a crash.
        print(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(0)
    soft = pd.read_parquet(_lab)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids) | set(holdout_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    # ---- CV fold hook: hold out fold `a.fold`, train on the rest ----
    oof_ids = []
    if a.folds > 0:
        assert 0 <= a.fold < a.folds, f"--fold must be in [0,{a.folds}) when --folds>0"
        fmap = json.load(open(a.fold_file))["folds"]
        held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
        oof_ids = [u for u in train_ids if u in held]
        train_ids = [u for u in train_ids if u not in held]
        print(f"[cv] fold {a.fold}/{a.folds}: train {len(train_ids)} | OOF held-out {len(oof_ids)} "
              f"| fold_file {os.path.basename(a.fold_file)}", flush=True)
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in oof_ids: labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids + holdout_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    n_ext = sum(u.startswith("fm_") for u in train_ids)
    print(f"train {len(train_ids)} | gold-val {len(gold_ids)}"
          + (f" | oof {len(oof_ids)}" if oof_ids else "")
          + (f" | external {n_ext} (loss weight {a.ext_weight})" if n_ext else ""), flush=True)

    # positive weights from the competition studies only: the external population has its own prevalence
    prev = np.clip(np.stack([labels[u] for u in train_ids if not u.startswith("fm_")]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, max(1.0, a.pos_weight_cap)), dtype=torch.float32, device=dev)
    if a.pos_weight_cap != 10.0:
        print(f"[loss] pos_weight cap {a.pos_weight_cap:g}: " + ", ".join(f"{n} {w:.1f}" for n, w in zip(LAB, pw.tolist())),
              flush=True)

    # ---- ROI boxes (only when --roi) ----
    roi_sbox = roi_cen = None
    if a.roi:
        rb = np.load(a.roi_boxes)
        rb_ids = rb["ids"].astype(str)
        if not np.array_equal(rb_ids, ids):
            rmap = {u: i for i, u in enumerate(rb_ids)}
            missing = [u for u in ids if u not in rmap]
            if missing:
                raise RuntimeError(f"roi_boxes missing {len(missing)} corpus ids (e.g. {missing[:2]})")
            order = np.array([rmap[u] for u in ids])
            roi_sbox = rb["sbox"][order]; roi_cen = rb["cen"][order]
        else:
            roi_sbox = rb["sbox"]; roi_cen = rb["cen"]
        print(f"[roi] ENABLED mode={a.roi_mode} pad={a.roi_pad} overlap={a.roi_overlap} "
              f"boxes={os.path.basename(a.roi_boxes)} sbox={roi_sbox.shape}", flush=True)
    _roi_kw = dict(roi=a.roi, roi_mode=a.roi_mode, roi_pad=a.roi_pad, roi_overlap=a.roi_overlap,
                   roi_sbox=roi_sbox, roi_cen=roi_cen)

    sides = None
    if a.canon:
        _cf = a.canon if os.path.exists(a.canon) else _find(os.path.basename(a.canon))
        if not _cf:
            raise SystemExit(f"--canon {a.canon} not found")
        sides = {u: v == "L" for u, v in json.load(open(_cf)).items()}
        _n = [sides.get(u) for u in train_ids]
        print(f"[canon] {os.path.basename(_cf)}: train left {sum(v is True for v in _n)} | right {sum(v is False for v in _n)} "
              f"| unknown (kept as is) {sum(v is None for v in _n)}", flush=True)
        if a.aug_flip > 0:
            print("[canon] WARNING: --aug_flip > 0 undoes the canonical side for part of the training studies", flush=True)
    tds = StudyWindows(HERE, train_ids, id2row, labels, a.res, a.k, train=True, norm=a.norm, aug_geo=a.aug_geo,
                       aug_flip=a.aug_flip, aug_erase=a.aug_erase, ext_weight=a.ext_weight, sample=a.sample, sides=sides, **_roi_kw)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)

    use_timm = a.ckpt in ("timm", "pretrained")
    bb = build_backbone(a.arch, pretrained=use_timm and not a.bb_file, drop_path=a.drop_path)
    if a.bb_file:
        _bf = a.bb_file if os.path.exists(a.bb_file) else _find(os.path.basename(a.bb_file))
        if not _bf:
            raise SystemExit(f"--bb_file {a.bb_file} not found")
        bb.load_state_dict(torch.load(_bf, map_location="cpu"), strict=True)
        print(f"[raptor] backbone weights from {_bf}", flush=True)
    if a.aug_geo or a.drop_path or a.aug_flip or a.aug_erase:
        print(f"[reg] aug_geo {a.aug_geo} | drop_path {a.drop_path} | aug_flip {a.aug_flip} | aug_erase {a.aug_erase}",
              flush=True)
    src = load_raptor(bb, a.ckpt)
    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)
    F_dim = bb.num_features
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True); print("[raptor] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)
    META = {"head": a.head, "sample": a.sample, "k_eval": a.k_eval, "canon": bool(a.canon)}   # what the scoring code needs to rebuild the model
    if a.head == "mil":
        model = RaptorClassifier(bb, F_dim=F_dim)
    else:
        model = SeqClassifier(bb, F_dim=F_dim, use_tr=a.head in ("seq", "seqtok"), use_sp=a.head in ("tok", "seqtok"))
        print(f"[head] {a.head} | sample {a.sample} | head parameters "
              f"{sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith('backbone.')) / 1e6:.2f}M", flush=True)
    if a.init_model:
        _ck = torch.load(a.init_model, map_location="cpu", weights_only=False)
        _st = _ck.get("model", _ck) if isinstance(_ck, dict) else _ck
        _miss, _unex = model.load_state_dict(_st, strict=False)
        print(f"[init] warm-start from {os.path.basename(a.init_model)} | missing {len(_miss)} "
              f"unexpected {len(_unex)}", flush=True)
        if len(_miss) > 0.1 * len(model.state_dict()):
            raise SystemExit(f"--init_model does not match {a.arch}: {len(_miss)} tensors missing")
        src = f"init:{os.path.basename(a.init_model)}"
        del _ck, _st
    model = model.to(dev)
    if a.freeze_blocks > 0:
        # Backbones name their layer groups differently: ViT-style models use blocks.N., while
        # CoAtNet, MaxViT and ConvNeXt use stages.N. Matching only one of those froze nothing at
        # all on the other family, and did it silently. Try both, and always say what happened.
        pref = tuple([f"blocks.{b}." for b in range(a.freeze_blocks)]
                     + [f"stages.{b}." for b in range(a.freeze_blocks)])
        frozen = 0
        for nm, p in model.backbone.named_parameters():
            if nm.startswith(pref):
                p.requires_grad = False
                frozen += p.numel()
        n_t = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        if frozen:
            print(f"[raptor] froze the first {a.freeze_blocks} layer group(s): "
                  f"{n_t} tensors, {frozen/1e6:.2f}M parameters", flush=True)
        else:
            print(f"[raptor] WARNING: --freeze_blocks {a.freeze_blocks} matched no parameters on "
                  f"{a.arch}; every parameter is still being trained", flush=True)

    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(len(tl) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=steps, pct_start=0.15)
    lw = None
    if a.label_weights:
        lw = np.array([float(x) for x in a.label_weights.split(",")], dtype=np.float32)
        assert lw.shape == (len(LAB),) and (lw > 0).all(), f"--label_weights needs {len(LAB)} positive values"
        lw = lw / lw.mean()
        print("[loss] label weights " + ", ".join(f"{n} {w:.2f}" for n, w in zip(LAB, lw)), flush=True)
        lw = torch.tensor(lw, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, weight=lw, reduction="none")
    amp_dtype = torch.float16 if a.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (a.amp == "fp16" and dev == "cuda") else None

    # ---- TTA inference helper: average sigmoid over shifted eval-window grids ----
    def _predict_ids(ids_sub, k_win, passes=None):
        npass = max(1, int(a.eval_passes if passes is None else passes))
        acc = None
        model.eval()
        for p in range(npass):
            phase = 0.0 if npass == 1 else p / float(npass)
            ds = StudyWindows(HERE, ids_sub, id2row, labels, a.res, k_win, train=False,
                              norm=a.norm, eval_phase=phase, sample=a.sample, sides=sides, **_roi_kw)
            dl = DataLoader(ds, batch_size=max(2, a.bs // 2), shuffle=False,
                            num_workers=a.workers, collate_fn=collate, persistent_workers=False)
            Ps = []
            with torch.no_grad():
                for x, _y, _w, pos in dl:
                    with torch.autocast(dev, dtype=amp_dtype, enabled=dev == "cuda"):
                        Ps.append(torch.sigmoid(model(x.to(dev), pos.to(dev)).float()).cpu().numpy())
            P = np.concatenate(Ps)
            acc = P if acc is None else acc + P
        model.eval()
        return acc / npass

    @torch.no_grad()
    def evaluate():
        model.eval()
        P = _predict_ids(gold_ids, a.k_eval)
        Y = np.stack([labels[u] for u in gold_ids]).astype(np.float32)
        aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    # ---- EMA of the weights: the gold-29 is far too small to pick an epoch reliably, and an
    # exponential average of the trajectory is the standard defence against that. ----
    ema = None
    if a.ema and a.ema > 0:
        with torch.no_grad():
            ema = {k: v.detach().float().clone()
                   for k, v in model.state_dict().items() if v.is_floating_point()}
        print(f"[ema] tracking an exponential moving average of {len(ema)} tensors "
              f"(decay {a.ema})", flush=True)

    def _apply_ema(ema_dict):
        sd = model.state_dict()
        with torch.no_grad():
            for k, v in ema_dict.items():
                sd[k].copy_(v.to(sd[k].dtype))

    if a.eval_only:
        # Score a finished checkpoint (public v5, a weight soup, ...) with the SAME windows,
        # dev split and holdout as every member, so it is directly comparable.
        assert a.init_model, "--eval_only needs --init_model"
        au, aucs, P, Y = evaluate()
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"), pred=P, truth=Y, ids=np.array(gold_ids))
        if holdout_ids:
            Ph = _predict_ids(holdout_ids, a.k_eval)
            np.savez(os.path.join(HERE, f"raptor_holdout_{a.tag}.npz"), pred=Ph,
                     truth=np.stack([labels[u] for u in holdout_ids]), ids=np.array(holdout_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": au, "aucs": aucs, "eval_only": True},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"[eval_only] {os.path.basename(a.init_model)} gold-dev {au:.4f} | holdout written", flush=True)
        return

    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
    TOPK = max(1, a.topk); topk = []
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, y, w, pos in tl:
            x, y, w = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), w.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(dev, dtype=amp_dtype, enabled=dev == "cuda"):
                logits = model(x, pos.to(dev, non_blocking=True))
                # per-study mean over findings, weighted mean over the batch (all weights 1 = the old loss)
                loss = (lossf(logits.float(), y).mean(1) * w).sum() / w.sum()
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt); scaler.update()
            sched.step(); tot += loss.item()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if k in ema:
                            ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
        au, aucs, P, Y = evaluate()
        hist.append({"ep": ep, "loss": tot / len(tl), "gold_auc": au})
        if au > best:
            best = au; best_P = P
            best_state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                          "arch": a.arch, "lab": LAB, "epoch": ep, **META}
        if au >= best and best_state is not None:
            _tmp = os.path.join(HERE, f"raptor_ft_{a.tag}.pt.tmp")
            torch.save(best_state, _tmp)
            os.replace(_tmp, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                     pred=best_P, truth=Y, ids=np.array(gold_ids))
            json.dump({"tag": a.tag, "src": src, "best_gold_auc": best,
                       "aucs": best_state["aucs"], "hist": hist, "res": a.res,
                       "epochs": a.epochs, "bb_lr": a.bb_lr, "head_lr": a.head_lr,
                       "n_train": len(train_ids), "n_gold": len(gold_ids),
                       "roi": a.roi, "roi_mode": a.roi_mode,
                       "ema": a.ema, "eval_passes": a.eval_passes,
                       "partial": True, "epochs_done": ep + 1},
                      open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
            print(f"  [ckpt] best-so-far saved at ep{ep} ({best:.4f})", flush=True)
        if len(topk) < TOPK or au > min(t["gold_auc"] for t in topk):
            topk.append({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                         "arch": a.arch, "lab": LAB, "epoch": ep, "P": P, **META})
            topk.sort(key=lambda t: -t["gold_auc"])
            del topk[TOPK:]
        print(f"ep{ep} loss {tot/len(tl):.3f} | GOLD macro-AUC {au:.4f} (best {best:.4f}) | {time.time()-t0:.0f}s",
              flush=True)
    # ---- snapshot the very end of the trajectory before any checkpoint is re-loaded. The gold-29
    # peaks early (measured: ep4) while the weak-label signal keeps rising, so 'best'/'epens'/'swa'
    # are all early-epoch based and the tail of the run is thrown away. Keeping the last-epoch
    # weights and letting them vote in the OOF ensemble costs one extra inference pass and stops
    # that mis-selection from silently discarding whatever the late epochs learned.
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _, aucs, _, Y = evaluate()
    print(f"\nDONE {src} | BEST GOLD macro-AUC {best:.4f}", flush=True)
    for k, v in (best_state["aucs"] if best_state else aucs).items():
        print(f"   {k:18s} {v:.3f}", flush=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                 pred=best_P, truth=Y, ids=np.array(gold_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "aucs": best_state["aucs"],
                   "hist": hist, "res": a.res, "epochs": a.epochs, "bb_lr": a.bb_lr,
                   "head_lr": a.head_lr, "n_train": len(train_ids), "n_gold": len(gold_ids),
                   "roi": a.roi, "roi_mode": a.roi_mode, "ema": a.ema,
                   "eval_passes": a.eval_passes, "label_weights": a.label_weights,
                   "partial": False, "epochs_done": a.epochs},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"saved raptor_ft_{a.tag}.pt / raptor_gold_{a.tag}.npz / raptor_ft_{a.tag}.json", flush=True)

    # ---- CV OOF: every surviving checkpoint is re-predicted on the held-out fold ----
    oof_comp = {}          # name -> (N,12) probs on the held-out fold at the k_eval windows
    gold_ema = None; gold_last = None
    if a.folds > 0 and oof_ids and best_state is not None:
        Yo = np.stack([labels[u] for u in oof_ids]).astype(np.float32)
        oof_ids_arr = np.array(oof_ids)
        model.load_state_dict(best_state["model"]); model.eval()
        Po = _predict_ids(oof_ids, a.k_eval)
        oof_comp["best"] = Po
        np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_best_fold{a.fold}.npz"),
                 pred=Po, truth=Yo, ids=oof_ids_arr, fold=a.fold, nfolds=a.folds)
        print(f"[cv] best-epoch OOF: {len(oof_ids)} studies, truth = soft labels", flush=True)
        # --- top-K epoch OOF (single pass each: the mean over epochs is the variance reducer) ---
        oof_by_ep = {best_state["epoch"]: Po}
        for t in topk:
            e = int(t["epoch"])
            if e in oof_by_ep:
                continue
            model.load_state_dict(t["model"]); model.eval()
            Pe = _predict_ids(oof_ids, a.k_eval, passes=1)
            oof_by_ep[e] = Pe
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_ep{e}_fold{a.fold}.npz"),
                     pred=Pe, truth=Yo, ids=oof_ids_arr, fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote top-K epoch OOF ep{e}", flush=True)
        if len(oof_by_ep) > 1:
            Pens = np.mean(list(oof_by_ep.values()), axis=0)
            oof_comp["epens"] = Pens
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_epens_fold{a.fold}.npz"),
                     pred=Pens, truth=Yo, ids=oof_ids_arr, fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote epoch-ensemble OOF over epochs {sorted(oof_by_ep)}", flush=True)
        # --- last-epoch member: a late-trajectory snapshot, averaged in like any other member ---
        model.load_state_dict(last_state); model.eval()
        au_last, aucs_last, _, _ = evaluate()
        gold_last = au_last
        Pl = _predict_ids(oof_ids, a.k_eval, passes=1)
        oof_comp["last"] = Pl
        np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_last_fold{a.fold}.npz"),
                 pred=Pl, truth=Yo, ids=oof_ids_arr, fold=a.fold, nfolds=a.folds)
        print(f"[cv] last-epoch member: gold {au_last:.4f}; OOF written (epoch {a.epochs - 1})",
              flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    if holdout_ids and best_state is not None:
        model.load_state_dict(best_state["model"]); model.eval()
        Ph = _predict_ids(holdout_ids, a.k_eval)
        np.savez(os.path.join(HERE, f"raptor_holdout_{a.tag}.npz"), pred=Ph,
                 truth=np.stack([labels[u] for u in holdout_ids]), ids=np.array(holdout_ids))
        print(f"[holdout] wrote raptor_holdout_{a.tag}.npz ({len(holdout_ids)} studies)", flush=True)

    # --- weight-averaged checkpoint (new) ---
    if len(topk) > 1:
        sds = [t["model"] for t in topk]
        avg = {}
        for k in sds[0]:
            v0 = sds[0][k]
            if v0.is_floating_point():
                avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
            else:
                avg[k] = v0.clone()          # e.g. num_batches_tracked
        swa_state = {"model": avg, "gold_auc": None, "aucs": {}, "src": src, "res": a.res,
                     "arch": a.arch, "lab": LAB, "epoch": [int(t["epoch"]) for t in topk],
                     "swa_over": [int(t["epoch"]) for t in topk], **META}
        model.load_state_dict(avg); model.eval()
        au_swa, aucs_swa, P_swa, Y_swa = evaluate()
        swa_state["gold_auc"] = au_swa; swa_state["aucs"] = aucs_swa
        torch.save(swa_state, os.path.join(HERE, f"raptor_ft_{a.tag}_swa.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_swa.npz"),
                 pred=P_swa, truth=Y_swa, ids=np.array(gold_ids))
        print(f"SWA over epochs {[int(t['epoch']) for t in topk]} | gold {au_swa:.4f} "
              f"(best-epoch {best:.4f}) {'BETTER' if au_swa > best else 'no gain'}", flush=True)
        if a.folds > 0 and oof_ids:
            Ps = _predict_ids(oof_ids, a.k_eval)
            oof_comp["swa"] = Ps
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_swa_fold{a.fold}.npz"),
                     pred=Ps, truth=np.stack([labels[u] for u in oof_ids]).astype(np.float32),
                     ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print("[cv] wrote SWA OOF", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    # --- EMA checkpoint (new): averaged weights, evaluated and OOF-predicted like any other ---
    if ema is not None and best_state is not None:
        _apply_ema(ema)
        au_ema, aucs_ema, P_ema, Y_ema = evaluate()
        gold_ema = au_ema
        ema_state = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "gold_auc": au_ema, "aucs": aucs_ema, "src": src, "res": a.res,
                     "arch": a.arch, "lab": LAB, "epoch": a.epochs - 1, "ema_decay": a.ema, **META}
        torch.save(ema_state, os.path.join(HERE, f"raptor_ft_{a.tag}_ema.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_ema.npz"),
                 pred=P_ema, truth=Y_ema, ids=np.array(gold_ids))
        print(f"EMA(decay {a.ema}) | gold {au_ema:.4f} (best-epoch {best:.4f}) "
              f"{'BETTER' if au_ema > best else 'no gain'}", flush=True)
        if a.folds > 0 and oof_ids:
            Pe = _predict_ids(oof_ids, a.k_eval)
            oof_comp["ema"] = Pe
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_ema_fold{a.fold}.npz"),
                     pred=Pe, truth=np.stack([labels[u] for u in oof_ids]).astype(np.float32),
                     ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print("[cv] wrote EMA OOF", flush=True)
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "ema_gold_auc": au_ema,
                   "last_gold_auc": gold_last,
                   "swa_gold_auc": (swa_state["gold_auc"] if len(topk) > 1 else None),
                   "aucs": best_state["aucs"], "aucs_ema": aucs_ema, "hist": hist,
                   "res": a.res, "epochs": a.epochs, "bb_lr": a.bb_lr, "head_lr": a.head_lr,
                   "n_train": len(train_ids), "n_gold": len(gold_ids), "roi": a.roi,
                   "roi_mode": a.roi_mode, "ema": a.ema, "eval_passes": a.eval_passes,
                   "aug_geo": a.aug_geo, "drop_path": a.drop_path, "pos_weight_cap": a.pos_weight_cap,
                   "aug_flip": a.aug_flip, "aug_erase": a.aug_erase, "ext_weight": a.ext_weight, "n_ext": n_ext,
                   "head": a.head, "sample": a.sample, "k": a.k, "k_eval": a.k_eval, "canon": bool(a.canon),
                   "oof_members": sorted(oof_comp), "partial": False, "epochs_done": a.epochs},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        model.load_state_dict(best_state["model"]); model.eval()

    # --- primary OOF: equal-weight average of this run's own checkpoints. Each member is a real
    # saved checkpoint (so the same ensemble is usable at submit time); no fold-0 labels are used
    # to weight them. The plain best-epoch file stays as ..._best_fold<f>.npz. ---
    if oof_comp:
        Yo = np.stack([labels[u] for u in oof_ids]).astype(np.float32)
        members = sorted(oof_comp)
        Po = np.mean([oof_comp[m] for m in members], axis=0)
        if len(members) > 1:
            for nm in members:
                pp = oof_comp[nm]
                _a = {}
                for j, name in enumerate(LAB):
                    if len(set(Yo[:, j].astype(int))) > 1:
                        _a[name] = float(roc_auc_score(Yo[:, j], pp[:, j]))
                if _a:
                    print(f"[oof] {nm:6s} vs soft truth macro-AUC {np.mean(list(_a.values())):.4f}",
                          flush=True)
        np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_fold{a.fold}.npz"),
                 pred=Po, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds,
                 members=np.array(members))
        print(f"[cv] primary OOF = mean of {members} ({len(oof_ids)} studies)", flush=True)

    if len(topk) > 1:
        eps = [t["epoch"] for t in topk]
        for rank, t in enumerate(topk):
            if rank == 0:
                continue
            P_t = t.pop("P")
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_ep{t['epoch']}.npz"),
                     pred=P_t, truth=Y, ids=np.array(gold_ids))
            t["P"] = P_t
        Pens = np.mean([t["P"] for t in topk], axis=0)
        ens_aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                ens_aucs[name] = float(roc_auc_score(Y[:, j], Pens[:, j]))
        ens = float(np.mean(list(ens_aucs.values())))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_epens.npz"),
                 pred=Pens, truth=Y, ids=np.array(gold_ids))
        print(f"TOPK epochs {eps} | best {best:.4f} | epoch-ensemble {ens:.4f} "
              f"({'BETTER' if ens > best else 'no gain'})", flush=True)


if __name__ == "__main__":
    main()
