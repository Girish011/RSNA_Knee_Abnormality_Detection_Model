"""Minimal dense512 geometry primitives frozen from this project.

Coordinates are DICOM patient LPS millimetres. View axes canonicalize laterality;
actual sequence metadata and robust centres are supplied by definition_reader.
No training, public teacher or submission orchestration dependencies live here.
"""
import json
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

BLOCKS = {"SagF": (0, 22, "sag"), "SagNF": (22, 40, "sag"), "CorF": (40, 55, "cor"),
          "CorNF": (55, 65, "cor"), "Ax": (65, 80, "ax")}

IMG = 512

AXES = {"cor": ((1, 0, 0), (0, 0, -1)), "sag": ((0, -1, 0), (0, 0, -1)), "ax": ((1, 0, 0), (0, -1, 0))}

NORMAL = {"cor": (0, 1, 0), "sag": (1, 0, 0), "ax": (0, 0, 1)}

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
            ids.append(np.load(self._p(f"ids_s{i}.npy"), allow_pickle=False).astype(str))
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
            z = np.load(side_file, allow_pickle=False)
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
