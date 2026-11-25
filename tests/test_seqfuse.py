"""--chan seqfuse windows (fluid / non-fluid / difference): pairing, fallbacks, and train == test-time code."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "runpod" / "train_knee.py"
LEG = REPO / "outputs" / "kernels" / "_ours_leg" / "ours_leg_v2.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tk():
    return _load(TRAINER, "train_knee_seqfuse_test")


def _src(path: Path, names: set[str]) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out.append(ast.unparse(node))
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            out.append(ast.unparse(node))
    return out


def _vol():
    # each slot filled with its own index -> a channel's value tells which slot it came from
    vol = np.stack([np.full((8, 8), i * 3, np.uint8) for i in range(80)])
    return vol, np.ones(80, np.uint8)


def test_trainer_and_leg_share_the_same_code():
    names = {"SEQ_PAIRS", "_seqfuse_tri", "BLOCKS"}
    a, b = _src(TRAINER, names), _src(LEG, names)
    assert len(a) == 3 and a == b


@pytest.mark.parametrize("c,fluid,nonfluid", [
    (1, 1, 23),      # first usable sagittal fluid slot (slot 0 is clamped away): 22 + round(1/21 * 17)
    (21, 21, 39),    # last sagittal fluid -> last sagittal non-fluid (22 vs 18 slots)
    (11, 11, 31),    # middle: 22 + round(11/21 * 17) = 31
    (39, 21, 39),    # non-fluid centre: fluid partner comes first
    (40, 40, 55),
    (54, 54, 64),    # last coronal fluid -> last coronal other (15 vs 10 slots)
    (64, 54, 64),
])
def test_pairing_follows_relative_position(tk, c, fluid, nonfluid):
    vol, mask = _vol()
    s, e = next((s, e) for s, e in tk.BLOCKS if s <= c < e)
    b = tk.BLOCKS.index((s, e))
    s2, e2 = tk.BLOCKS[tk.SEQ_PAIRS[b]]
    p = s2 + int(round((c - s) / (e - s - 1.0) * (e2 - s2 - 1)))
    exp_f, exp_n = (c, p) if b % 2 == 0 else (p, c)
    assert (exp_f, exp_n) == (fluid, nonfluid)
    tri = tk._seqfuse_tri(vol, mask, c)
    assert tri.shape == (3, 8, 8) and tri.dtype == np.float32
    assert np.allclose(tri[0], exp_f * 3 / 255.0) and np.allclose(tri[1], exp_n * 3 / 255.0)
    assert np.allclose(tri[2], np.clip(0.5 + (tri[0] - tri[1]) / 2, 0, 1))
    assert tri.min() >= 0.0 and tri.max() <= 1.0


def test_axial_and_empty_partner_fall_back_to_adjacent(tk):
    vol, mask = _vol()
    adj = lambda c: np.stack([vol[c - 1], vol[c], vol[c + 1]]).astype(np.float32) / 255.0
    assert np.array_equal(tk._seqfuse_tri(vol, mask, 70), adj(70))          # axial
    mask[22:40] = 0                                                          # no sagittal non-fluid series
    assert np.array_equal(tk._seqfuse_tri(vol, mask, 10), adj(10))
    assert np.array_equal(tk._seqfuse_tri(vol, mask, 79), adj(78))          # clamped like the adjacent path
    v = np.zeros((30, 8, 8), np.uint8)                                       # not a dense80 volume
    assert tk._seqfuse_tri(v, np.ones(30), 5).shape == (3, 8, 8)


def test_dataset_adj_unchanged_and_seqfuse_differs(tk, tmp_path):
    rng = np.random.default_rng(0)
    vols = rng.integers(0, 255, (1, 80, 16, 16), dtype=np.uint8)
    labels = {"a": np.zeros(12, np.float32)}

    def first(**kw):
        d = tk.StudyWindows(str(tmp_path), ["a"], {"a": 0}, labels, res=16, k=6, train=False, sample="strat", **kw)
        d.vols, d.masks = vols, np.ones((1, 80), np.uint8)
        return d[0][0]

    x_default, x_adj, x_sf = first(), first(chan="adj"), first(chan="seqfuse")
    assert x_default.shape == x_sf.shape == (6, 3, 16, 16)
    assert torch.equal(x_default, x_adj)          # default = the old adjacent-slice windows
    assert not torch.equal(x_adj, x_sf)
