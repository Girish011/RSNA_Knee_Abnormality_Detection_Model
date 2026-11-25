"""Region specialists: geometry, crops and model shapes on a synthetic dense512 corpus."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("region_spec", REPO / "runpod" / "region_spec.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def _geom(plane, k, ps=0.3125, n=512, x_off=80.0):
    """One slice of an orthogonal stack centred on (x_off, 0, 0); 140 mm crop of a 160 mm field."""
    g = np.zeros(20, np.float32)
    half = ps * n / 2
    if plane == "sag":
        g[2:5] = (x_off + k, -half, half); g[5:8] = (0, 1, 0); g[8:11] = (0, 0, -1)
    elif plane == "cor":
        g[2:5] = (x_off - half, k, half); g[5:8] = (1, 0, 0); g[8:11] = (0, 0, -1)
    else:
        g[2:5] = (x_off - half, -half, k); g[5:8] = (1, 0, 0); g[8:11] = (0, 1, 0)
    cpx = int(round(140 / ps))
    g[0] = g[1] = ps; g[11] = g[12] = (n - cpx) // 2; g[13] = cpx; g[14] = g[15] = n; g[19] = 1
    return g


def _corpus(tmp_path, n=4):
    ids = np.array([f"s{i}" for i in range(n)])
    vols = np.random.randint(0, 255, (n, 80, 512, 512), np.uint8)
    geom = np.zeros((n, 80, 20), np.float32)
    for name, (lo, hi, plane) in rs.BLOCKS.items():
        for j, k in enumerate(np.linspace(-40, 40, hi - lo)):
            geom[:, lo + j] = _geom(plane, k)
    np.save(tmp_path / "vols_s0.npy", vols); np.save(tmp_path / "ids_s0.npy", ids)
    np.save(tmp_path / "masks_s0.npy", np.ones((n, 80), np.uint8)); np.save(tmp_path / "geom_s0.npy", geom)
    json.dump({u: [{"Laterality": "L" if i % 2 == 0 else ""}] for i, u in enumerate(ids)},
              open(tmp_path / "series_s0.json", "w"))
    return rs.Corpus(1, str(tmp_path))


def test_joint_centre_and_side(tmp_path):
    c = _corpus(tmp_path)
    assert np.allclose(c.P, [[80, 0, 0]] * 4, atol=1.0)
    left, src = c.left_prob()
    assert src["s0"] == "tag" and src["s1"] == "x" and left["s1"] == 1.0   # x > 0 -> left knee
    # the joint centre projects to the middle of every slice
    for lo in (0, 40, 65):
        assert np.allclose(rs.to_pixel(c.geom[0, lo + 5], c.P[:1]), 256, atol=2.0)


def test_tiles_and_model(tmp_path):
    c = _corpus(tmp_path)
    labels = {u: np.random.rand(12).astype(np.float32) for u in c.ids}
    weights = {u: np.ones(12, np.float32) for u in c.ids}
    left, _ = c.left_prob()
    for region, n_out in (("tf", 2), ("pf", 1), ("rec", 3), ("bm", 2)):
        ds = rs.RegionTiles(c, region, list(c.ids[:2]), labels, weights, left, True)
        assert len(ds) == 2 * len(rs.REGIONS[region]["variants"])
        x, y, w = rs.collate([ds[0], ds[1]])
        for gi, g in enumerate(rs.REGIONS[region]["groups"]):
            assert x[f"x{gi}"].shape == (2, g["n"][0], 3, g["out"][1], g["out"][0])
            assert x[f"x{gi}"].sum() > 0
        assert y.shape == (2, n_out)
    m = rs.RegionNet("resnet18", len(rs.REGIONS["bm"]["groups"]), 1, 2, pretrained=False).eval()
    with torch.no_grad():
        assert m(x).shape == (2, 2)


def test_medial_variant_is_mirrored(tmp_path):
    c = _corpus(tmp_path)
    c.study(0)                      # open memmaps, then plant a marker 30 mm patient-left of the joint
    v = np.array(c.vols[0]); v[:] = 0
    u, w = rs.to_pixel(c.geom[0, 47], c.P[:1] + [[30, 0, 0]])[0].astype(int)
    v[0, 40:55, w - 6:w + 6, u - 6:u + 6] = 255
    c.vols = [v]
    labels = {u_: np.zeros(12, np.float32) for u_ in c.ids}
    left, _ = c.left_prob()
    ds = rs.RegionTiles(c, "tf", ["s0"], labels, labels, left, False)
    lat, med = ds[0][0]["x0"], ds[1][0]["x0"]      # left knee: +x is lateral
    assert lat.sum() > 0 and med.sum() == 0
