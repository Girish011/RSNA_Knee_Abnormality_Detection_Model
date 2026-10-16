"""Region specialists: geometry, crops and model shapes on a synthetic dense512 corpus."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
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


def test_seqfuse_channels_are_fluid_nonfluid_difference(tmp_path):
    c = _corpus(tmp_path)
    c.study(0)
    v = np.array(c.vols[0]); v[:] = 0
    v[0, 0:22] = 200        # SagF bright, SagNF dark -> difference channel above 0.5
    v[0, 22:40] = 50
    c.vols = [v]
    labels = {u_: np.zeros(12, np.float32) for u_ in c.ids}
    left, _ = c.left_prob()
    adj = rs.RegionTiles(c, "rec", ["s0"], labels, labels, left, False)[0][0]
    fus = rs.RegionTiles(c, "rec", ["s0"], labels, labels, left, False, chan="seqfuse")[0][0]
    g = [i for i, d in enumerate(rs.REGIONS["rec"]["groups"]) if d["block"] == "SagF"][0]
    x = fus[f"x{g}"]
    m = x[:, 0] > 0.5                               # inside the imaged field (the box is wider than the 140 mm crop)
    assert m.float().mean() > 0.5
    assert torch.allclose(x[:, 0][m], torch.tensor(200 / 255), atol=1e-3)
    assert torch.allclose(x[:, 1][m], torch.tensor(50 / 255), atol=1e-3)          # same footprint: registered in patient mm
    assert torch.allclose(x[:, 2], (0.5 + (x[:, 0] - x[:, 1]) / 2).clamp(0, 1), atol=1e-6)
    a0 = adj[f"x{g}"]
    assert torch.allclose(a0[:, 1][m], torch.tensor(200 / 255), atol=1e-3)        # default: adjacent slices of one series
    for gi, d in enumerate(rs.REGIONS["rec"]["groups"]):                                           # other groups unchanged
        if d["block"] not in rs.PARTNER:
            assert torch.equal(adj[f"x{gi}"], fus[f"x{gi}"])


def test_aux_grades_targets_are_masked_ordinal(tmp_path):
    import pandas as pd
    c = _corpus(tmp_path)
    g = pd.DataFrame({"StudyInstanceUID": list(c.ids)})
    for f in rs.LAB:
        g[f] = 0
    g["Lateral OA"] = [4, 1, 2, 0]          # severe, not mentioned, low-grade, explicitly normal
    g["Medial OA"] = [3, 3, 1, 1]
    g.to_parquet(tmp_path / "grades.parquet")
    aux = rs.load_aux(str(tmp_path / "grades.parquet"), 0.6)
    labels = {u: np.random.rand(12).astype(np.float32) for u in c.ids}
    weights = {u: np.ones(12, np.float32) for u in c.ids}
    left, _ = c.left_prob()
    ds = rs.RegionTiles(c, "tf", list(c.ids), labels, weights, left, False, aux=aux)
    got = {ds.items[i]: ds[i][1:] for i in range(len(ds))}          # (uid, variant) -> (y, w); variant 0 = lateral
    y, w = got[("s0", 0)]
    assert y.shape == (5,) and y[2:].tolist() == [1, 1, 1] and np.allclose(w[2:], 0.2)
    assert got[("s1", 0)][1][2:].tolist() == [0, 0, 0]              # not mentioned -> unknown, no loss
    assert got[("s2", 0)][0][2:].tolist() == [1, 0, 0] and got[("s3", 0)][0][2:].tolist() == [0, 0, 0]
    assert np.allclose(got[("s3", 0)][1][2:], 0.2)                  # explicitly normal -> known negative
    assert got[("s0", 1)][0][2:].tolist() == [1, 1, 0] and got[("s2", 1)][1][2:].tolist() == [0, 0, 0]   # medial variant
    assert np.allclose(got[("s0", 0)][0][:2], labels["s0"][[rs.L["Lateral Meniscus"], rs.L["Lateral OA"]]])


def test_aux_grades_train_and_predict_end_to_end(tmp_path, monkeypatch):
    """A --aux_grades model trains, saves n_aux, and --predict returns only the region's findings."""
    import pandas as pd
    import sys
    c = _corpus(tmp_path, n=8)
    ids = list(c.ids)
    tr = pd.DataFrame({"StudyInstanceUID": ids})
    for f in rs.LAB:
        tr[f] = [1.0, 0.0] + [np.nan] * 6                           # s0, s1 = expert studies
    tr.to_csv(tmp_path / "train.csv", index=False)
    lab = pd.DataFrame({"StudyInstanceUID": ids, **{f: np.random.rand(8) for f in rs.LAB}})
    lab.to_parquet(tmp_path / "labels.parquet")
    g = pd.DataFrame({"StudyInstanceUID": ids, **{f: np.random.randint(0, 5, 8) for f in rs.LAB}})
    g.to_parquet(tmp_path / "grades.parquet")
    json.dump({"folds": {u: i % 2 for i, u in enumerate(ids)}}, open(tmp_path / "folds.json", "w"))
    json.dump(["s1"], open(tmp_path / "hold.json", "w"))
    out = tmp_path / "out"
    base = ["region_spec.py", "--data_root", str(tmp_path), "--n_shards", "1", "--out", str(out), "--workers", "0", "--bs", "2"]
    monkeypatch.setattr(sys, "argv", base + [
        "--labels", str(tmp_path / "labels.parquet"), "--region", "pf", "--train_csv", str(tmp_path / "train.csv"),
        "--fold_file", str(tmp_path / "folds.json"), "--holdout", str(tmp_path / "hold.json"), "--arch", "resnet18",
        "--no_pretrained", "--smoke", "--tag", "t", "--aux_grades", str(tmp_path / "grades.parquet")])
    assert rs.main() == 0
    ck = torch.load(out / "raptor_rs_t.pt", map_location="cpu", weights_only=False)
    assert ck["n_aux"] == 3 and ck["model"]["cls_b"].shape == (4,)
    monkeypatch.setattr(sys, "argv", base + ["--labels", "x", "--region", "pf", "--tag", "p", "--eval_passes", "1",
                                             "--predict", f"{out / 'raptor_rs_t_last.pt'},{out / 'raptor_rs_t.pt'}"])
    assert rs.main() == 0
    z = np.load(out / "raptor_rspred_p.npz")
    k = rs.L["PF OA"]
    assert z["pred"].shape == (8, 12) and np.isfinite(z["pred"][:, k]).all()
    assert np.isnan(np.delete(z["pred"], k, axis=1)).all()


def test_predict_only_findings_scores_one_variant(tmp_path, monkeypatch):
    """--only_findings 'Lateral OA' scores the lateral compartment alone: same numbers as the full call, medial NaN."""
    import pandas as pd
    import sys
    c = _corpus(tmp_path, n=8)
    ids = list(c.ids)
    tr = pd.DataFrame({"StudyInstanceUID": ids})
    for f in rs.LAB:
        tr[f] = [1.0, 0.0] + [np.nan] * 6
    tr.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids, **{f: np.random.rand(8) for f in rs.LAB}}).to_parquet(tmp_path / "labels.parquet")
    json.dump({"folds": {u: i % 2 for i, u in enumerate(ids)}}, open(tmp_path / "folds.json", "w"))
    json.dump(["s1"], open(tmp_path / "hold.json", "w"))
    out = tmp_path / "out"
    base = ["region_spec.py", "--data_root", str(tmp_path), "--n_shards", "1", "--out", str(out), "--workers", "0", "--bs", "2"]
    monkeypatch.setattr(sys, "argv", base + [
        "--labels", str(tmp_path / "labels.parquet"), "--region", "tf", "--train_csv", str(tmp_path / "train.csv"),
        "--fold_file", str(tmp_path / "folds.json"), "--holdout", str(tmp_path / "hold.json"), "--arch", "resnet18",
        "--no_pretrained", "--smoke", "--tag", "t"])
    assert rs.main() == 0
    pred = base + ["--labels", "x", "--region", "tf", "--eval_passes", "1", "--predict", str(out / "raptor_rs_t.pt")]
    monkeypatch.setattr(sys, "argv", pred + ["--tag", "all"])
    assert rs.main() == 0
    monkeypatch.setattr(sys, "argv", pred + ["--tag", "lat", "--only_findings", "Lateral OA", "--channels_last"])
    assert rs.main() == 0
    full, lat = np.load(out / "raptor_rspred_all.npz")["pred"], np.load(out / "raptor_rspred_lat.npz")["pred"]
    kl = [rs.L["Lateral Meniscus"], rs.L["Lateral OA"]]; km = [rs.L["Medial Meniscus"], rs.L["Medial OA"]]
    assert np.isfinite(full[:, kl + km]).all()
    assert np.allclose(lat[:, kl], full[:, kl], atol=1e-5) and np.isnan(lat[:, km]).all()
    monkeypatch.setattr(sys, "argv", pred + ["--tag", "bad", "--only_findings", "PF OA"])
    with pytest.raises(SystemExit):
        rs.main()
