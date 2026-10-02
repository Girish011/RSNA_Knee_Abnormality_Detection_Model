"""Family A trainer: end-to-end CPU smoke on a tiny synthetic dense80 corpus."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "runpod"))
import family_a as fa  # noqa: E402


def _corpus(tmp: Path, n_per=6, shards=2, hw=48):
    rng = np.random.default_rng(0)
    ids = []
    for s in range(shards):
        sid = np.array([f"1.2.{s}.{i}" for i in range(n_per)], dtype=object)
        vols = rng.integers(0, 255, (n_per, 80, hw, hw), dtype=np.uint8)
        masks = np.ones((n_per, 80), np.uint8)
        masks[0, 40:65] = 0                       # study without coronal series
        np.save(tmp / f"vols_s{s}.npy", vols); np.save(tmp / f"masks_s{s}.npy", masks)
        np.save(tmp / f"ids_s{s}.npy", sid)
        ids += list(sid)
    return ids


def test_route_mask_and_forward():
    m = fa.RegionMIL("resnet18", pretrained=False, d=32, heads=2, mix_layers=1)
    B, nf, nh = 2, 5, 4
    meta = {"gf": torch.tensor([[0, 0, 1, 6, 6]] * B), "gh": torch.tensor([[2, 3, 4, 5]] * B),
            "pf": torch.zeros(B, nf, dtype=torch.long), "ph": torch.zeros(B, nh, dtype=torch.long),
            "vf": torch.ones(B, nf, dtype=torch.long), "vh": torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])}
    out = m(torch.rand(B, nf, 3, 32, 32), torch.rand(B, nh, 3, 32, 24), meta)
    assert out.shape == (B, 12) and torch.isfinite(out).all()
    j = fa.LAB.index("PF OA")
    assert m.route[j, fa.G["Ax"]] and not m.route[j, fa.G["CorF-a"]]


@pytest.mark.parametrize("mirror", [0.0, 1.0])
def test_dataset_shapes(tmp_path, mirror):
    ids = _corpus(tmp_path)
    c = fa.Corpus(2, str(tmp_path))
    lab = {u: np.full(12, 0.3, np.float32) for u in ids}
    win = {"SagF": 3, "SagNF": 2, "CorF": 2, "CorNF": 1, "Ax": 2}
    ds = fa.StudyTiles(c, ids, lab, lab, win, True, res_full=32, half_hw=(32, 24), mirror=mirror)
    xf, xh, meta, y, w = ds[0]
    assert xf.shape == (7, 3, 32, 32) and xh.shape == (6, 3, 32, 24)
    assert meta["vh"].sum() == 0 and meta["vf"].sum() == 7          # study 0 has no coronal series
    xf, xh, meta, y, w = fa.collate([ds[1], ds[2]])
    assert xf.shape[0] == 2 and meta["gh"].shape == (2, 6)


def test_cli_smoke(tmp_path):
    ids = _corpus(tmp_path)
    tr = pd.DataFrame({"StudyInstanceUID": ids, **{f: np.nan for f in fa.LAB}})
    gold = ids[-4:]
    for k, u in enumerate(gold):                                       # 2 dev + 2 holdout experts
        tr.loc[tr.StudyInstanceUID == u, fa.LAB] = float(k % 2)
    tr.to_csv(tmp_path / "train.csv", index=False)
    rng = np.random.default_rng(1)
    lab = pd.DataFrame({"StudyInstanceUID": ids[:-4], **{f: rng.random(len(ids) - 4) for f in fa.LAB}})
    lab.to_parquet(tmp_path / "labels.parquet")
    json.dump({"folds": {u: i % 2 for i, u in enumerate(ids[:-4])}, "n_folds": 2},
              open(tmp_path / "folds.json", "w"))
    json.dump(gold[2:], open(tmp_path / "hold.json", "w"))
    out = tmp_path / "out"
    cmd = [sys.executable, str(REPO / "runpod" / "family_a.py"), "--labels", str(tmp_path / "labels.parquet"),
           "--train_csv", str(tmp_path / "train.csv"), "--fold_file", str(tmp_path / "folds.json"),
           "--holdout", str(tmp_path / "hold.json"), "--data_root", str(tmp_path), "--n_shards", "2",
           "--arch", "resnet18", "--no_pretrained", "--res_full", "32", "--half_h", "32", "--half_w", "24",
           "--win", "SagF:2,SagNF:1,CorF:1,CorNF:1,Ax:1", "--win_eval", "SagF:2,SagNF:1,CorF:1,CorNF:1,Ax:1",
           "--epochs", "1", "--bs", "2", "--accum", "1", "--workers", "0", "--eval_every", "1",
           "--eval_passes", "1", "--tag", "t", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.load(open(out / "raptor_ft_t.json"))
    assert res["n_holdout"] == 2 and res["n_oof"] == 4 and res["n_train"] == 6
    z = np.load(out / "raptor_oof_t_fold0.npz")
    assert z["pred"].shape == (4, 12) and np.isfinite(z["pred"]).all()
    assert (out / "raptor_fa_t.pt").exists() and (out / "raptor_holdout_t.npz").exists()
