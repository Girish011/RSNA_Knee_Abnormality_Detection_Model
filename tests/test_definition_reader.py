"""Meaningful CPU checks for the independently supervised reader and deployment."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("definition_reader", REPO / "runpod/definition_reader.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)


def test_sampler_excludes_missing_and_cross_sequence_and_eval_disjoint():
    mask = np.ones(80, bool)
    mask[10] = False
    c = dr.valid_centres(mask, 0, 22)
    assert not {0, 9, 10, 11, 21, 22}.intersection(c)
    assert all(mask[k - 1:k + 2].all() for k in c)
    a = dr.choose_centres(c, 5, phase=0, phases=2)
    b = dr.choose_centres(c, 5, phase=1, phases=2)
    assert len(a) == len(b) == 5 and not set(a).intersection(b)
    assert np.array_equal(a, dr.choose_centres(c, 5, phase=0, phases=2))
    assert len(dr.choose_centres(c[:2], 5, phase=0, phases=1)) == 2


def test_actual_sequence_metadata_and_bounded_physical_crop(tmp_path):
    root = dr.synthetic(tmp_path)
    info = json.loads((root / "series_s0.json").read_text())
    # Cache claims fluid=1, actual CSV says 0. Actual metadata must win.
    info["synthetic-0"][0]["fluid"] = 1
    (root / "series_s0.json").write_text(json.dumps(info))
    corpus = dr.Corpus(root, 1, root / "train_series.csv")
    assert corpus.sequence["synthetic-0"][0][0] == 0
    unknown = dr.Corpus(root, 1)
    assert unknown.sequence["synthetic-0"][0] == [-1., -1.]
    g = corpus.geom[0, 45]
    image = np.full((32, 32), 200, np.uint8)
    x, retained = dr.bounded_crop(image, g, corpus.P[0] + [100, 0, 0],
                                  np.array([1, 0, 0]), np.array([0, 0, -1]), 190, 64)
    assert 0 < retained < 1 and x.min() == 200
    ds = dr.Windows(corpus, ["synthetic-0"], size=32, windows=2)
    x, meta, valid, _ = ds[0]
    assert x.shape[0] == valid.sum() and np.isfinite(meta.numpy()).all()
    assert not (x.sum((1, 2, 3)) == 0).any()
    assert set(meta[:, 1].tolist()) == {0., 1., 2., 3., 4.}


def test_loss_masks_unknowns_and_aux_does_not_change_primary_schema():
    logit = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
    target = torch.tensor([[0., 1.], [1., 0.]])
    mask = torch.tensor([[1., 0.], [0., 1.]])
    dr.masked_loss(logit, target, mask).backward()
    assert logit.grad[0, 1] == logit.grad[1, 0] == 0
    assert logit.grad[0, 0] > 0 and logit.grad[1, 1] > 0
    m = dr.Reader(hidden=16, dummy=True)
    x = torch.ones((2, 3, 3, 32, 32), dtype=torch.uint8)
    z, grades = m(x, torch.zeros((2, 3, 9)), torch.ones((2, 3), dtype=torch.bool))
    assert z.shape == (2, 12) and grades.shape == (2, 12, 3)


def test_single_weak_label_confidence_scales_gradient_absolutely():
    target = torch.tensor([[1., 0.], [0., 0.]])
    mask = torch.tensor([[1., 0.], [0., 0.]])
    strong = torch.zeros((2, 2), requires_grad=True)
    weak = torch.zeros((2, 2), requires_grad=True)
    strong_loss = dr.masked_loss(strong, target, mask, torch.ones_like(mask))
    weak_loss = dr.masked_loss(weak, target, mask, torch.full_like(mask, .2))
    strong_loss.backward(); weak_loss.backward()
    assert torch.allclose(weak_loss, strong_loss * .2)
    assert torch.allclose(weak.grad, strong.grad * .2)
    assert weak.grad[0, 0] != 0
    assert torch.count_nonzero(weak.grad) == 1


def test_gold_in_label_bundle_is_rejected(tmp_path):
    root = dr.synthetic(tmp_path)
    tr = pd.read_csv(root / "train.csv")
    tr.loc[0, dr.LAB] = 0
    tr.to_csv(root / "train.csv", index=False)
    with pytest.raises(ValueError, match="gold"):
        dr.load_labels(root / "labels.npz", root / "train.csv", "definition")


def test_smoke_deployed_ema_matches_oof_and_resume(tmp_path):
    out = tmp_path / "run"
    args = ["--smoke", "--dummy-encoder", "--out", str(out), "--tag", "t", "--device", "cpu"]
    assert dr.main(args) == 0
    ck = torch.load(out / "definition_t_ema.pt", map_location="cpu", weights_only=False)
    resume = torch.load(out / "definition_t_resume.pt", map_location="cpu", weights_only=False)
    assert resume["epoch"] == 1 and resume["batch_cursor"] == 0
    assert resume["step"] == 2  # 5 studies, 3 batches, accumulation2: final short group is retained
    for key in ("optimizer", "scheduler", "scaler", "rng", "dataset_epoch"):
        assert key in resume
    root = out / "synthetic"
    dest = out / "pred.npz"
    assert dr.main(["--data-root", str(root), "--n-shards", "1", "--series-csv", str(root / "train_series.csv"),
                    "--predict", str(out / "definition_t_ema.pt"), "--prediction-out", str(dest),
                    "--out", str(out), "--workers", "0", "--device", "cpu"]) == 0
    p, v = np.load(dest), np.load(out / "definition_t_oof.npz")
    index = {u: i for i, u in enumerate(p["ids"])}
    # Encoder GEMMs can differ by one float32 ULP when other batch studies differ.
    assert np.allclose(p["pred"][[index[u] for u in v["ids"]]], v["pred"], atol=1e-7, rtol=0)
    assert json.loads(str(v["metadata"])) == resume["contract"]
    subset = out / "selected.npy"
    np.save(subset, p["ids"][[3, 0]])
    predict_args = ["--data-root", str(root), "--n-shards", "1", "--series-csv", str(root / "train_series.csv"),
                    "--predict", str(out / "definition_t_ema.pt"), "--prediction-out", str(dest),
                    "--out", str(out), "--workers", "0", "--device", "cpu", "--predict-ids", str(subset)]
    assert dr.main(predict_args) == 0
    selected = np.load(dest)
    assert selected["ids"].tolist() == p["ids"][[3, 0]].tolist()
    assert np.allclose(selected["pred"], p["pred"][[3, 0]], atol=1e-7, rtol=0)
    np.save(subset, np.array(["synthetic-0", "synthetic-0"]))
    with pytest.raises(ValueError, match="unique"):
        dr.main(predict_args)
    np.save(subset, np.array(["missing-study"]))
    with pytest.raises(ValueError, match="absent"):
        dr.main(predict_args)
    assert dr.main(args + ["--resume", str(out / "definition_t_resume.pt")]) == 0
    p2 = torch.load(out / "definition_t_ema.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(ck["model"][k], p2["model"][k]) for k in ck["model"])


def test_time_stop_checkpoint_then_resume_exact(tmp_path):
    out = tmp_path / "stopped"
    args = ["--smoke", "--dummy-encoder", "--out", str(out), "--tag", "t", "--device", "cpu"]
    assert dr.main(args + ["--max-seconds", "0"]) == 0
    resume = out / "definition_t_resume.pt"
    ck = torch.load(resume, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 0 and ck["batch_cursor"] == 0
    assert dr.main(args + ["--resume", str(resume)]) == 0
    full = tmp_path / "full"
    assert dr.main(["--smoke", "--dummy-encoder", "--out", str(full), "--tag", "t", "--device", "cpu"]) == 0
    a = torch.load(out / "definition_t_ema.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(full / "definition_t_ema.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
