"""ours_leg_v2 (parallel study loading, every GPU) must reproduce ours_leg (v1) on the same studies.

Synthetic DICOM studies + two random-init checkpoints at different resolutions, CPU only.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("cv2")

LEG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "kernels" / "_ours_leg"
PLANES = {"Sagittal": ([0, 1, 0, 0, 0, -1], 0), "Coronal": ([1, 0, 0, 0, 0, -1], 1), "Axial": ([1, 0, 0, 0, 1, 0], 2)}


def _write_series(d: Path, plane: str, n: int, rng: np.random.Generator) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    d.mkdir(parents=True)
    iop, axis = PLANES[plane]
    for k in rng.permutation(n):          # file order is not slice order
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(d / f"{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
        ds.Rows = ds.Columns = 96
        ds.PixelSpacing = [1.5, 1.5]
        ds.ImageOrientationPatient = iop
        pos = [0.0, 0.0, 0.0]
        pos[axis] = 3.0 * k
        ds.ImagePositionPatient = pos
        ds.InstanceNumber = int(k) + 1
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 0
        img = rng.integers(0, 900, (96, 96)).astype(np.uint16)
        img[20 + k:50 + k, 30:60] += 800   # slice-dependent structure, not left-right symmetric
        ds.PixelData = img.tobytes()
        ds.save_as(str(d / f"{k}.dcm"), write_like_original=False)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEG_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("legroot")
    rng = np.random.default_rng(0)
    rows, ids = [], []
    layouts = [  # (plane, fluid) series per study; the third study has no axial, the fourth one series only
        [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 1)],
        [("Sagittal", 1), ("Coronal", 1), ("Coronal", 0), ("Axial", 0)],
        [("Sagittal", 0), ("Sagittal", 1), ("Coronal", 0)],
        [("Coronal", 1)],
        [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 1)],
        [("Axial", 1), ("Sagittal", 1), ("Coronal", 1)],
    ]
    for s, lay in enumerate(layouts):
        sid = f"1.2.3.{s}"
        ids.append(sid)
        for j, (plane, fluid) in enumerate(lay):
            uid = f"{sid}.{j}"
            _write_series(root / "test_series" / sid / uid, plane, int(rng.integers(9, 22)), rng)
            rows.append(dict(StudyInstanceUID=sid, SeriesInstanceUID=uid, Fluid_Sensitive=fluid,
                             Fat_Suppression=fluid, Anatomical_Plane=plane))
    ids.append("1.2.3.99")                 # a study with no series rows at all
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "test_series.csv", index=False)

    v1 = _load("ours_leg")
    wdir = root / "weights"
    wdir.mkdir()
    for fname, arch, res in (("a.pt", "resnet10t", 64), ("b.pt", "resnet10t", 96), ("c.pt", "resnet18", 64)):
        torch.manual_seed(len(fname) + res)
        bb = v1.build_backbone(arch)
        m = v1.RaptorClassifier(bb, F_dim=bb.num_features)
        torch.nn.init.normal_(m.clsW, std=0.5)
        torch.save({"arch": arch, "res": res, "model": m.state_dict()}, wdir / fname)
    return root, wdir, ids


def _run(mod, world, monkeypatch, out: Path, **env) -> pd.DataFrame:
    root, wdir, _ = world
    monkeypatch.setattr(mod, "find_test_root", lambda: str(root))
    monkeypatch.setattr(mod, "find_weight_file", lambda f: str(wdir / f))
    monkeypatch.setenv("OURS_MEMBERS", json.dumps(["a.pt", "b.pt", "c.pt"]))
    monkeypatch.setenv("OURS_OUT", str(out))
    for k in ("OURS_FLIP", "OURS_WORKERS", "OURS_LIMIT", "OURS_ROOT", "OURS_DEADLINE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    mod.main()
    return pd.read_csv(out)


@pytest.mark.parametrize("workers", [0, 2])
def test_v2_matches_v1(world, monkeypatch, tmp_path, workers):
    if workers and "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    if workers:
        monkeypatch.setattr(torch.multiprocessing, "get_start_method", lambda *a, **k: "fork", raising=False)
        mp.set_start_method("fork", force=True)
    v1, v2 = _load("ours_leg"), _load("ours_leg_v2")
    a = _run(v1, world, monkeypatch, tmp_path / "v1.csv")
    b = _run(v2, world, monkeypatch, tmp_path / "v2.csv", OURS_WORKERS=workers)
    assert a["StudyInstanceUID"].tolist() == b["StudyInstanceUID"].tolist() == world[2]
    np.testing.assert_allclose(a[v1.LAB].to_numpy(), b[v2.LAB].to_numpy(), atol=1e-6)
    for j in range(3):                     # per-member rank files too
        pa, pb = pd.read_csv(tmp_path / f"v1_m{j}.csv"), pd.read_csv(tmp_path / f"v2_m{j}.csv")
        np.testing.assert_allclose(pa[v1.LAB].to_numpy(), pb[v2.LAB].to_numpy(), atol=1e-6)


def test_v2_flip_changes_only_the_second_pass(world, monkeypatch, tmp_path):
    v2 = _load("ours_leg_v2")
    vol = np.random.default_rng(1).integers(0, 255, (v2.MAXS, 48, 48)).astype(np.uint8)
    mask = np.ones(v2.MAXS, np.uint8)
    x0 = v2.eval_windows(vol, mask, k=24, res=48, phase=0.5, norm="none")
    x1 = v2.eval_windows(vol, mask, k=24, res=48, phase=0.5, norm="none", flip=True)
    cs = v2._eval_centers(mask, v2.MAXS, 24, phase=0.5)
    for j, c in enumerate(cs):
        want = x0[j].flip(0) if c < v2.SAG_SLOTS else x0[j].flip(-1)
        assert torch.equal(x1[j], want)
    assert any(c < v2.SAG_SLOTS for c in cs) and any(c >= v2.SAG_SLOTS for c in cs)
    a = _run(v2, world, monkeypatch, tmp_path / "noflip.csv", OURS_WORKERS=0)
    b = _run(v2, world, monkeypatch, tmp_path / "flip.csv", OURS_WORKERS=0, OURS_FLIP=1)
    assert not np.allclose(a[v2.LAB].to_numpy(), b[v2.LAB].to_numpy())


def test_v2_deadline_aborts_nonzero(world, monkeypatch, tmp_path):
    v2 = _load("ours_leg_v2")
    codes = []
    monkeypatch.setattr(v2.os, "_exit", lambda c: (_ for _ in ()).throw(SystemExit(c)))
    with pytest.raises(SystemExit) as e:
        _run(v2, world, monkeypatch, tmp_path / "late.csv", OURS_WORKERS=0, OURS_DEADLINE=1.0)
    codes.append(e.value.code)
    assert codes == [3] and not (tmp_path / "late.csv").exists()


@pytest.mark.parametrize("head", ["seq", "tok", "seqtok"])
def test_v2_slot_aware_member_matches_trainer(head, tmp_path):
    """A slot-aware checkpoint (train_knee --head / --sample strat) scores the same in the leg as in the trainer's eval."""
    import timm
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
    import train_knee as tk
    v2 = _load("ours_leg_v2")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vols = (rng.random((2, 80, 48, 48)) * 255).astype(np.uint8)
    masks = np.ones((2, 80), np.uint8)
    masks[1, 22:40] = 0
    vols[1, 22:40] = 0
    bb = timm.create_model("convnext_atto", pretrained=False, num_classes=0, global_pool="avg")
    m = tk.SeqClassifier(bb, F_dim=bb.num_features, use_tr=head in ("seq", "seqtok"), use_sp=head in ("tok", "seqtok"))
    with torch.no_grad():   # move every zero-gated part away from zero so that it shows in the output
        for n_, p in m.named_parameters():
            if not n_.startswith("backbone.") and float(p.abs().sum()) == 0:
                p.normal_(0, 0.5)
    m.eval()
    pt = tmp_path / f"raptor_ft_{head}.pt"
    torch.save({"model": m.state_dict(), "arch": "convnext_atto", "res": 48, "head": head, "sample": "strat", "k_eval": 10}, pt)
    lm, cfg = v2.load_model(str(pt), "cpu")
    assert cfg == (48, "strat", 10, False) and type(lm).__name__ == "SeqClassifier"
    ids = ["a", "b"]
    for i, u in enumerate(ids):
        want = []
        for ph in (0.0, 0.5):
            ds = tk.StudyWindows(".", ids, {"a": 0, "b": 1}, {x: np.zeros(12, np.float32) for x in ids}, 48, 10, train=False,
                                 norm="imagenet", eval_phase=ph, sample="strat")
            ds.vols, ds.masks = vols, masks
            x, _y, _w, pos = ds[i]
            with torch.no_grad():
                want.append(torch.sigmoid(m(x[None], pos[None]))[0].numpy())
        wp = [v2.eval_windows_pos(vols[i], masks[i], k=10, res=48, phase=ph, norm="imagenet", sample="strat") for ph in (0.0, 0.5)]
        got = v2.infer_probs(lm, torch.stack([w[0] for w in wp]), "cpu", torch.stack([w[1] for w in wp]))
        np.testing.assert_allclose(got, np.mean(want, 0), atol=1e-5)


def test_v2_canonical_side(tmp_path):
    """Knee side from the headers (stack centre x, tag near the midline) and a canon member = trainer with --canon."""
    import timm
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
    import train_knee as tk
    v2 = _load("ours_leg_v2")
    assert v2.knee_is_left([80.0, 75.0, 90.0], ["", "", ""]) and not v2.knee_is_left([-80.0, -70.0], ["L"])
    assert v2.knee_is_left([2.0], ["L"]) and not v2.knee_is_left([2.0], ["R"]) and v2.knee_is_left([2.0], [""])
    # header geometry: a coronal stack of 96 px x 1.5 mm starting at x = 0 is centred at x = +72 mm (patient left)
    rng = np.random.default_rng(0)
    _write_series(tmp_path / "s" / "cor", "Coronal", 10, rng)
    geo = {}
    v2._make_reader()[0](str(tmp_path / "s" / "cor"), geo)
    assert abs(geo["cx"] - 72.0) < 1e-6 and geo["tag"] == ""
    # member with canon: the leg flips a left knee, the trainer mirrors it through --canon
    torch.manual_seed(0)
    vols = (rng.random((1, 80, 48, 48)) * 255).astype(np.uint8)
    masks = np.ones((1, 80), np.uint8)
    bb = timm.create_model("convnext_atto", pretrained=False, num_classes=0, global_pool="avg")
    m = tk.SeqClassifier(bb, F_dim=bb.num_features, use_tr=True, use_sp=True)
    with torch.no_grad():
        for n_, p in m.named_parameters():
            if not n_.startswith("backbone.") and float(p.abs().sum()) == 0:
                p.normal_(0, 0.5)
    m.eval()
    pt = tmp_path / "raptor_ft_c.pt"
    torch.save({"model": m.state_dict(), "arch": "convnext_atto", "res": 48, "head": "seqtok", "sample": "strat", "k_eval": 10,
                "canon": True}, pt)
    lm, cfg = v2.load_model(str(pt), "cpu")
    assert cfg == (48, "strat", 10, True)
    for left in (True, False):
        want = []
        for ph in (0.0, 0.5):
            ds = tk.StudyWindows(".", ["a"], {"a": 0}, {"a": np.zeros(12, np.float32)}, 48, 10, train=False, norm="imagenet",
                                 eval_phase=ph, sample="strat", sides={"a": left})
            ds.vols, ds.masks = vols, masks
            x, _y, _w, pos = ds[0]
            with torch.no_grad():
                want.append(torch.sigmoid(m(x[None], pos[None]))[0].numpy())
        wp = [v2.eval_windows_pos(vols[0], masks[0], k=10, res=48, phase=ph, norm="imagenet", flip=left, sample="strat")
              for ph in (0.0, 0.5)]
        got = v2.infer_probs(lm, torch.stack([w[0] for w in wp]), "cpu", torch.stack([w[1] for w in wp]))
        np.testing.assert_allclose(got, np.mean(want, 0), atol=1e-5)
