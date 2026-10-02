"""Family B: fastMRI extraction + heatmap pretraining + Family A encoder init (CPU, synthetic)."""

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "runpod"))
import fastmri_pretrain as fp  # noqa: E402
import family_a as fa  # noqa: E402

h5py = pytest.importorskip("h5py")


def _tar(tmp: Path, n=4, S=8) -> Path:
    tp = tmp / "knee_singlecoil_train.tar"
    rng = np.random.default_rng(0)
    with tarfile.open(tp, "w") as tf:
        for i in range(n):
            b = io.BytesIO()
            with h5py.File(b, "w") as h:
                h["reconstruction_esc"] = rng.random((S, 320, 320)).astype(np.float32)
                h["kspace"] = np.zeros((S, 4, 4), np.complex64)
            data = b.getvalue()
            info = tarfile.TarInfo(f"singlecoil_train/file100000{i}.h5"); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    subprocess.run(["xz", "-z", "-f", str(tp)], check=True)
    return Path(str(tp) + ".xz")


def test_flip_and_geometry():
    r = np.zeros((2, 320, 320), np.float32); r[:, 0, :] = 1.0          # bright top row
    v = fp.to_vol(r)
    assert v.shape == (2, 336, 336) and v.dtype == np.uint8
    assert v[0, -1].mean() > v[0, 0].mean()                            # flipped up/down


def test_extract_pretrain_and_init(tmp_path):
    xz = _tar(tmp_path)
    vol_dir = tmp_path / "vols"
    assert fp.extract(str(xz), str(vol_dir)) == 4
    assert np.load(vol_dir / "file1000000.npy").shape == (8, 336, 336)
    rows = [("file1000000", 3, "No", 100, 120, 40, 20, "Meniscus Tear"),
            ("file1000001", 4, "No", 30, 200, 60, 30, "Joint Effusion"),
            ("file1000002", 4, "No", 200, 150, 20, 20, "Bone- Subchondral edema"),
            ("file1000003", 0, "Yes", None, None, None, None, "artifact")]
    pd.DataFrame(rows, columns=["file", "slice", "study_level", "x", "y", "width", "height", "label"]) \
        .to_csv(tmp_path / "knee.csv", index=False)
    boxes = fp.load_boxes(str(tmp_path / "knee.csv"), {f"file100000{i}" for i in range(4)})
    assert len(boxes) == 3 and ("file1000003", 0) not in boxes
    ds = fp.Tiles(str(vol_dir), [("file1000000", 3, 0)], boxes, "half", False)
    x, t = ds[0]
    assert x.shape == (3, 384, 288) and t.shape == (len(fp.CNAMES), 12, 9)
    assert t[fp.CNAMES.index("meniscus")].sum() > 0                    # box lands in the medial half

    out = tmp_path / "out"
    cmd = [sys.executable, str(REPO / "runpod" / "fastmri_pretrain.py"), "--vol_dir", str(vol_dir),
           "--boxes", str(tmp_path / "knee.csv"), "--arch", "resnet18", "--no_pretrained",
           "--epochs", "1", "--bs", "2", "--workers", "0", "--val_frac", "0.5", "--tag", "t", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ck = torch.load(out / "raptor_fmri_t.pt", weights_only=False)
    m = fa.RegionMIL("resnet18", pretrained=False)
    miss, _ = m.backbone.load_state_dict(ck["backbone"], strict=False)
    assert not miss and json.load(open(out / "raptor_ft_t.json"))["n_volumes"] == 4


def test_links_never_in_bundle_or_logs(monkeypatch, capsys):
    monkeypatch.setenv("FASTMRI_LINKS", "knee_singlecoil_train.tar.xz: https://secret.example/knee_singlecoil_train.tar.xz?Signature=a\n\n"
                       "knee_singlecoil_val.tar.xz https://secret.example/knee_singlecoil_val.tar.xz?Signature=abc\n"
                       'curl -C - "https://secret.example/knee_DICOMs_batch1.tar?Signature=abc" --output x\n')
    assert [fp._name(u) for u in fp._links()] == ["knee_singlecoil_train.tar.xz", "knee_singlecoil_val.tar.xz"]
    with pytest.raises(SystemExit):
        fp.download("https://127.0.0.1:9/knee_singlecoil_val.tar.xz?Signature=abc", "/tmp/_fm_x", tries=1)
    assert "Signature" not in capsys.readouterr().out
