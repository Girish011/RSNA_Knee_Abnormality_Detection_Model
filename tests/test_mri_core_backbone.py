"""Smoke tests for public image encoder dispatch (no GPU required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna_knee.models.backbone import BACKBONE_DIMS, create_image_encoder


ROOT = Path(__file__).resolve().parents[1]
MRI_WEIGHTS = ROOT / "data" / "external" / "mri_core" / "MRI_CORE_vitb.pth"
MRI_REPO = ROOT / "third_party" / "mri_foundation"


def test_backbone_dims_include_mri_core() -> None:
    assert BACKBONE_DIMS["mri_core_vitb"] == 768
    assert BACKBONE_DIMS["dinov2_vits14"] == 384


def test_mri_core_requires_weights() -> None:
    with pytest.raises(ValueError, match="explicit public checkpoint"):
        create_image_encoder("mri_core_vitb", weights_path=None)


@pytest.mark.skipif(not MRI_WEIGHTS.exists(), reason="MRI-CORE weights not downloaded")
@pytest.mark.skipif(not (MRI_REPO / "models" / "sam" / "build_sam.py").exists(), reason="MRI-CORE source missing")
def test_mri_core_loads_and_encodes_cpu() -> None:
    import torch

    enc = create_image_encoder(
        "mri_core_vitb",
        weights_path=MRI_WEIGHTS,
        freeze=True,
        image_size=384,
    )
    enc.eval()
    x = torch.rand(2, 3, 224, 224)  # cache-sized; encoder upsamples
    with torch.no_grad():
        y = enc(x)
    assert y.shape == (2, 768)
    assert float(enc.image_encoder.pos_embed.abs().mean()) > 0.0
