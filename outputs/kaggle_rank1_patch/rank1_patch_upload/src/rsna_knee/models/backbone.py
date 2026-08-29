"""DINOv2 backbone loader with offline weight + local hub support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


DINOV2_NAMES = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
    "vits14": "dinov2_vits14",
    "vitb14": "dinov2_vitb14",
    "vitl14": "dinov2_vitl14",
}

DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


def _find_dinov2_repo() -> Path | None:
    """Locate vendored facebookresearch/dinov2 (hubconf.py) for offline torch.hub."""
    env = os.environ.get("DINOV2_REPO")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[3] / "third_party" / "dinov2",  # <repo>/src/rsna_knee/models
            Path.cwd() / "third_party" / "dinov2",
            Path("/kaggle/input/datasets/girishbose/rsna-knee-code/third_party/dinov2"),
        ]
    )

    for p in candidates:
        if (p / "hubconf.py").exists():
            return p.resolve()
    return None


def create_dinov2_encoder(
    name: str = "dinov2_vits14",
    *,
    weights_path: str | Path | None = None,
    freeze: bool = False,
    pretrained: bool = True,
):
    """Return a DINOv2 encoder module (or a small fallback if hub unavailable)."""
    torch, nn = _torch()
    hub_name = DINOV2_NAMES.get(name, name)
    embed_dim = DINOV2_DIMS.get(hub_name, 384)

    class IdentityEncoder(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.embed_dim = dim
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(3, dim),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class DinoV2Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.backbone = self._load_backbone()
            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        def _hub_load(self, *, pretrained_flag: bool) -> nn.Module:
            local = _find_dinov2_repo()
            if local is not None:
                return torch.hub.load(
                    str(local),
                    hub_name,
                    source="local",
                    pretrained=pretrained_flag,
                )
            return torch.hub.load(
                "facebookresearch/dinov2",
                hub_name,
                pretrained=pretrained_flag,
                trust_repo=True,
            )

        def _load_backbone(self) -> nn.Module:
            if weights_path is not None:
                model = self._hub_load(pretrained_flag=False)
                state = torch.load(Path(weights_path), map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                model.load_state_dict(state, strict=False)
                return model
            if pretrained:
                try:
                    return self._hub_load(pretrained_flag=True)
                except Exception:
                    return IdentityEncoder(embed_dim)
            return IdentityEncoder(embed_dim)

        def forward(self, x: Any) -> Any:
            if hasattr(self.backbone, "forward_features"):
                feats = self.backbone.forward_features(x)
                if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                    return feats["x_norm_clstoken"]
            out = self.backbone(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            return out

    return DinoV2Encoder()
