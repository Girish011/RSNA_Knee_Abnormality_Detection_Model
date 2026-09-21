"""DINOv2 backbone loader with offline weight + local hub support."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
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

BACKBONE_DIMS = {**DINOV2_DIMS, "mri_core_vitb": 768}


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


def _find_mri_core_repo() -> Path | None:
    env = os.environ.get("MRI_CORE_REPO")
    candidates = [Path(env)] if env else []
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[3] / "third_party" / "mri_foundation",
            Path.cwd() / "third_party" / "mri_foundation",
            Path("/kaggle/input/datasets/girishbose/rsna-knee-code/third_party/mri_foundation"),
        ]
    )
    for path in candidates:
        if (path / "models" / "sam" / "build_sam.py").exists():
            return path.resolve()
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


def create_mri_core_encoder(
    *,
    weights_path: str | Path,
    image_size: int = 384,
    freeze: bool = True,
):
    """Load the public MRI-CORE ViT-B image encoder from a vendored repository.

    The published ``MRI_CORE_vitb.pth`` is a DINOv2-style teacher dict under
    ``teacher`` / ``backbone.*``. We map those into the SAM ViT-B image encoder
    used by mazurowski-lab/mri_foundation, interpolate absolute pos embeds to
    ``image_size``, and **skip the SAM neck** (not present in the checkpoint).
    Slice vectors are therefore 768-d (ViT-B width), not the 256-d neck dim.
    """
    torch, nn = _torch()
    import torch.nn.functional as F

    repo = _find_mri_core_repo()
    if repo is None:
        raise FileNotFoundError(
            "MRI-CORE source not found. Set MRI_CORE_REPO or vendor "
            "mazurowski-lab/mri_foundation under third_party/mri_foundation."
        )
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"MRI-CORE checkpoint not found: {weights}")

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    sam_module = import_module("models.sam")
    args = SimpleNamespace(
        arch="vit_b",
        if_encoder_adapter=False,
        encoder_adapter_depths=[],
        if_mask_decoder_adapter=False,
        decoder_adapt_depth=0,
        if_encoder_lora_layer=False,
        if_decoder_lora_layer=False,
        encoder_lora_layer=[],
    )
    # Build architecture only; we load teacher weights ourselves (upstream
    # build_sam mis-assigns pos_embed and hard-codes cuda:0).
    sam = sam_module.sam_model_registry["vit_b"](
        args,
        checkpoint=None,
        num_classes=1,
        image_size=image_size,
        pretrained_sam=False,
    )
    image_encoder = sam.image_encoder
    _load_mri_core_teacher_into_sam_encoder(image_encoder, weights, image_size=image_size)

    target_hw = (int(image_size), int(image_size))
    token = int(image_size) // 16
    assert tuple(image_encoder.pos_embed.shape) == (1, token, token, 768), (
        image_encoder.pos_embed.shape,
        token,
    )
    assert float(image_encoder.pos_embed.detach().abs().mean()) > 0.0, "pos_embed failed to load"

    class MRICoreEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 768
            self.image_size = int(image_size)
            self.image_encoder = image_encoder
            self.register_buffer(
                "pixel_mean",
                torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "pixel_std",
                torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
                persistent=False,
            )
            if freeze:
                for parameter in self.image_encoder.parameters():
                    parameter.requires_grad = False

        def forward(self, x: Any) -> Any:
            if tuple(x.shape[-2:]) != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            x = (x * 255.0 - self.pixel_mean) / self.pixel_std
            # Manual ViT forward: patch → +pos → blocks → spatial mean (skip neck).
            tokens = self.image_encoder.patch_embed(x)
            tokens = tokens + self.image_encoder.pos_embed
            for blk in self.image_encoder.blocks:
                tokens = blk(tokens)
            return tokens.mean(dim=(1, 2))

    return MRICoreEncoder()


def _load_mri_core_teacher_into_sam_encoder(image_encoder, weights: Path, *, image_size: int) -> None:
    """Map MRI-CORE teacher ``backbone.*`` keys onto a SAM ImageEncoderViT."""
    torch, _nn = _torch()
    import torch.nn.functional as F

    raw = torch.load(Path(weights), map_location="cpu")
    if isinstance(raw, dict) and "teacher" in raw:
        teacher = raw["teacher"]
    elif isinstance(raw, dict) and "model" in raw:
        teacher = raw["model"]
    else:
        teacher = raw

    vit_patch = 16
    token_size = int(image_size) // vit_patch
    mapped: dict[str, Any] = {}
    for key, value in teacher.items():
        if not str(key).startswith("backbone."):
            continue
        rest = str(key)[len("backbone.") :]
        if rest in {"cls_token", "mask_token"} or rest.startswith("dino_head") or rest.startswith("norm."):
            continue
        if rest == "pos_embed":
            # (1, 1+N, C) with CLS — drop CLS, reshape square, interpolate.
            pos = value[:, 1:, :]
            side = int(pos.shape[1] ** 0.5)
            pos = pos.view(1, side, side, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(token_size, token_size), mode="bilinear", align_corners=False)
            mapped["pos_embed"] = pos.permute(0, 2, 3, 1).contiguous()
            continue
        if rest.startswith("patch_embed."):
            mapped[rest] = value
            continue
        if rest.startswith("blocks."):
            # backbone.blocks.{chunk}.{i}.{...} → blocks.{flat}.{...} with fc→lin
            parts = rest.split(".")
            # parts: blocks, chunk, i, ...
            if len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit():
                flat = int(parts[1]) * 1000 + int(parts[2])  # temp; fix below
                # Official MRI-CORE layout is blocks.{stage}.{block} with stages
                # packing sequential ViT blocks; flatten in encounter order later.
                rest_tail = ".".join(parts[3:]).replace("fc", "lin")
                mapped[f"blocks.{parts[1]}.{parts[2]}.{rest_tail}"] = value
            continue

    # Flatten blocks.stage.i → blocks.k in lexicographic (stage, i) order.
    block_ids = sorted(
        {
            (int(k.split(".")[1]), int(k.split(".")[2]))
            for k in mapped
            if k.startswith("blocks.") and k.split(".")[1].isdigit() and k.split(".")[2].isdigit()
        }
    )
    flat_map = {pair: idx for idx, pair in enumerate(block_ids)}
    flat_mapped: dict[str, Any] = {}
    for key, value in mapped.items():
        if key.startswith("blocks.") and key.split(".")[1].isdigit() and key.split(".")[2].isdigit():
            parts = key.split(".")
            pair = (int(parts[1]), int(parts[2]))
            idx = flat_map[pair]
            flat_mapped["blocks." + str(idx) + "." + ".".join(parts[3:])] = value
        else:
            flat_mapped[key] = value

    missing, unexpected = image_encoder.load_state_dict(flat_mapped, strict=False)
    # Neck / relative-pos leftovers are expected; everything else is a hard fail.
    bad_missing = [
        m
        for m in missing
        if not m.startswith("neck.") and "rel_pos" not in m
    ]
    if bad_missing:
        raise RuntimeError(f"MRI-CORE load missing unexpected keys: {bad_missing[:20]}")
    if "pos_embed" not in flat_mapped:
        raise RuntimeError("MRI-CORE teacher missing backbone.pos_embed")
    print(
        f"MRI-CORE teacher → SAM encoder: loaded {len(flat_mapped)} tensors; "
        f"skipped neck ({sum(1 for m in missing if m.startswith('neck.'))} tensors)",
        flush=True,
    )


def create_image_encoder(
    name: str,
    *,
    weights_path: str | Path | None = None,
    freeze: bool = True,
    pretrained: bool = True,
    image_size: int = 384,
):
    """Create a supported public image encoder by canonical name."""
    if name == "mri_core_vitb":
        if weights_path is None:
            raise ValueError("mri_core_vitb requires an explicit public checkpoint")
        return create_mri_core_encoder(
            weights_path=weights_path,
            image_size=image_size,
            freeze=freeze,
        )
    return create_dinov2_encoder(
        name,
        weights_path=weights_path,
        freeze=freeze,
        pretrained=pretrained,
    )
