"""Thin baseline factory: DINOv2-small multiseries study model."""

from __future__ import annotations

from rsna_knee.models.multiseries import create_multiseries_model


def create_baseline_model(
    weights_path: str | None = None,
    *,
    freeze_backbone: bool = True,
    pretrained: bool = True,
):
    """DINOv2 ViT-S/14 baseline used for first LB submission."""
    return create_multiseries_model(
        "dinov2_vits14",
        weights_path=weights_path,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
        dropout=0.1,
    )
