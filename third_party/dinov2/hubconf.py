# Copyright (c) Meta Platforms, Inc. and affiliates.
# Apache-2.0 — slim hubconf for offline Kaggle (no cell/xray/depther imports).

dependencies = ["torch"]

from dinov2.hub.backbones import dinov2_vitb14, dinov2_vits14

__all__ = ["dinov2_vits14", "dinov2_vitb14"]
