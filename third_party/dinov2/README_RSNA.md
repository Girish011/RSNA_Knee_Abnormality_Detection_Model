# Vendored Meta DINOv2 (Apache-2.0)

Source: https://github.com/facebookresearch/dinov2  
Purpose: offline `torch.hub.load(..., source="local")` on Kaggle (no GitHub at train/submit).

`hubconf.py` is slimmed to export only `dinov2_vits14` / `dinov2_vitb14`.
Weights stay in separate public datasets (`dinov2-vits14-rsna-knee`, etc.).
