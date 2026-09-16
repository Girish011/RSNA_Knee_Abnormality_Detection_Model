# External assets

All submission assets must be public, downloaded before the offline submission run, and
packaged with their original license/notice.

## MRI-CORE ViT-B

- Source: https://github.com/mazurowski-lab/mri_foundation
- Checkpoint: official `MRI_CORE_vitb.pth` linked by the source repository
- License: Apache-2.0
- Use: frozen MRI slice encoder for the max-AUC track
- Offline packaging: `scripts/fetch_mri_core.sh`, then `scripts/package_kaggle_datasets.sh`
- Modification: input positional embeddings are interpolated by the official loader for 384px

## Meta DINOv2

- Source: https://github.com/facebookresearch/dinov2
- Checkpoints: public `dinov2_vits14` and `dinov2_vitb14`
- License: Apache-2.0 for the bundled source/checkpoints, subject to upstream notices
- Use: established image baseline and efficiency track

## Qwen2.5-7B-Instruct

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- License: Apache-2.0
- Use: train-only report labeling (consensus v5/v6 fills unlabeled weak_v1 cells; v6 uses constrained JSON)
- Submission dependency: none; reports and the LLM are excluded from inference

## Multilingual E5

- Model: `intfloat/multilingual-e5-base`
- Source: https://huggingface.co/intfloat/multilingual-e5-base
- License: MIT
- Use: optional report embedding during training only
- Submission dependency: none; reports and the text encoder are excluded from inference
