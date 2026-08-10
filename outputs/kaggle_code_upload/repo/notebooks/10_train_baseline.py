# Train DINOv2-small baseline (Kaggle GPU notebook).
# 1. Attach competition data + DINOv2 weights dataset
# 2. pip install / add src to path
# 3. Freeze folds if missing
# 4. Train one fold smoke, then full 5-fold
# 5. Log OOF macro AUC to docs/experiments.md

from pathlib import Path

print("Config: configs/baseline_dinov2_s.yaml")
print("Model factory: rsna_knee.models.baseline.create_baseline_model")
print("Dataset: rsna_knee.data.dataset.StudyDataset")
print("Loss: rsna_knee.training.loss.masked_bce_with_logits")
