# Offline Kaggle submit — MAIN track
# Requirements: internet off, write /kaggle/working/submission.csv, <= 9h

from pathlib import Path

print("Load fold checkpoints, run StudyDataset over test_series, blend, write submission.csv")
print("Validate with rsna_knee.infer.validate_submission")
