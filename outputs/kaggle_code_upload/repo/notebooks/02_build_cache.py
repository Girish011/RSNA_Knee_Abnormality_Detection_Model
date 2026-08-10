# Build resized series cache on Kaggle (write to /kaggle/working then save as Dataset).
# Prefer running as a Kaggle notebook with long runtime; do not re-decode 570GB every train.

from pathlib import Path

print("See src/rsna_knee/data/cache.py::write_series_cache")
print("Wire study/series loops here after audit confirms paths and transfer syntaxes.")
print("Output manifest should record image_size, n_slices, code git sha, and count.")
