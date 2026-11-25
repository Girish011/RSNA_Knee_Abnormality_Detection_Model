#!/usr/bin/env python3
"""Build the DINO warm-start fine-tune kernel (cross-architecture distillation, PLAN_TO_TOP "Next build").

Takes the public trainer pilkwang/rsna-knee-baseline-v1 (pixel pipeline, model, fingerprint,
predict), keeps its definitions, backs the 12-slice train cache with a disk memmap (~36 GB does
not fit in Kaggle RAM) and replaces main() with scripts/dino_ft_main.py: fine-tune each of the
20 public DINO members (pilkwang/rsna-knee-weights) on our fused labels, paired holdout OOF.

    python scripts/build_dino_ft_kernel.py [--epochs 3]
    kaggle kernels push -p outputs/kernels/fork-dino-ft-fused
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "girishbose"
SRC = "pilkwang/rsna-knee-baseline-v1"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))

ALLOC_OLD = "    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)\n"
ALLOC_NEW = '''    shape = (len(studies), N_SLOT, CACHE_SLICES, IMG, IMG)
    if CACHE_MEMMAP[0]:                     # FORK: disk-backed train cache (36 GB at 336 px x 12)
        cache = np.lib.format.open_memmap(str(Path(CACHE_MEMMAP[0]) / f"cache_{tag.replace(' ', '_')}.npy"),
                                          mode="w+", dtype=np.uint8, shape=shape)
    else:
        cache = np.zeros(shape, np.uint8)
'''


def build(epochs: int) -> Path:
    name = "fork-dino-ft-fused" if epochs == 3 else f"fork-dino-ft-fused-e{epochs}"
    d = REPO / "outputs" / "kernels" / name
    up = REPO / "outputs" / "kernels" / "fork-dino-ft-fused" / "_upstream"
    if not (up / "kernel-metadata.json").exists():
        subprocess.run([KAGGLE, "kernels", "pull", SRC, "-m", "-p", str(up)], check=True)
    meta = json.loads((up / "kernel-metadata.json").read_text())
    nb = json.loads((up / meta["code_file"]).read_text())

    cells, patched = [], 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        t = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "_find_cover" in t or "def main():" in t or t.lstrip().startswith("try:\n    main()"):
            continue                                   # cover image, upstream main, its runner
        if ALLOC_OLD in t:
            t = "CACHE_MEMMAP = [None]\n\n" + t.replace(ALLOC_OLD, ALLOC_NEW)
            patched += 1
        cells.append(t)
    if patched != 1:
        raise SystemExit(f"cache allocation anchor matched {patched} cells (want 1)")
    main = (REPO / "scripts" / "dino_ft_main.py").read_text()
    if epochs != 3:
        main = main.replace('os.environ.get("FT_EPOCHS", "3")', f'os.environ.get("FT_EPOCHS", "{epochs}")')
    cells.append(main)

    d.mkdir(parents=True, exist_ok=True)
    out_nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                         "source": t} for t in cells],
              "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 4}
    (d / f"{name}.ipynb").write_text(json.dumps(out_nb, indent=1))
    py = "\n\n".join(cells).replace("from __future__ import annotations\n", "")
    (d / f"{name}.py").write_text("from __future__ import annotations\n" + py)   # local smoke / review copy
    km = {
        "id": f"{USER}/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["pilkwang/rsna-knee-weights", f"{USER}/rsna-knee-labels-ours"],
        "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "docker_image": meta.get("docker_image"), "machine_shape": "NvidiaTeslaT4",
    }
    (d / "kernel-metadata.json").write_text(json.dumps(km, indent=1))
    (d / "FORK.json").write_text(json.dumps({"src": SRC, "labels": "labels_fused_full.csv", "epochs": epochs,
                                             "lr": "public / 5", "members": "pilkwang/rsna-knee-weights x20",
                                             "gate": "gold OOF macro (58) > 0.8425 and > paired parent"},
                                            indent=1))
    print(f"built {d} ({len(cells)} cells)")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    build(ap.parse_args().epochs)
