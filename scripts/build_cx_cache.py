#!/usr/bin/env python3
"""Cache for the public 2.5D ConvNeXt reader recipe (goodpjw2008/rsna-knee-2-5d-convnext-reader), built with ITS
preprocess.load_series unchanged, on free Kaggle CPU kernels (9 Oct).

    python scripts/build_cx_cache.py [--shards 6] [--push]

Kernel `girishbose/cx-cache-s<k>` writes one compressed file per study, `<StudyInstanceUID>.npz` (key = series uid,
value = uint8 [n, 384, 384]), and `meta_s<k>.csv` (series, n, plane, errors). Training reads them through the
recipe's `loader=` hook.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = r'''
import glob, os, subprocess, sys, time, types
import numpy as np, pandas as pd
from multiprocessing import Pool
SHARD, NSHARDS, SPLIT = __SHARD__, __NSHARDS__, "train"
t0 = time.time()
pp = [p for p in glob.glob("/kaggle/input/**/preprocess.py", recursive=True)]
assert pp, "public recipe dataset not mounted"
rd = os.path.dirname(pp[0])
whl = [w for w in glob.glob(rd + "/*.whl") if ("cp%d%d" % sys.version_info[:2]) in w or "py3-none" in w]
print("recipe dir", rd, "| wheels", [os.path.basename(w) for w in whl], flush=True)
if whl:
    print(subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], capture_output=True, text=True).stderr[-300:], flush=True)
root = next(d for d in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection") if os.path.exists(d + f"/{SPLIT}_series.csv"))
code = open(pp[0]).read()
pre = types.ModuleType("pre"); pre.__dict__["__name__"] = "pre"
exec(compile(code, pp[0], "exec"), pre.__dict__)      # their file, unchanged; only load_series is used
se = pd.read_csv(root + f"/{SPLIT}_series.csv")
studies = sorted(se.StudyInstanceUID.unique())[SHARD::NSHARDS]
grp = {s: g for s, g in se[se.StudyInstanceUID.isin(studies)].groupby("StudyInstanceUID")}
print(f"shard {SHARD}/{NSHARDS}: {len(studies)} studies, {sum(len(g) for g in grp.values())} series", flush=True)
OUT = "/kaggle/working"

def job(st):
    rows, vols = [], {}
    for sr, pl in zip(grp[st].SeriesInstanceUID, grp[st].Anatomical_Plane):
        info = {"study": st, "series": sr}
        try:
            vol, extra = pre.load_series(f"{root}/{SPLIT}_series/{st}/{sr}", pl)
            vols[sr] = vol; info.update(extra)
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(info)
    np.savez_compressed(f"{OUT}/{st}.npz", **vols)
    return rows

if __name__ == "__main__":
    rows = []
    with Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(job, studies, chunksize=2)):
            rows += r
            if i % 100 == 0:
                gb = sum(os.path.getsize(f) for f in glob.glob(OUT + "/*.npz")) / 1e9
                print(f"{i}/{len(studies)} | {time.time() - t0:.0f}s | {gb:.1f} GB", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/meta_s{SHARD}.csv", index=False)
    print("done", len(df), "series | errors", int(df.get("error", pd.Series(dtype=str)).notna().sum()), "| n<3:", int((df.get("n", 0) < 3).sum()),
          "| GB", round(sum(os.path.getsize(f) for f in glob.glob(OUT + "/*.npz")) / 1e9, 1), f"| {time.time() - t0:.0f}s", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    for k in range(a.shards):
        d = REPO / "outputs" / "kernels" / f"cx-cache-s{k}"
        d.mkdir(parents=True, exist_ok=True)
        src = SRC.replace("__SHARD__", str(k)).replace("__NSHARDS__", str(a.shards))
        compile(src, "cx", "exec")
        (d / "cx_cache.py").write_text(src)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"girishbose/cx-cache-s{k}", "title": f"cx-cache-s{k}", "code_file": "cx_cache.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader"], "kernel_sources": [], "model_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=1))
        if a.push:
            r = subprocess.run([str(Path(sys.executable).with_name("kaggle")), "kernels", "push", "-p", str(d)], capture_output=True, text=True)
            print(k, (r.stdout + r.stderr).strip().splitlines()[-1][:110])
    print(f"{a.shards} kernels in outputs/kernels/cx-cache-s*")


if __name__ == "__main__":
    main()
