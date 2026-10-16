
import glob, os, subprocess, sys, time, types
import numpy as np, pandas as pd
from multiprocessing import Pool
SHARD, NSHARDS, SPLIT = 2, 6, "train"
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
