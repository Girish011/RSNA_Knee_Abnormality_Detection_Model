#!/usr/bin/env python3
"""Dense 80-slot training cache (Kaggle CPU, one shard per kernel).

Slice picker, ordering, normalisation and 140 mm crop follow the public widedense inference
code (build_study_pair / _fill_variant_volume, primary 2-98% span) so that train and test
volumes are built identically; only the slot counts are scaled from 64 to 80.
Outputs in /kaggle/working: vols_s{SHARD}.npy (n,80,336,336) uint8, masks_s{SHARD}.npy,
ids_s{SHARD}.npy, meta_s{SHARD}.json.
"""
import glob
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

SHARD = __SHARD__
N_SHARDS = __N_SHARDS__
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15),
         ("Coronal", 0, 10), ("Axial", -1, 15)]
MAXS = sum(s[2] for s in SLOTS)
TIME_BUDGET_S = 8.3 * 3600
OUT = "/kaggle/working"
cv2.setNumThreads(1)


def find_root():
    for c in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.exists(c + "/train.csv"):
            return c
    for d, _, f in os.walk("/kaggle/input"):
        if "train_series.csv" in f:
            return d
    raise SystemExit("competition data not found")


def order_and_meta(sdir):
    fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
    for f in fs:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None); ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps); recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    med_ps = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps) for _, f, ps in recs], med_ps


def read_px(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps):
    h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def fill(vol, offset, picks, cache, files, med_ps):
    arrays, spacings = [], []
    for p in picks:
        p = min(int(p), len(files) - 1)
        arrays.append(cache.get(p)); ps = files[p][1]; spacings.append(ps if ps > 0 else med_ps)
    valid = [a for a in arrays if a is not None]
    lo, hi = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
              if valid else (0.0, 1.0))
    for j, (a, ps) in enumerate(zip(arrays, spacings)):
        if offset + j >= MAXS:
            break
        if a is None:
            continue
        x = mm_crop_resize(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1), ps)
        vol[offset + j] = (x * 255).astype(np.uint8)


def build_study(args):
    sid, rows, sdir = args
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    used, offset = set(), 0
    try:
        for plane, fluid, count in SLOTS:
            rec = pick_series(rows, plane, fluid, used)
            if rec is None:
                offset += count; continue
            used.add(rec["SeriesInstanceUID"])
            files, med_ps = order_and_meta(f"{sdir}/{sid}/{rec['SeriesInstanceUID']}")
            if not files:
                offset += count; continue
            n = len(files)
            lo = int(n * SPAN_LO); hi = max(int(n * SPAN_HI) - 1, lo)
            picks = (np.linspace(lo, hi, count).round().astype(int) if n > 1
                     else np.zeros(count, dtype=int))
            cache = {}
            for p in sorted(set(picks.tolist())):
                p = min(int(p), n - 1)
                try:
                    cache[p] = read_px(files[p][0])
                except Exception:
                    cache[p] = None
            fill(vol, offset, picks, cache, files, med_ps)
            offset += count
            if offset >= MAXS:
                break
        err = ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, err


def main():
    t0 = time.time()
    root = find_root()
    sdir = root + "/train_series"
    if not os.path.isdir(sdir):
        sdir = root + "/train_images"
    ids = pd.read_csv(root + "/train.csv")["StudyInstanceUID"].astype(str).tolist()
    ser = pd.read_csv(root + "/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    b = np.linspace(0, len(ids), N_SHARDS + 1).round().astype(int)
    mine = ids[b[SHARD]:b[SHARD + 1]]
    print(f"root {root} | series dir {sdir} | shard {SHARD}/{N_SHARDS}: {len(mine)} studies | "
          f"cpus {os.cpu_count()}", flush=True)

    vols = np.lib.format.open_memmap(f"{OUT}/vols_s{SHARD}.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(mine), MAXS, IMG, IMG))
    masks = np.zeros((len(mine), MAXS), np.uint8)
    errors, done = {}, 0
    jobs = [(sid, SER.get(sid, []), sdir) for sid in mine]
    with Pool(os.cpu_count()) as pool:
        for i, (vol, mask, err) in enumerate(pool.imap(build_study, jobs, chunksize=2)):
            vols[i] = vol; masks[i] = mask; done = i + 1
            if err:
                errors[mine[i]] = err
            if done % 50 == 0 or done == len(mine):
                el = time.time() - t0
                print(f"{done}/{len(mine)} | {el/60:.1f} min | eta {el/done*(len(mine)-done)/60:.1f} min "
                      f"| mean slots {masks[:done].sum(1).mean():.1f} | errors {len(errors)}", flush=True)
            if time.time() - t0 > TIME_BUDGET_S:
                print("time budget hit; writing partial shard", flush=True)
                break
    vols.flush()
    np.save(f"{OUT}/masks_s{SHARD}.npy", masks)
    np.save(f"{OUT}/ids_s{SHARD}.npy", np.array(mine))
    json.dump({"shard": SHARD, "n_shards": N_SHARDS, "n": len(mine), "done": done,
               "slots": SLOTS, "span": [SPAN_LO, SPAN_HI], "img": IMG, "crop_mm": CROP_MM,
               "errors": errors, "empty_studies": int((masks[:done].sum(1) == 0).sum()),
               "minutes": round((time.time() - t0) / 60, 1)},
              open(f"{OUT}/meta_s{SHARD}.json", "w"), indent=1)
    print("DONE", json.dumps({"done": done, "errors": len(errors)}), flush=True)


if __name__ == "__main__":
    main()
