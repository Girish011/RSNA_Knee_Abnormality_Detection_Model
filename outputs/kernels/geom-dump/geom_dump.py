#!/usr/bin/env python3
"""Per-series geometry dump (Kaggle CPU, headers only): orientation, slice-order direction, position of the volume in
patient coordinates, laterality tags. Answers: can knee side be read from the headers for every study (the Laterality
tag exists for half of train), and how much do in-plane orientation and slice order vary between series?

Output /kaggle/working/geom_{train,test}.csv, one row per series:
  iop (6 values of the first slice), nx ny nz (slice normal = row x col), ps, rows, cols, n,
  cx cy cz (centre of the stack in patient LPS mm), sx sy sz (std of the slice positions per axis),
  Laterality, ImageLaterality, SeriesDescription, ProtocolName, PatientPosition, BodyPartExamined.
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom

TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns", "Laterality",
        "ImageLaterality", "SeriesDescription", "ProtocolName", "PatientPosition", "BodyPartExamined", "InstanceNumber"]


def find_root():
    for c in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.exists(c + "/train.csv"):
            return c
    for d, _, f in os.walk("/kaggle/input"):
        if "train_series.csv" in f:
            return d
    raise SystemExit("competition data not found")


def one(args):
    sid, ser, sdir, plane = args
    row = {"StudyInstanceUID": sid, "SeriesInstanceUID": ser, "plane": plane}
    fs = glob.glob(f"{sdir}/{sid}/{ser}/*.dcm")
    row["n"] = len(fs)
    ipps, first = [], None
    for f in fs:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS)
        except Exception as e:
            row["err"] = type(e).__name__
            continue
        if first is None:
            first = h
        ipp = getattr(h, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            ipps.append([float(v) for v in ipp])
    if first is None:
        return row
    for k in ("Laterality", "ImageLaterality", "SeriesDescription", "ProtocolName", "PatientPosition", "BodyPartExamined"):
        row[k] = str(getattr(first, k, "") or "").strip()[:60]
    iop = getattr(first, "ImageOrientationPatient", None)
    ps = getattr(first, "PixelSpacing", None)
    rows_, cols_ = int(getattr(first, "Rows", 0) or 0), int(getattr(first, "Columns", 0) or 0)
    row.update(rows=rows_, cols=cols_, ps=float(ps[0]) if ps is not None else np.nan)
    if iop is not None and len(iop) == 6 and ipps:
        r, c = np.array(iop[:3], float), np.array(iop[3:], float)
        nrm = np.cross(r, c)
        P = np.array(ipps)
        psx = float(ps[1]) if ps is not None else 0.5
        psy = float(ps[0]) if ps is not None else 0.5
        cen = P.mean(0) + 0.5 * cols_ * psx * r + 0.5 * rows_ * psy * c
        row.update({f"iop{i}": round(float(v), 4) for i, v in enumerate(iop)})
        row.update(nx=round(float(nrm[0]), 4), ny=round(float(nrm[1]), 4), nz=round(float(nrm[2]), 4),
                   cx=round(float(cen[0]), 1), cy=round(float(cen[1]), 1), cz=round(float(cen[2]), 1),
                   sx=round(float(P[:, 0].std()), 1), sy=round(float(P[:, 1].std()), 1), sz=round(float(P[:, 2].std()), 1))
    return row


def dump(root, split):
    csv = f"{root}/{split}_series.csv"
    if not os.path.exists(csv):
        return
    ser = pd.read_csv(csv, dtype=str)
    sdir = f"{root}/{split}_series"
    if not os.path.isdir(sdir):
        sdir = f"{root}/{split}_images"
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID, sdir, r.Anatomical_Plane) for r in ser.itertuples()]
    rows = []
    with Pool(os.cpu_count()) as pool:
        for i, row in enumerate(pool.imap(one, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 2000 == 0:
                print(split, i + 1, len(jobs), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"/kaggle/working/geom_{split}.csv", index=False)
    print(split, df.shape, flush=True)
    if "Laterality" in df:
        print(df.groupby("plane")["Laterality"].value_counts(dropna=False).head(20).to_string(), flush=True)


if __name__ == "__main__":
    root = find_root()
    print("root", root, flush=True)
    dump(root, "test")
    dump(root, "train")
    print("DONE")
