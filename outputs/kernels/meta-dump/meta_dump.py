#!/usr/bin/env python3
"""Per-study DICOM header dump (Kaggle CPU): patient / scanner tags for train and the public test stub.

One header per series (first file), no pixels. Outputs in /kaggle/working: meta_train.csv, meta_test.csv,
tags_seen.json (keyword -> number of studies carrying a non-empty value, first 300 train studies).
"""
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import pandas as pd
import pydicom

TAGS = ["PatientSex", "PatientAge", "PatientBirthDate", "PatientWeight", "PatientSize", "Manufacturer",
        "ManufacturerModelName", "MagneticFieldStrength", "InstitutionName", "StationName", "StudyDate",
        "StudyDescription", "BodyPartExamined", "Laterality", "ImageLaterality", "SoftwareVersions",
        "DeviceSerialNumber", "ProtocolName", "PatientPosition", "ReceiveCoilName", "PatientID",
        "EthnicGroup", "PregnancyStatus", "AdditionalPatientHistory", "StudyComments", "ImageComments"]
SER_TAGS = ["SeriesDescription", "RepetitionTime", "EchoTime", "SliceThickness", "PixelSpacing", "Rows", "Columns"]


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
    sid, series, sdir, full = args
    row = {"StudyInstanceUID": sid, "n_series": len(series)}
    seen = []
    nfiles = []
    for j, ser in enumerate(series):
        fs = sorted(glob.glob(f"{sdir}/{sid}/{ser}/*.dcm"))
        nfiles.append(len(fs))
        if not fs:
            continue
        try:
            h = pydicom.dcmread(fs[0], stop_before_pixels=True, force=True)
        except Exception as e:
            row.setdefault("err", f"{type(e).__name__}"); continue
        for k in TAGS:
            v = str(getattr(h, k, "") or "").strip()
            if v and not row.get(k):
                row[k] = v[:80]
        row[f"s{j}"] = "|".join(str(getattr(h, k, "") or "")[:40] for k in SER_TAGS)
        if full and j == 0:
            seen = [e.keyword for e in h if e.keyword and str(e.value).strip() not in ("", "None")]
    row["n_files"] = sum(nfiles)
    return row, seen


def dump(root, split, full_n):
    csv = f"{root}/{split}_series.csv"
    if not os.path.exists(csv):
        return
    ser = pd.read_csv(csv, dtype=str)
    sdir = f"{root}/{split}_series"
    grp = ser.groupby("StudyInstanceUID")["SeriesInstanceUID"].apply(list)
    jobs = [(sid, s, sdir, i < full_n) for i, (sid, s) in enumerate(grp.items())]
    rows, seen = [], Counter()
    with Pool(os.cpu_count()) as pool:
        for i, (row, kws) in enumerate(pool.imap(one, jobs, chunksize=8)):
            rows.append(row); seen.update(set(kws))
            if (i + 1) % 500 == 0:
                print(split, i + 1, len(jobs), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"/kaggle/working/meta_{split}.csv", index=False)
    json.dump(dict(seen.most_common()), open(f"/kaggle/working/tags_seen_{split}.json", "w"), indent=1)
    print(split, df.shape)
    for k in TAGS:
        if k in df:
            print(f"  {k}: non-empty {df[k].notna().sum()} | top {df[k].value_counts().head(6).to_dict()}")


if __name__ == "__main__":
    root = find_root()
    print("root", root, flush=True)
    dump(root, "test", 10)
    dump(root, "train", 300)
    print("DONE")
