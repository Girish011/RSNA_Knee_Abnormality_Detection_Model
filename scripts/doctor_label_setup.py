#!/usr/bin/env python3
"""Build the doctor's labelling folder from the export kernel's output.

    python scripts/doctor_label_setup.py            # download + unpack + manifest
    python scripts/doctor_label_setup.py --no-download

Fetches girishbose/rsna-knee-doctor-export (one zip per case), unpacks to outputs/doctor_label/KneeLabelling/data/<case>/ and
writes KneeLabelling/manifest.js for KneeLabelling/index.html (that folder is what the doctor receives). The page gets case ids only; warm-up cases (20 of the 58 expert
studies) also carry the expert labels so the page can show them after the doctor has answered. The case -> study key
stays in outputs/doctor_label/cases.json (not shipped). The folder holds competition images: team members only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "doctor_label"
KERNEL = "girishbose/rsna-knee-doctor-export"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    zips = OUT / "zips"
    if not a.no_download:
        zips.mkdir(parents=True, exist_ok=True)
        kaggle = str(Path(sys.executable).with_name("kaggle"))
        subprocess.run([kaggle, "kernels", "output", KERNEL, "-p", str(zips)], check=True)
    cases = json.loads((OUT / "cases.json").read_text())
    metas = {m["case"]: m for m in json.loads((zips / "manifest.json").read_text())}
    gold = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    pkg = OUT / "KneeLabelling"
    data = pkg / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest, slices, empty = [], 0, []
    for c in cases:
        cid = c["case"]
        z = zips / f"{cid}.zip"
        if z.exists() and not (data / cid / "meta.json").exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(data)
        m = metas.get(cid, {"series": []})
        row = {"case": cid, "set": c["set"], "series": m["series"]}
        if c["set"] == "warmup":
            row["expert"] = {k: int(gold.loc[c["uid"], k]) for k in LAB}
        if not m["series"]:
            empty.append(cid)
        slices += sum(s["n"] for s in m["series"])
        manifest.append(row)
    (pkg / "manifest.js").write_text("window.MANIFEST = " + json.dumps(manifest, separators=(",", ":")) + ";\n")
    size = sum(p.stat().st_size for p in data.rglob("*.jpg")) / 1e9
    print(f"{len(manifest)} cases | {slices} slices | {size:.2f} GB images | empty cases: {empty or 'none'}")
    print(f"open {pkg / 'index.html'}")


if __name__ == "__main__":
    main()
