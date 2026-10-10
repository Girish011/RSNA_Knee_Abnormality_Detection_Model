#!/usr/bin/env python3
"""Build the reference examples for the labelling folder (expert-labelled studies outside the warm-up set).

    python scripts/doctor_reference_setup.py [--no-download]

Fetches girishbose/rsna-knee-doctor-reference (one zip per study), unpacks to
outputs/doctor_label/KneeLabelling/ref_data/<ref>/ and writes KneeLabelling/reference.js for reference.html:
series, the experts' twelve labels and the original report text per study (and report_en when
outputs/doctor_label/reference_translations.json has one). Competition data: team members only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "doctor_label"
KERNEL = "girishbose/rsna-knee-doctor-reference"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def is_english(text: str) -> bool:
    t = str(text).lower()
    if re.search(r"[Ѐ-ӿ一-鿿؀-ۿ]", t):
        return False
    en = len(re.findall(r"\b(the|of|with|is|there|no|and|seen|noted)\b", t))
    other = len(re.findall(r"\b(el|la|de|con|sin|los|del|der|die|und|mit|ve|ile|izlen\w*|mevcut\w*|le|les|des|avec)\b", t))
    return en > other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    zips = OUT / "ref_zips"
    if not a.no_download:
        zips.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(Path(sys.executable).with_name("kaggle")), "kernels", "output", KERNEL, "-p", str(zips)], check=True)
    cases = json.loads((OUT / "reference_cases.json").read_text())
    metas = {m["case"]: m for m in json.loads((zips / "manifest.json").read_text())}
    tr = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    tp = OUT / "reference_translations.json"
    trans = json.loads(tp.read_text()) if tp.exists() else {}
    data = OUT / "KneeLabelling" / "ref_data"
    data.mkdir(parents=True, exist_ok=True)
    rows, slices = [], 0
    for c in cases:
        cid = c["case"]
        z = zips / f"{cid}.zip"
        if z.exists() and not (data / cid / "meta.json").exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(data)
        m = metas.get(cid, {"series": []})
        rep = str(tr.loc[c["uid"], "Report"])
        row = {"case": cid, "series": m["series"], "labels": {k: int(tr.loc[c["uid"], k]) for k in LAB},
               "report": rep, "lang": "en" if is_english(rep) else "other"}
        if cid in trans:
            row["report_en"] = trans[cid]
        slices += sum(s["n"] for s in m["series"])
        rows.append(row)
    (OUT / "KneeLabelling" / "reference.js").write_text("window.REFERENCE = " + json.dumps(rows, separators=(",", ":")) + ";\n")
    print(f"{len(rows)} reference studies | {slices} slices | non-English reports: "
          f"{[r['case'] for r in rows if r['lang'] != 'en' and 'report_en' not in r]}")


if __name__ == "__main__":
    main()
