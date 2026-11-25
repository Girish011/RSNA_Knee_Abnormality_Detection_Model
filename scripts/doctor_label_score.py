#!/usr/bin/env python3
"""Read the doctor's progress file(s): warm-up agreement with the competition experts + labels for the main cases.

    python scripts/doctor_label_score.py outputs/doctor_label/returns/knee_labels_*.json

Several files are merged (the newest answer per case wins). Writes data/processed/labels_doctor.parquet
(StudyInstanceUID + twelve findings, Unsure = NaN, plus reader / seconds / inadequate) for the finished main cases.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> None:
    files = [Path(p) for p in sys.argv[1:]]
    assert files, "pass at least one progress file"
    merged, reader = {}, ""
    for f in files:
        d = json.loads(f.read_text())
        reader = d.get("reader") or reader
        for c, r in d.get("cases", {}).items():
            if c not in merged or r.get("ts", 0) > merged[c].get("ts", 0):
                merged[c] = r
    cases = {c["case"]: c for c in json.loads((REPO / "outputs" / "doctor_label" / "cases.json").read_text())}
    gold = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    done = {c: r for c, r in merged.items() if r.get("done") and c in cases}
    warm = {c: r for c, r in done.items() if cases[c]["set"] == "warmup"}
    main_ = {c: r for c, r in done.items() if cases[c]["set"] == "main"}
    secs = [r.get("t", 0) for r in done.values()]
    print(f"reader {reader!r} | finished: warm-up {len(warm)}/20, main {len(main_)}/300 | "
          f"median {np.median(secs) / 60 if secs else 0:.1f} min per case")
    if warm:
        print(f"\nWarm-up vs competition experts ({len(warm)} cases; Unsure counts as a miss)")
        print(f"{'finding':18s} agree   expert+  doctor+  both+")
        tot_a = tot_n = 0
        for k in LAB:
            e = np.array([int(gold.loc[cases[c]["uid"], k]) for c in warm])
            d = np.array([r["labels"].get(k, -1) for r in warm.values()])
            agree = int((d == e).sum()); tot_a += agree; tot_n += len(e)
            print(f"{k:18s} {agree:2d}/{len(e):2d}   {int(e.sum()):3d}      {int((d == 1).sum()):3d}     {int(((d == 1) & (e == 1)).sum()):3d}")
        print(f"{'all findings':18s} {tot_a}/{tot_n} = {tot_a / tot_n:.3f}")
    if main_:
        rows = []
        for c, r in main_.items():
            row = {"StudyInstanceUID": cases[c]["uid"], "case": c, "fold": cases[c].get("fold"), "reader": reader,
                   "seconds": r.get("t", 0), "inadequate": bool(r.get("badq")), "note": r.get("note", "")}
            for k in LAB:
                v = r["labels"].get(k, -1)
                row[k] = np.nan if v not in (0, 1) else float(v)
            rows.append(row)
        df = pd.DataFrame(rows)
        out = REPO / "data" / "processed" / "labels_doctor.parquet"
        df.to_parquet(out, index=False)
        print(f"\nwrote {out.relative_to(REPO)}: {len(df)} studies | positives per finding: "
              + ", ".join(f"{k} {int(df[k].sum())}" for k in LAB))


if __name__ == "__main__":
    main()
