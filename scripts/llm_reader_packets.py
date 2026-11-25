#!/usr/bin/env python3
"""Blind packets for the LLM-reader test (9 Oct): each warm-up study is copied under a random id, with no study uid,
report or label inside, plus the per-finding guide and an answer template. The id -> case map stays in the repo
(outputs/doctor_label/claude_reader/map_<set>.json) and is not given to the reading agents.

    python scripts/llm_reader_packets.py --out <dir> [--set warmup] [--seed 9]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DL = REPO / "outputs" / "doctor_label"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def guide() -> str:
    s = (DL / "KneeLabelling" / "reference.html").read_text()
    s = s[s.index("const INFO = {"):]
    out = ["# The twelve findings (textbook-level notes; the competition's experts are the reference)\n"]
    for k in LAB:
        m = re.search(r'"%s": \{ def: "(.*?)",\s*what: "(.*?)",\s*look: "(.*?)",' % re.escape(k), s, re.S)
        out.append(f"## {k} - {m.group(1)}\nWhat it looks like: {m.group(2)}\nWhere to look: {m.group(3)}\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", default="warmup")
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    cases = [c["case"] for c in json.loads((DL / "cases.json").read_text()) if c["set"] == a.set]
    rng = random.Random(a.seed)
    rng.shuffle(cases)
    a.out.mkdir(parents=True, exist_ok=True)
    g = guide()
    mapping = {}
    for case in cases:
        rid = "k" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
        d = a.out / rid
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(DL / "KneeLabelling" / "data" / case, d)
        meta = json.loads((d / "meta.json").read_text())
        (d / "meta.json").write_text(json.dumps({"case": rid, "series": meta["series"]}))
        (d / "guide.md").write_text(g)
        (d / "answer_template.json").write_text(json.dumps({
            "packet": rid,
            "findings": {k: {"answer": "Present | Absent | Unsure", "confidence": 0.0,
                             "evidence": "series and slice numbers this rests on, and what is seen there"} for k in LAB},
            "views_used": 0, "notes": ""}, indent=1))
        mapping[rid] = case
    m = DL / "claude_reader"
    m.mkdir(exist_ok=True)
    (m / f"map_{a.set}.json").write_text(json.dumps({"seed": a.seed, "order": list(mapping), "map": mapping}, indent=1))
    print(f"{len(mapping)} packets -> {a.out} | order (pilot = first two): {list(mapping)[:4]} ...")


if __name__ == "__main__":
    main()
