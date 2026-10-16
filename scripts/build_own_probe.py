#!/usr/bin/env python3
"""Own-system-alone leaderboard probe: the leg `_ours_leg/ours_leg_v2.py` IS the submission (rank mean of the
members, no public stack), optionally followed by the `--decor` post-processing of build_ensemble_kernel.py.

    python scripts/build_own_probe.py --name own20-alone --members job:tag,job:tag,... [--decor 'Lateral OA|Medial OA|0.3' ...]
    kaggle kernels push -p outputs/kernels/own20-alone ; check the log ; scripts/submit_gate.py submit ...

Each job:tag is the result dataset `girishbose/rsna-knee-rp-<job>-out` holding `raptor_ft_<tag>.pt`. About 2.4 s a
study for fifteen models on 2 x T4 (leg-bench, 5 Oct): twenty models score the hidden test in ~1.3 h.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "girishbose"
LEG = REPO / "outputs" / "kernels" / "_ours_leg" / "ours_leg_v2.py"
SRC = '''import json, os, subprocess, sys, glob
W = "/kaggle/working"
open(W + "/_leg.py", "w").write(__LEG__)
env = dict(os.environ, OURS_MEMBERS=json.dumps(__FILES__), OURS_OUT=W + "/submission.csv", **__ENV__)
rc = subprocess.call([sys.executable, W + "/_leg.py"], env=env)
print(f"[own] leg exit {rc}", flush=True)
if rc != 0:
    raise SystemExit(f"leg failed rc={rc}")
for f in glob.glob(W + "/submission_m*.csv") + [W + "/_leg.py"]:
    os.remove(f)
import pandas as pd
sub = pd.read_csv(W + "/submission.csv")
print("[own] submission.csv", sub.shape, list(sub.columns)[:3], flush=True)
__DECOR__
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", required=True, help="comma list of job:tag (pod result dataset) or k/<kernel>:tag (Kaggle kernel output)")
    ap.add_argument("--decor", action="append", default=[])
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the leg, e.g. OURS_FLIP=1 OURS_K_EVAL=39 OURS_PASSES=4")
    ap.add_argument("--negate", default=None, help="';'-separated findings whose column order is reversed "
                                                   "(per-finding LB probe: AUC_f = (1 + 12 * (base - LB)) / 2)")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("bek", REPO / "scripts" / "build_ensemble_kernel.py")
    bek = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bek)
    pairs = [x.rsplit(":", 1) for x in a.members.split(",") if x]
    files = [f"raptor_ft_{t}.pt" for _, t in pairs]
    decor = [bek.parse_decor(x) for x in a.decor]
    cell = bek.DECOR.replace("__SPEC__", repr([list(x) for x in decor])) if decor else ""
    negate = [x for x in a.negate.split(";") if x] if a.negate else []
    if bad := [f for f in negate if f not in bek.TARGETS]:
        raise SystemExit(f"--negate: unknown findings {bad}")
    if negate:
        cell += "\n" + bek.NEGATE.replace("__COLS__", repr(negate))
    env = dict(x.split("=", 1) for x in a.env)
    src = (SRC.replace("__LEG__", repr(LEG.read_text())).replace("__FILES__", repr(files)).replace("__DECOR__", cell)
           .replace("__ENV__", repr(env)))
    compile(src, a.name, "exec")
    d = REPO / "outputs" / "kernels" / a.name
    d.mkdir(parents=True, exist_ok=True)
    code = a.name.replace("-", "_") + ".py"
    (d / code).write_text(src)
    meta = json.loads((REPO / "outputs/kernels/leg-bench/kernel-metadata.json").read_text())
    meta.update(id=f"{USER}/{a.name}", title=a.name, code_file=code,
                dataset_sources=[f"{USER}/rsna-knee-rp-{j}-out" for j, _ in pairs if not j.startswith("k/")],
                kernel_sources=list(dict.fromkeys(f"{USER}/{j[2:]}" for j, _ in pairs if j.startswith("k/"))))
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    (d / "PROBE.json").write_text(json.dumps({"members": a.members.split(","), "decor": [list(x) for x in decor],
                                               "negate": negate}, indent=1))
    print(f"wrote {d} | {len(files)} members | decor {len(decor)} | push: kaggle kernels push -p {d}")


if __name__ == "__main__":
    main()
