#!/usr/bin/env python3
"""Launch region-specialist pods (runpod/region_spec.py on the dense512 cache), one pod per (specialist, fold).

    python scripts/launch_region.py --labels labels_t2.parquet --specs tf,pf,rec,recsf --folds 0,1 --dry-run
    python scripts/launch_region.py --labels labels_t2.parquet --specs tf,pf,rec,recsf --folds 0,1

Specialists (12 Oct plan; targets = the expert-trained checkpoints' ordering, labels_t2):
  tf     one tibiofemoral compartment per sample (lateral + mirrored medial)  -> Lateral OA (and the meniscus / medial columns)
  pf     patellofemoral: axial anterior + mid-sagittal anterior               -> PF OA
  rec    recesses: axial + sagittal fluid / non-fluid                         -> Synovitis (Effusion, Baker's)
  recsf  rec with --chan seqfuse (fluid, registered non-fluid, difference)    -> Synovitis
  tfg    tf + --aux_grades: three masked ordinal outputs for the report severity grade of the compartment's OA
  pfg    pf + --aux_grades                                                     (labels_graded_v1.parquet, 6 Oct pass)
  bm     bone marrow, wide coronal / sagittal / axial fields                    -> Contusion, Fracture (16 Oct)
Job `rs2-<spec>-f<fold>` -> dataset girishbose/rsna-knee-rp-rs2-<spec>-f<fold>-out with <tag>/raptor_rs_<tag>.pt (EMA)
and _last.pt; probe: scripts/build_region_probe.py --dataset ... --spec "Finding|raptor_rs_<tag>_last.pt,raptor_rs_<tag>.pt|1.0".
The knee-side file of the train studies travels in the bundle (launcher.build_bundle).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rsna_knee.agent.launcher import launch

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "agent" / "runs" / "region_jobs.json"
COMMON = "--epochs 16 --grad_ckpt --workers 8"
SPECS = {"tf": ("--region tf", 4.5), "pf": ("--region pf", 5.0), "rec": ("--region rec", 6.0),
         "recsf": ("--region rec --chan seqfuse", 6.5),      # (args, pod hours incl. the 93 GB corpus download)
         "tfg": ("--region tf --aux_grades labels_graded_v1.parquet", 4.5),
         "pfg": ("--region pf --aux_grades labels_graded_v1.parquet", 5.0),
         "bm": ("--region bm", 6.0)}      # bone marrow: coronal / sagittal fluid, axial, coronal non-fluid -> Contusion, Fracture


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--prefix", default="rs2", help="job prefix (a new round of the same specialist needs a new prefix)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    assert (REPO / "data" / "processed" / a.labels).exists(), a.labels
    jobs = json.loads(LOG.read_text()) if LOG.exists() else []
    for sp in a.specs.split(","):
        args, hours = SPECS[sp]
        for fold in (int(x) for x in a.folds.split(",")):
            job, tag = f"{a.prefix}-{sp}-f{fold}", f"{a.prefix}{sp}{fold}"
            arms = [(tag, a.labels, f"{args} --fold {fold} {COMMON}")]
            if a.dry_run or any(j["job"] == job for j in jobs):
                print(("DRY " if a.dry_run else "already launched: ") + job, arms)
                continue
            out = launch(job, arms, hours, corpus="dense512", gpu_class="48", trainer="region_spec.py")
            jobs.append({"job": job, "tag": tag, "spec": sp, "fold": fold, "labels": a.labels, "pod": out["pod"],
                         "price": out["costPerHr"], "launched": time.strftime("%Y-%m-%dT%H:%M:%S")})
            LOG.write_text(json.dumps(jobs, indent=1))
            print(job, out["pod"], out["costPerHr"], flush=True)


if __name__ == "__main__":
    main()
