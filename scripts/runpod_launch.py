#!/usr/bin/env python3
"""Launch a hand-specified RunPod job (the agent loop calls ``rsna_knee.agent.launcher`` directly)."""

from __future__ import annotations

import argparse
import json

from rsna_knee.agent.launcher import build_bundle, launch

COATNET = ("--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 16 --bs 8 --k 12 "
           "--k_eval 24 --grad_ckpt --norm imagenet --seed 42 --workers 8")
JOBS = {
    "labels-ab-v1": {
        "arms": [("dread_s42", "labels_dread.parquet", COATNET),
                 ("blendv2_s42", "labels_llm_blend_v2.parquet", COATNET)],
        "hours": 7.0,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = JOBS[a.job]
    if a.dry_run:
        print(sorted(p.name for p in build_bundle(a.job, spec["arms"]).iterdir()))
        return
    print(json.dumps(launch(a.job, spec["arms"], spec["hours"]), indent=1))


if __name__ == "__main__":
    main()
