#!/usr/bin/env python3
"""Own-alone LB probe kernel: mean of N dense80 CoAt checkpoints through the `submit-dense80-5bc0` leg (LB 0.920).

    python scripts/build_probe_kernel.py --name probe-r1-coat5 --jobs r1-coat-f0:r1coat0,r1-coat-f1:r1coat1,...
    kaggle kernels push -p outputs/kernels/probe-r1-coat5        # then scripts/submit_gate.py

Each job:tag is the private result dataset `rsna-knee-rp-<job>-out` holding `<tag>/raptor_ft_<tag>.pt`.
The kernel is the 5bc0 submit script with one change: every study is scored by all checkpoints and the
probabilities are averaged (same windows, same 2 eval passes). One job:tag = the single-model leg itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs" / "kernels" / "submit-dense80-5bc0"
USER = "girishbose"


def rep(s: str, old: str, new: str) -> str:
    assert s.count(old) == 1, f"expected exactly one {old!r} in the base kernel"
    return s.replace(old, new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs", required=True, help="comma list of job:tag")
    a = ap.parse_args()
    pairs = [x.split(":") for x in a.jobs.split(",")]
    names = [f"raptor_ft_{t}.pt" for _, t in pairs]
    s = (BASE / "submit_dense80_5bc0.py").read_text()
    s = rep(s, 'WEIGHT_NAME = "raptor_ft_5bc0a0f45d75.pt"', f"WEIGHT_NAMES = {names!r}")
    s = rep(s, "    wp = find_weight_file(WEIGHT_NAME)\n    model, res = load_model(wp, dev)\n"
               '    print(f"loaded {wp} | res {res} | {time.time() - t0:.0f}s", flush=True)\n',
            "    models = []\n    for wn in WEIGHT_NAMES:\n        wp = find_weight_file(wn)\n"
            "        model, res = load_model(wp, dev)\n        models.append((model, res))\n"
            '        print(f"loaded {wp} | res {res} | {time.time() - t0:.0f}s", flush=True)\n')
    # each checkpoint is scored at its own training resolution (families at 224 and 336 px can share one probe)
    s = rep(s, "                xw = eval_windows(vol, mask, k=K_EVAL, res=res, phase=ph, norm=NORM)\n"
               "                acc.append(infer_probs(model, xw, dev))\n                del xw\n",
            "                for r_ in sorted({r for _, r in models}):\n"
            "                    xw = eval_windows(vol, mask, k=K_EVAL, res=r_, phase=ph, norm=NORM)\n"
            "                    acc.extend(infer_probs(m, xw, dev) for m, mr in models if mr == r_)\n"
            "                    del xw\n")
    s = rep(s, "    del model\n", "    del models\n")
    d = REPO / "outputs" / "kernels" / a.name
    d.mkdir(parents=True, exist_ok=True)
    code = a.name.replace("-", "_") + ".py"
    (d / code).write_text(s)
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta.update(id=f"{USER}/{a.name}", title=a.name, code_file=code,
                dataset_sources=[f"{USER}/rsna-knee-rp-{j}-out" for j, _ in pairs])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    compile(s, code, "exec")
    print(f"wrote {d} | {len(pairs)} checkpoints | datasets {meta['dataset_sources']}")


if __name__ == "__main__":
    main()
