#!/usr/bin/env python3
"""Build a Kaggle submit kernel = public D4 stack (0.943) + our CoAtNet members, rank-blended.

The public stack writes ``submission.csv``; three appended cells then (1) stash it, (2) run
our leg (``outputs/kernels/_ours_leg/ours_leg.py``) in a subprocess against the same test
studies, (3) blend per finding in rank space with ONE global weight ``w``. Any failure in our
leg (error, deadline, constant column) restores the public output, so the floor is the base.

    python scripts/build_ensemble_kernel.py --name ens-d4-5bc0 --members 5bc0a0f45d75 --w 0.2
    python scripts/build_ensemble_kernel.py --name ens-d4-m3 --members m1234,m5678 --w 0.3
    python scripts/build_ensemble_kernel.py --name tail-pf --swap-v5 soupbestof4 \\
        --tail "PF OA|m5b5942352cf,m440e7701125|0.4" --tail "Synovitis;Contusion|soupX|0.3"

Members are online node ids (fold-0 weights, dataset ``rsna-knee-rp-n-<id>-out``) or
full-data member tags from ``agent/runs/members.json``. This script only BUILDS the kernel;
pushing and competition-submitting stay user-approved (see scripts/submit_gate.py).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs" / "kernels" / "_base_d4"
LEG = REPO / "outputs" / "kernels" / "_ours_leg" / "ours_leg.py"
MEMBERS = REPO / "agent" / "runs" / "members.json"
USER = "girishbose"
DOCKER = ("gcr.io/kaggle-private-byod/python@sha256:"
          "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
RUNTIME_S = 9 * 3600
LEG_MARGIN_S = 25 * 60  # leave time for the blend + Kaggle scoring overhead

STASH = '''# ==== OURS 1/3: stash the public stack output (fallback if anything below fails) ====
import shutil as _ens_shutil, time as _ens_time
import pandas as _ens_pd
_ENS_SUB = "/kaggle/working/submission.csv"
_ENS_PUB = "/kaggle/working/_public_stack.csv"
_ENS_TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
                "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_ens_pub = _ens_pd.read_csv(_ENS_SUB, dtype={"StudyInstanceUID": str})
assert _ens_pub.columns.tolist() == ["StudyInstanceUID", *_ENS_TARGETS], _ens_pub.columns.tolist()
assert _ens_pub[_ENS_TARGETS].notna().all().all()
_ens_shutil.copy2(_ENS_SUB, _ENS_PUB)
print(f"[ours 1/3] stashed public stack {_ens_pub.shape}", flush=True)
'''

RUN_LEG = '''# ==== OURS 2/3: our dense80 CoAtNet members (subprocess: clean GPU memory) ====
import gc as _ens_gc, json as _ens_json, os as _ens_os, subprocess as _ens_sp, sys as _ens_sys
try:
    import torch as _ens_torch
    _ens_gc.collect(); _ens_torch.cuda.empty_cache()
except Exception:
    pass
_ENS_LEG = "/kaggle/working/_ours_leg.py"
_ENS_OURS = "/kaggle/working/_ours.csv"
open(_ENS_LEG, "w").write(__LEG_SRC__)
_ens_t0 = globals().get("_ENS_START") or _ens_time.time()
_ens_deadline = float(_ens_t0) + __RUNTIME_S__ - __MARGIN_S__
_ens_env = dict(_ens_os.environ, OURS_MEMBERS=_ens_json.dumps(__MEMBER_FILES__),
                OURS_OUT=_ENS_OURS, OURS_DEADLINE=str(_ens_deadline))
print(f"[ours 2/3] leg start; {(_ens_deadline - _ens_time.time()) / 60:.0f} min before deadline", flush=True)
_ens_rc = _ens_sp.call([_ens_sys.executable, _ENS_LEG], env=_ens_env)
print(f"[ours 2/3] leg exit {_ens_rc}", flush=True)
'''

BLEND = '''# ==== OURS 3/3: per-finding rank blend; any problem -> public stack stands ====
import numpy as _ens_np
# finding -> (member indices into the leg, weight of our leg); findings not listed stay public.
_ENS_TAILS = __TAILS__
_ENS_NM = __N_MEMBERS__
def _ens_load(path):
    f = _ens_pd.read_csv(path, dtype={"StudyInstanceUID": str})
    f = f.set_index("StudyInstanceUID").reindex(_ens_pub["StudyInstanceUID"]).reset_index()
    if f[_ENS_TARGETS].isna().any().any():
        raise RuntimeError(f"{path} is missing studies")
    if len(_ens_pub) > 10 and float(f[_ENS_TARGETS].to_numpy(float).std(axis=0).min()) < 1e-9:
        raise RuntimeError(f"{path} has a constant column")
    return f[_ENS_TARGETS].rank(method="average", pct=True)
try:
    if _ens_rc != 0 or not _ens_os.path.exists(_ENS_OURS):
        raise RuntimeError(f"leg failed rc={_ens_rc}")
    _ens_all = _ens_load(_ENS_OURS)
    _ens_per = {}
    _ens_out = _ens_pub.copy()
    _ens_pr = _ens_pub[_ENS_TARGETS].rank(method="average", pct=True)
    _ens_log = []
    for _f, (_idx, _w) in _ENS_TAILS.items():
        if len(_idx) == _ENS_NM:
            _o = _ens_all[_f]
        else:
            for _j in _idx:
                if _j not in _ens_per:
                    _ens_per[_j] = _ens_load(_ENS_OURS[:-4] + f"_m{_j}.csv")
            _o = sum(_ens_per[_j][_f] for _j in _idx) / len(_idx)
        _ens_out[_f] = (1 - _w) * _ens_pr[_f].to_numpy() + _w * _o.rank(method="average", pct=True).to_numpy()
        _ens_log.append(f"{_f}:w{_w}/m{list(_idx)}/corr{_ens_pr[_f].corr(_o):.3f}")
    assert _ens_np.isfinite(_ens_out[_ENS_TARGETS].to_numpy(float)).all()
    _ens_out.to_csv(_ENS_SUB, index=False)
    print("[ours 3/3] blended " + " | ".join(_ens_log), flush=True)
except Exception as _ens_e:
    _ens_shutil.copy2(_ENS_PUB, _ENS_SUB)
    print(f"[ours 3/3] NOT blended ({type(_ens_e).__name__}: {_ens_e}); public stack stands", flush=True)
_ens_final = _ens_pd.read_csv(_ENS_SUB)
assert _ens_final.columns.tolist() == ["StudyInstanceUID", *_ENS_TARGETS] and len(_ens_final) == len(_ens_pub)
print(f"[ours] submission.csv ok {_ens_final.shape}", flush=True)
'''


def resolve_members(ids: list[str]) -> list[tuple[str, str]]:
    """-> [(dataset slug, weight file name)]."""
    done = json.loads(MEMBERS.read_text())["done"] if MEMBERS.exists() else {}
    out = []
    for i in ids:
        if i in done and done[i].get("eval_of"):
            raise SystemExit(f"{i} is an eval record of {done[i]['eval_of']}, not a model")
        if i in done:
            out.append((done[i]["dataset"], done[i]["weights"]))
        elif (i.startswith("m") and len(i) == 12) or i.startswith("soup"):
            raise SystemExit(f"member {i} not in {MEMBERS.name} done (still training or not collected yet)")
        else:
            out.append((f"{USER}/rsna-knee-rp-n-{i}-out", f"raptor_ft_{i}.pt"))
    return out


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


V5 = "raptor_ft_coatnet_v5_full_swa.pt"


def _src(cell: dict) -> str:
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def swap_v5(nb: dict, weights: list[str], datasets: list[str]) -> dict:
    """Point the stack's maxspan-v5 arms at our member(s): one weight → both arms (normal w0.60 +
    reverse w0.10 of the Raptor family); two → normal=first, reverse=second. Our datasets are
    registered with the asset finder. Everything else stays public."""
    nb = copy.deepcopy(nb)
    if isinstance(weights, str):
        weights, datasets = [weights], [datasets]
    if len(set(weights)) > 1:
        # The stack runs forward + reverse maxspan through ONE loaded model (fast path) and raises
        # "MaxSpan forward/reverse arms no longer share preprocessing" otherwise (seen 2026-09-29).
        raise SystemExit("maxspan forward/reverse share one model: swap a single member "
                         "(or a weight-soup of several, as one checkpoint)")
    order = [weights[0], weights[-1]]
    roots = "".join(f"'/kaggle/input/datasets/{d}', '/kaggle/input/{d.split('/', 1)[1]}', "
                    for d in dict.fromkeys(datasets))
    n_roots = n_arms = 0
    for c in nb["cells"]:
        t = _src(c)
        if "ASSET_ROOTS = [" in t and n_roots == 0:
            t = t.replace("ASSET_ROOTS = [", "ASSET_ROOTS = [" + roots, 1)
            n_roots += 1
        if '"name": "maxspan-v5"' in t:
            for w in order:
                if f'"file": "{V5}"' in t:
                    t = t.replace(f'"file": "{V5}"', f'"file": "{w}"', 1)
                    n_arms += 1
        c["source"] = t
    if n_roots != 1 or n_arms != 2:
        raise SystemExit(f"swap anchors drifted: roots {n_roots}, v5 arms {n_arms} (want 1, 2)")
    return nb


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEAK5 = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis", "Contusion"]


def parse_tail(spec: str) -> tuple[list[str], list[str], float]:
    """'PF OA;Synovitis|m1,m2|0.4' -> (findings, members, w); 'weak5' expands to the weak five."""
    try:
        fs, ms, w = spec.split("|")
    except ValueError:
        raise SystemExit(f"--tail wants 'FINDING;FINDING|member,member|w', got {spec!r}")
    findings = [x for f in fs.split(";") for x in (WEAK5 if f.strip() == "weak5" else [f.strip()])]
    if bad := [f for f in findings if f not in TARGETS]:
        raise SystemExit(f"unknown findings {bad}; use {TARGETS}")
    members = [m.strip() for m in ms.split(",") if m.strip()]
    if not members or not 0.0 < float(w) < 1.0:
        raise SystemExit(f"--tail {spec!r}: need members and 0 < w < 1")
    return findings, members, float(w)


def build(name: str, member_ids: list[str], w: float, out_root: Path | None = None,
          swap: str | None = None, tails: list[tuple[list[str], list[str], float]] | None = None) -> Path:
    """Global mode: every finding blends all ``member_ids`` at ``w``. Tail mode: each finding gets
    its own member subset and weight; findings with no tail keep the public stack's values."""
    if tails:
        member_ids = list(dict.fromkeys(m for _, ms, _ in tails for m in ms))
        if dup := [f for f, n in __import__("collections").Counter(f for fs, _, _ in tails for f in fs).items() if n > 1]:
            raise SystemExit(f"findings in more than one --tail: {dup}")
        w = max(t[2] for t in tails)
    elif member_ids and w > 0:
        tails = [(TARGETS, member_ids, w)]
    if not 0.0 <= w < 1.0 or (w == 0.0 and not swap):
        raise SystemExit("--w must be in (0, 1), or 0 with --swap-v5")
    nb_path = next(BASE.glob("*.ipynb"))
    nb = json.loads(nb_path.read_text())
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    swap_dss: list[str] = []
    if swap:
        pairs = resolve_members(swap.split(","))[:2]
        swap_dss = [d for d, _ in pairs]
        nb = swap_v5(nb, [w for _, w in pairs], swap_dss)
    members = resolve_members(member_ids) if w > 0 else []
    files = [f for _, f in members]
    if len(set(files)) != len(files):
        raise SystemExit("duplicate member weight names")
    run_leg = (RUN_LEG.replace("__LEG_SRC__", repr(LEG.read_text()))
               .replace("__RUNTIME_S__", str(RUNTIME_S)).replace("__MARGIN_S__", str(LEG_MARGIN_S))
               .replace("__MEMBER_FILES__", repr(files)))
    nb2 = copy.deepcopy(nb)
    start = code_cell("import time as _ens_t\n_ENS_START = _ens_t.time()  # wall clock for our leg's deadline\n")
    tail_map = {f: ([member_ids.index(m) for m in ms], round(float(tw), 4))
                for fs, ms, tw in (tails or []) for f in fs}
    blend = (BLEND.replace("__TAILS__", repr(tail_map))
             .replace("__N_MEMBERS__", str(len(member_ids))))
    tail = ([code_cell(STASH), code_cell(run_leg), code_cell(blend)] if members else [])
    nb2["cells"] = [start] + nb["cells"] + tail
    d = (out_root or REPO / "outputs" / "kernels") / name
    d.mkdir(parents=True, exist_ok=True)
    code_file = f"{name}.ipynb"
    (d / code_file).write_text(json.dumps(nb2, indent=1))
    datasets = [s for s in meta.get("dataset_sources", []) if s]
    for slug in [m[0] for m in members] + swap_dss:
        if slug not in datasets:
            datasets.append(slug)
    meta2 = {**{k: v for k, v in meta.items() if k not in ("id_no",)},
             "id": f"{USER}/{name}", "title": name, "code_file": code_file, "is_private": True,
             "enable_internet": False, "dataset_sources": datasets,
             "docker_image": meta.get("docker_image") or DOCKER}
    (d / "kernel-metadata.json").write_text(json.dumps(meta2, indent=1))
    (d / "ENSEMBLE.json").write_text(json.dumps(
        {"base": nb_path.name, "members": member_ids if members else [], "weights": files, "w": w,
         "swap_v5": swap, "tails": {f: [[member_ids[j] for j in ix], tw] for f, (ix, tw) in tail_map.items()}},
        indent=1))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", default="", help="comma list of node ids / member tags")
    ap.add_argument("--w", type=float, default=0.2, help="global rank weight of our leg (0 = no leg)")
    ap.add_argument("--swap-v5", default=None,
                    help="member tag replacing v5 in both maxspan arms, or 'normal,reverse' tags")
    ap.add_argument("--tail", action="append", default=[],
                    help="per-finding leg, repeatable: 'PF OA|m5b5942352cf,m440e7701125|0.4' or "
                         "'weak5|soupX|0.3' (findings ';'-separated); overrides --members/--w")
    a = ap.parse_args()
    d = build(a.name, [m for m in a.members.split(",") if m], a.w, swap=a.swap_v5,
              tails=[parse_tail(t) for t in a.tail] or None)
    print(f"built {d}\npush (user-approved): kaggle kernels push -p {d}")


if __name__ == "__main__":
    main()
