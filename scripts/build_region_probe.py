#!/usr/bin/env python3
"""Submit kernel = a base notebook + our REGION SPECIALISTS (runpod/region_spec.py) on chosen findings.

    # swap probe: the 0.950 notebook with its Lateral OA column replaced by the tibiofemoral specialist
    python scripts/build_region_probe.py --name v5-rs-latoa --base outputs/kernels/pub/rsna-versia-5 \
        --dataset girishbose/rsna-knee-rp-rs-f0-v2-out --side raptor_rs_side_last.pt,raptor_rs_side.pt \
        --spec "Lateral OA|raptor_rs_tf-fused_last.pt,raptor_rs_tf-fused.pt|1.0"
    # ALONE probe, CPU notebook, no GPU quota: every column 0.5 except the specialist's finding.
    # Leaderboard reading: that finding's AUC = 12 x score - 5.5 (+-0.006). Any failure = ERROR (costs no slot).
    python scripts/build_region_probe.py --name cpu-latoa-tf --alone --dataset ... --side ... \
        --spec "Lateral OA|raptor_rs_rs3tf0.pt+raptor_rs_rs3tf1.pt|1.0"
    # parity check on a free CPU kernel: the leg on TRAIN studies against the predictions the training run stored
    # (raptor_holdout_<tag>.npz, or with --parity_oof FILE --parity_n N the out-of-fold ones, raptor_oof_<tag>_fold*.npz)
    python scripts/build_region_probe.py --name rs-parity --parity tf-fused,pf-fused --dataset ... --side ... --spec ...

--spec "finding[;finding]|checkpoints|weight" (repeatable): the checkpoints must be of ONE region. "a.pt,b.pt" = one
round (one model call, mean of the two); "a.pt+b.pt" = two ROUNDS: round 0 is scored on every study first, a later
round only if it is projected to end before the time limit, so the result is always the mean of whole rounds (on a
CPU a round of one tibiofemoral checkpoint is ~2.7 h for 1,322 studies). The mean probability is rank-blended into
the base's column at `weight` (1.0 = the base column is replaced). Only the variants that hold a listed finding are
scored (Lateral OA = the lateral compartment alone). Test time = training time by construction: the cell runs the
dense512 cache builder (`outputs/kernels/dense512_cache/dense512_cache_template.py`, only its paths changed), then
`region_spec.py --predict` (same tiles, same side rule: Laterality tag > side model > sign of P.x) - in CHUNKS of
`--chunk` studies (corpus -> side -> specialists -> delete): one corpus for the ~1,322 hidden studies is ~28 GB and
killed `v5-rs-latoa` on 13 Oct. With a base, any failure leaves the base's submission.csv untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "girishbose"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TEMPLATE = REPO / "outputs" / "kernels" / "dense512_cache" / "dense512_cache_template.py"
REGION = REPO / "runpod" / "region_spec.py"

STASH = '''# ==== REGION 1/3: stash the base output (fallback if anything below fails) ====
import shutil as _rs_shutil
_RS_SUB = "/kaggle/working/submission.csv"; _RS_PUB = "/kaggle/working/_rs_base.csv"
_rs_shutil.copy2(_RS_SUB, _RS_PUB)
print("[region 1/3] stashed the base output", flush=True)
'''

LEG = '''# ==== REGION 2/3: native-resolution corpus of the studies in chunks, then the specialists ====
import glob as _rs_glob, json as _rs_json, os as _rs_os, shutil as _rs_sh, subprocess as _rs_sp, sys as _rs_sys, time as _rs_time
import numpy as _rs_np, pandas as _rs_pd
_RS_SPEC = __SPEC__
_RS_LAB = __LAB__
_RS_STRICT = __STRICT__     # True (alone / parity): a failure stops the notebook; False: the base stands
_RS_W = "/kaggle/working/_rs"; _RS_C = "/tmp/rs_corpus"; _RS_CSV = "/kaggle/working/_rs_pred.csv"
_RS_NB_T0 = globals().get("_RS_NB_T0", _rs_time.time())     # notebook start
_rs_ok = False


def _rs_find(name):   # weights only: never walk the competition's DICOM tree
    for d in sorted(_rs_glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower() or "abnormality-detection" in d:
            continue
        hits = _rs_glob.glob(d + "**/" + name, recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(name + " not found under /kaggle/input")


def _rs_run(cmd, env=None, tail=12):
    r = _rs_sp.run(cmd, env=dict(_rs_os.environ, **(env or {})), stdout=_rs_sp.PIPE, stderr=_rs_sp.STDOUT, text=True)
    print("\\n".join(r.stdout.splitlines()[-tail:]), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"{_rs_os.path.basename(cmd[1])} rc={r.returncode}")


def _rs_test_ids():
    if __IDS__:
        return [str(u) for u in __IDS__]
    if _rs_os.path.exists("/kaggle/working/submission.csv"):     # the base's own list of studies
        return _rs_pd.read_csv("/kaggle/working/submission.csv", dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
    for d, _, f in _rs_os.walk("/kaggle/input"):
        if "test.csv" in f and "test_series.csv" in f:
            return _rs_pd.read_csv(d + "/test.csv", dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
    raise RuntimeError("test.csv not found")


def _rs_score(ids, stop_at):
    # Rounds of checkpoints over chunks of studies. Round 0 must fit before `stop_at` (seconds of notebook time,
    # projected from its first chunk); a later round is started only if it is projected to end before it.
    regs, ch = _RS_SPEC["regions"], _RS_SPEC["chunk"]
    chunks = [ids[i:i + ch] for i in range(0, len(ids), ch)]
    base = [_rs_sys.executable, _RS_W + "/region_spec.py", "--labels", "none", "--region", "tf", "--data_root", _RS_C,
            "--n_shards", "1", "--out", _RS_W, "--workers", "3", "--bs", "6"]
    env = {"RS_CORPUS": _RS_C, "RS_SPLIT": __SPLIT__, "RS_IDS": _RS_W + "/ids.json"}
    tot, cnt, info = {}, {}, {"rounds": 0, "errors": 0, "empty": 0}
    for rd in range(max(len(r["rounds"]) for r in regs)):
        cost = sum(len(r["rounds"][rd]) for r in regs if len(r["rounds"]) > rd)
        if rd > 0:
            need = 1.08 * t_r0 * cost / cost0
            if _rs_time.time() - _RS_NB_T0 + need > stop_at:
                print(f"[region 2/3] round {rd} NOT started: needs ~{need / 3600:.2f} h, notebook at "
                      f"{(_rs_time.time() - _RS_NB_T0) / 3600:.2f} h of {stop_at / 3600:.1f} h", flush=True)
                break
        part, t_rd = {}, _rs_time.time()
        for ci, chunk in enumerate(chunks):
            tc = _rs_time.time()
            _rs_sh.rmtree(_RS_C, ignore_errors=True); _rs_os.makedirs(_RS_C)
            _rs_json.dump(chunk, open(_RS_W + "/ids.json", "w"))
            _rs_run([_rs_sys.executable, _RS_W + "/dense512_build.py"], env, tail=1)
            meta = _rs_json.load(open(_RS_C + "/meta_s0.json"))
            assert meta["done"] == len(chunk), f"corpus built {meta['done']} of {len(chunk)} studies"
            if rd == 0:
                info["errors"] += len(meta["errors"]); info["empty"] += meta["empty_studies"]
            side = []
            if _RS_SPEC["side"]:
                sf = _RS_W + f"/side_{ci}.npz"       # one side call a chunk, kept for the later rounds
                if not _rs_os.path.exists(sf):
                    _rs_run(base + ["--predict", ",".join(_rs_find(c) for c in _RS_SPEC["side"]), "--tag", "side",
                                    "--eval_passes", "1"], tail=2)
                    _rs_os.replace(_RS_W + "/raptor_side_side.npz", sf)
                    zs = _rs_np.load(sf, allow_pickle=True)
                    info.setdefault("side", {}).update(zip(zs["ids"].astype(str).tolist(), zs["p_left"].astype(float).tolist()))
                side = ["--side_file", sf]
            for i, r in enumerate(regs):
                if len(r["rounds"]) <= rd:
                    continue
                _rs_run(base + ["--predict", ",".join(_rs_find(c) for c in r["rounds"][rd]), "--tag", f"r{i}",
                                "--eval_passes", str(_RS_SPEC["passes"]), "--only_findings", ";".join(r["findings"])]
                        + _RS_SPEC.get("extra", []) + side, tail=2)
                z = _rs_np.load(_RS_W + f"/raptor_rspred_r{i}.npz", allow_pickle=True)
                assert z["ids"].astype(str).tolist() == chunk, "the specialists' studies are not the chunk's"
                for f in r["findings"]:
                    part.setdefault(f, []).append(_rs_pd.Series(z["pred"][:, _RS_LAB.index(f)].astype(float), index=chunk) * len(r["rounds"][rd]))
                    if ci == 0:
                        cnt[f] = cnt.get(f, 0) + len(r["rounds"][rd])
            _rs_sh.rmtree(_RS_C, ignore_errors=True)
            el = _rs_time.time() - t_rd
            print(f"[region 2/3] round {rd} chunk {ci + 1}/{len(chunks)} | {len(chunk)} studies in {_rs_time.time() - tc:.0f}s | "
                  f"round eta {el / (ci + 1) * (len(chunks) - ci - 1) / 60:.0f} min | notebook at {(_rs_time.time() - _RS_NB_T0) / 3600:.2f} h", flush=True)
            if rd == 0 and ci == 0 and len(chunks) > 1:
                proj = (_rs_time.time() - tc) * len(ids) / len(chunk)
                if tc - _RS_NB_T0 + proj > stop_at:
                    raise RuntimeError(f"no time left: round 0 needs ~{proj / 3600:.2f} h from {(tc - _RS_NB_T0) / 3600:.2f} h, limit {stop_at / 3600:.1f} h")
        if rd == 0:
            t_r0, cost0 = _rs_time.time() - t_rd, cost
        for f, ps in part.items():
            tot[f] = tot.get(f, 0.0) + _rs_pd.concat(ps)
        info["rounds"] = rd + 1
    df = _rs_pd.DataFrame({f: tot[f] / cnt[f] for f in tot}).reindex(ids)
    return df, info


try:
    _rs_t0 = _rs_time.time()
    _rs_ids = _rs_test_ids()
    assert len(set(_rs_ids)) == len(_rs_ids) > 0
    print(f"[region 2/3] {len(_rs_ids)} studies | chunks of {_RS_SPEC['chunk']} | notebook at {(_rs_t0 - _RS_NB_T0) / 3600:.2f} h | "
          f"limit {__LIMIT_S__ / 3600:.1f} h", flush=True)
    _rs_os.makedirs(_RS_W, exist_ok=True)
    open(_RS_W + "/dense512_build.py", "w").write(__BUILDER_SRC__)
    open(_RS_W + "/region_spec.py", "w").write(__REGION_SRC__)
    _rs_df, _rs_info = _rs_score(_rs_ids, __LIMIT_S__)
    _rs_df.index.name = "StudyInstanceUID"; _rs_df.to_csv(_RS_CSV)
    assert _rs_df.notna().all().all(), "a specialist left NaN"
    _rs_ok = True
    print(f"[region 2/3] scored {len(_rs_df)} studies | findings {list(_rs_df.columns)} | rounds {_rs_info['rounds']} | corpus errors "
          f"{_rs_info['errors']}, empty {_rs_info['empty']} | {(_rs_time.time() - _rs_t0) / 60:.1f} min", flush=True)
except Exception as _rs_e:
    print(f"[region 2/3] FAILED ({type(_rs_e).__name__}: {_rs_e})", flush=True)
    if _RS_STRICT:
        raise
for _rs_p in _rs_glob.glob(_RS_W + "/*.py") + _rs_glob.glob(_RS_W + "/__pycache__/*") + _rs_glob.glob(_RS_W + "/side_*.npz") + _rs_glob.glob(_RS_W + "/raptor_rspred_*.npz"):
    _rs_os.remove(_rs_p)
'''

ALONE = '''# ==== REGION 3/3 (alone): every column 0.5 except the specialists' findings ====
assert _rs_ok
_rs_o = _rs_pd.read_csv(_RS_CSV, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
_rs_out = _rs_pd.DataFrame({"StudyInstanceUID": _rs_ids})
for _rs_l in _RS_LAB:
    _rs_out[_rs_l] = 0.5
for _rs_f in _rs_o.columns:
    _rs_c = _rs_o[_rs_f].reindex(_rs_ids)
    assert _rs_c.notna().all(), f"{_rs_f}: studies missing"
    assert len(_rs_ids) <= 10 or float(_rs_c.std()) > 1e-9, f"{_rs_f}: constant column"
    _rs_out[_rs_f] = _rs_c.to_numpy()
assert _rs_np.isfinite(_rs_out[_RS_LAB].to_numpy(float)).all() and _rs_out.columns.tolist() == ["StudyInstanceUID", *_RS_LAB]
_rs_out.to_csv("/kaggle/working/submission.csv", index=False)
print(f"[region 3/3] alone: {list(_rs_o.columns)} from the specialists ({_rs_info['rounds']} round(s)), the other columns 0.5 | "
      f"submission.csv ok {_rs_out.shape} | notebook at {(_rs_time.time() - _RS_NB_T0) / 3600:.2f} h", flush=True)
'''

BLEND = '''# ==== REGION 3/3: per-finding rank blend into the base; any problem -> the base stands ====
_RS_WEIGHTS = __WEIGHTS__      # finding -> weight of the specialists' column (1.0 = replace the base column)
_rs_pub = _rs_pd.read_csv(_RS_PUB, dtype={"StudyInstanceUID": str})
try:
    if not _rs_ok:
        raise RuntimeError("the region leg failed")
    _rs_o = _rs_pd.read_csv(_RS_CSV, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(_rs_pub["StudyInstanceUID"])
    _rs_out = _rs_pub.copy(); _rs_log = []
    for _rs_f, _rs_w in _RS_WEIGHTS.items():
        _rs_c = _rs_o[_rs_f]
        if _rs_c.isna().any():
            raise RuntimeError(f"{_rs_f}: studies missing from the specialists' output")
        if len(_rs_pub) > 10 and float(_rs_c.std()) < 1e-9:
            raise RuntimeError(f"{_rs_f}: constant column")
        _rs_a = _rs_pub[_rs_f].rank(method="average", pct=True).to_numpy()
        _rs_b = _rs_c.rank(method="average", pct=True).to_numpy()
        _rs_out[_rs_f] = (1 - _rs_w) * _rs_a + _rs_w * _rs_b
        _rs_log.append(f"{_rs_f}:w{_rs_w}/corr{float(_rs_np.corrcoef(_rs_a, _rs_b)[0, 1]):.3f}")
    assert _rs_np.isfinite(_rs_out[_RS_LAB].to_numpy(float)).all()
    _rs_out.to_csv(_RS_SUB, index=False)
    print("[region 3/3] blended " + " | ".join(_rs_log), flush=True)
except Exception as _rs_e:
    _rs_shutil.copy2(_RS_PUB, _RS_SUB)
    print(f"[region 3/3] NOT blended ({type(_rs_e).__name__}: {_rs_e}); the base stands", flush=True)
_rs_final = _rs_pd.read_csv(_RS_SUB)
assert _rs_final.columns.tolist() == ["StudyInstanceUID", *_RS_LAB] and len(_rs_final) == len(_rs_pub)
print(f"[region] submission.csv ok {_rs_final.shape}", flush=True)
'''

PARITY = '''
# ==== parity: the leg's predictions against the ones the training run stored for the same studies ====
from scipy.stats import spearmanr as _rs_sr
assert _rs_ok, "the region leg failed"
_rs_got = _rs_pd.read_csv(_RS_CSV, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
for _rs_tag, _rs_r in zip(__TAGS__, _RS_SPEC["regions"]):
    _rs_z = _rs_np.load(_rs_find(f"raptor___REF___{_rs_tag}*.npz"), allow_pickle=True)
    _rs_m = {u: i for i, u in enumerate(_rs_z["ids"].astype(str))}
    _rs_u = [u for u in _rs_got.index if u in _rs_m]
    for _rs_f in _rs_r["findings"]:
        _rs_a = _rs_got.loc[_rs_u, _rs_f].to_numpy(); _rs_b = _rs_z["pred"][[_rs_m[u] for u in _rs_u], _RS_LAB.index(_rs_f)]
        print(f"PARITY {_rs_tag} | {_rs_f}: n {len(_rs_u)} of {len(_rs_got)} | max abs diff {float(_rs_np.abs(_rs_a - _rs_b).max()):.4f} | "
              f"mean abs diff {float(_rs_np.abs(_rs_a - _rs_b).mean()):.4f} | Spearman {_rs_sr(_rs_a, _rs_b).correlation:.4f}", flush=True)
if _RS_SPEC["side"]:      # against the side file the training runs used (same side model, all train studies)
    _rs_s = _rs_np.load(_rs_find("raptor_side_side.npz"), allow_pickle=True)
    _rs_m = dict(zip(_rs_s["ids"].astype(str), _rs_s["p_left"].astype(float)))
    _rs_u = [u for u in _rs_info["side"] if u in _rs_m]
    _rs_a = _rs_np.array([_rs_info["side"][u] for u in _rs_u]); _rs_b = _rs_np.array([_rs_m[u] for u in _rs_u])
    print(f"PARITY side: n {len(_rs_u)} | max abs diff {float(_rs_np.abs(_rs_a - _rs_b).max()):.4f} | same side call "
          f"{int(((_rs_a >= 0.5) == (_rs_b >= 0.5)).sum())} of {len(_rs_u)}", flush=True)
'''

BENCH = '''
# ==== CPU bench: the first region's round 0 on 16 studies under several settings (time, and numbers against the plain call) ====
from scipy.stats import spearmanr as _rs_sr2
_rs_os.makedirs(_RS_W, exist_ok=True); _rs_sh.rmtree(_RS_C, ignore_errors=True); _rs_os.makedirs(_RS_C)
open(_RS_W + "/dense512_build.py", "w").write(__BUILDER_SRC__); open(_RS_W + "/region_spec.py", "w").write(__REGION_SRC__)
_rs_json.dump(_rs_ids[:16], open(_RS_W + "/ids.json", "w"))
_rs_run([_rs_sys.executable, _RS_W + "/dense512_build.py"], {"RS_CORPUS": _RS_C, "RS_SPLIT": __SPLIT__, "RS_IDS": _RS_W + "/ids.json"}, tail=1)
_rs_b = {}; _rs_k0 = _RS_LAB.index(_RS_SPEC["regions"][0]["findings"][0])
for _rs_k, _rs_x in __BENCH__:
    _rs_t = _rs_time.time()
    _rs_run([_rs_sys.executable, _RS_W + "/region_spec.py", "--labels", "none", "--region", "tf", "--data_root", _RS_C, "--n_shards", "1",
             "--out", _RS_W, "--workers", "3", "--bs", "6", "--predict", ",".join(_rs_find(c) for c in _RS_SPEC["regions"][0]["rounds"][0]),
             "--tag", "bench", "--eval_passes", "1", "--only_findings", ";".join(_RS_SPEC["regions"][0]["findings"])] + _rs_x, tail=3)
    _rs_b[_rs_k] = (_rs_time.time() - _rs_t, _rs_np.load(_RS_W + "/raptor_rspred_bench.npz")["pred"][:, _rs_k0])
    print(f"BENCH {_rs_k}: 16 studies in {_rs_b[_rs_k][0]:.0f}s | max abs diff to plain {float(_rs_np.abs(_rs_b[_rs_k][1] - _rs_b['plain'][1]).max()):.4f} | "
          f"Spearman {_rs_sr2(_rs_b[_rs_k][1], _rs_b['plain'][1]).correlation:.4f}", flush=True)
_rs_sh.rmtree(_RS_C, ignore_errors=True); _rs_sh.rmtree(_RS_W, ignore_errors=True)
'''


def builder_src(split: str) -> str:
    """The dense512 cache builder with only its paths changed (one shard, RS_CORPUS, test or train series, optional id list)."""
    s = TEMPLATE.read_text()
    edits = [("SHARD = __SHARD__", "SHARD = 0"), ("N_SHARDS = __N_SHARDS__", "N_SHARDS = 1"),
             ('OUT = "/kaggle/working"', 'OUT = os.environ["RS_CORPUS"]\nSPLIT = os.environ.get("RS_SPLIT", "test")'),
             ('    sdir = root + "/train_series"\n', '    sdir = root + f"/{SPLIT}_series"\n'),
             ('        sdir = root + "/train_images"\n', '        sdir = root + f"/{SPLIT}_images"\n'),
             ('    ids = pd.read_csv(root + "/train.csv")["StudyInstanceUID"].astype(str).tolist()\n',
              '    ids = pd.read_csv(root + f"/{SPLIT}.csv")["StudyInstanceUID"].astype(str).tolist()\n'
              '    if os.environ.get("RS_IDS"):\n        ids = json.load(open(os.environ["RS_IDS"]))\n'),
             ('    ser = pd.read_csv(root + "/train_series.csv")\n', '    ser = pd.read_csv(root + f"/{SPLIT}_series.csv")\n')]
    for old, new in edits:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    compile(s, "dense512_build.py", "exec")
    return s


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default=None, help="folder with the base notebook + kernel-metadata.json")
    ap.add_argument("--spec", action="append", required=True,
                    help="'finding[;finding]|ckpt[,ckpt][+ckpt...]|weight' (repeatable); '+' separates rounds")
    ap.add_argument("--side", default="", help="side-model checkpoints (comma list); empty = Laterality tag / sign of P.x only")
    ap.add_argument("--dataset", action="append", default=[], help="dataset holding the checkpoints (repeatable)")
    ap.add_argument("--kernel", action="append", default=[], help="kernel whose output holds checkpoints (repeatable)")
    ap.add_argument("--alone", action="store_true", help="CPU notebook without a base: 0.5 in every column but the --spec findings")
    ap.add_argument("--passes", type=int, default=None, help="tile passes per checkpoint (default 2; 1 with --alone / --parity_oof)")
    ap.add_argument("--chunk", type=int, default=100, help="studies per corpus chunk")
    ap.add_argument("--limit_h", type=float, default=None, help="notebook hours by which a round must end (default 8.6; 8.4 alone)")
    ap.add_argument("--parity", default=None, help="comma list of training tags, one per --spec: CPU script kernel that scores "
                                                   "TRAIN studies and compares with the predictions stored by the training run")
    ap.add_argument("--parity_oof", default=None, help="local raptor_oof_<tag>_fold<k>.npz: take the studies from it "
                                                       "(default: the 29 expert-holdout studies, raptor_holdout_<tag>.npz)")
    ap.add_argument("--parity_n", type=int, default=300)
    ap.add_argument("--bench", action="store_true", help="--parity: also time CPU settings on 16 studies (threads, workers, tile count)")
    ap.add_argument("--extra", default="", help="extra region_spec.py --predict arguments for the specialists' calls, "
                                                "e.g. '--threads 4 --eval_tiles train' (CPU cost)")
    a = ap.parse_args()
    assert not (a.alone and (a.base or a.parity)), "--alone has no base and is not a parity kernel"
    regions, weights = [], {}
    for sp in a.spec:
        fs, cks, w = sp.split("|")
        finds = [f.strip() for f in fs.split(";") if f.strip()]
        assert all(f in LAB for f in finds), finds
        assert not set(finds) & set(weights), "a finding appears in two --spec"
        rounds = [[c for c in rd.split(",") if c] for rd in cks.split("+")]
        assert rounds and all(rounds), cks
        regions.append({"findings": finds, "rounds": rounds})
        weights.update({f: float(w) for f in finds})
        assert 0.0 < float(w) <= 1.0 and (not a.alone or float(w) == 1.0)
    cheap = a.alone or bool(a.parity_oof)
    spec = {"side": [c for c in a.side.split(",") if c], "regions": regions, "chunk": a.chunk,
            "passes": a.passes or (1 if cheap else 2), "extra": a.extra.split()}
    limit_h = a.limit_h or (8.4 if a.alone else 8.6)
    ids: list[str] = []
    if a.parity and a.parity_oof:
        import numpy as np
        z = np.load(a.parity_oof, allow_pickle=True)["ids"].astype(str)
        ids = z[np.linspace(0, len(z) - 1, min(a.parity_n, len(z))).round().astype(int)].tolist()
    elif a.parity:
        ids = json.loads((REPO / "agent" / "protocol" / "expert_holdout.json").read_text())
    fill = lambda t: (t.replace("__SPEC__", repr(spec)).replace("__LAB__", repr(LAB)).replace("__BUILDER_SRC__", repr(builder_src("x")))
                      .replace("__REGION_SRC__", repr(REGION.read_text())).replace("__SPLIT__", repr("train" if a.parity else "test"))
                      .replace("__IDS__", repr(ids)).replace("__LIMIT_S__", repr(limit_h * 3600))
                      .replace("__STRICT__", repr(bool(a.alone or a.parity))))
    leg = fill(LEG)
    d = REPO / "outputs" / "kernels" / a.name
    d.mkdir(parents=True, exist_ok=True)
    base_meta = json.loads((REPO / "outputs/kernels/pub/rsna-versia-5/kernel-metadata.json").read_text())
    cpu_meta = {"id": f"{USER}/{a.name}", "title": a.name, "language": "python", "is_private": True, "enable_gpu": False,
                "enable_tpu": False, "enable_internet": False, "dataset_sources": a.dataset, "kernel_sources": a.kernel,
                "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
                "docker_image": base_meta.get("docker_image")}
    if a.parity:
        tags = a.parity.split(",")
        assert len(tags) == len(regions)
        src = leg + PARITY.replace("__TAGS__", repr(tags)).replace("__REF__", "oof" if a.parity_oof else "holdout")
        if a.bench:
            src += fill(BENCH).replace("__BENCH__", repr([("plain", []), ("threads4", ["--threads", "4"]),
                                                          ("threads4-workers1", ["--threads", "4", "--workers", "1"]),
                                                          ("threads2-workers2", ["--threads", "2", "--workers", "2"]),
                                                          ("tiles-train", ["--eval_tiles", "train"])]))
        compile(src, a.name, "exec")
        code = a.name.replace("-", "_") + ".py"
        (d / code).write_text(src)
        meta = dict(cpu_meta, code_file=code, kernel_type="script")
    elif a.alone:
        for c in (leg, ALONE):
            compile(c, a.name, "exec")
        ref = json.loads(next((REPO / "outputs/kernels/pub/rsna-versia-5").glob("*.ipynb")).read_text())
        nb = {"cells": [code_cell(leg), code_cell(ALONE)], "metadata": {k: ref["metadata"][k] for k in ("kernelspec", "language_info")},
              "nbformat": 4, "nbformat_minor": 4}
        code = f"{a.name}.ipynb"
        (d / code).write_text(json.dumps(nb, indent=1))
        meta = dict(cpu_meta, code_file=code, kernel_type="notebook")
    else:
        base = Path(a.base).resolve()
        nb_path = next(base.glob("*.ipynb"))
        nb = json.loads(nb_path.read_text())
        blend = BLEND.replace("__WEIGHTS__", repr(weights))
        for c in (STASH, leg, blend):
            compile(c, a.name, "exec")
        start = code_cell("import time as _rs_clock\n_RS_NB_T0 = _rs_clock.time()  # notebook start, for the region part's time guard\n")
        nb["cells"] = [start] + nb["cells"] + [code_cell(STASH), code_cell(leg), code_cell(blend)]
        code = f"{a.name}.ipynb"
        (d / code).write_text(json.dumps(nb, indent=1))
        meta = json.loads((base / "kernel-metadata.json").read_text())
        meta.pop("id_no", None)
        meta.update(id=f"{USER}/{a.name}", title=a.name, code_file=code, is_private=True, enable_internet=False,
                    dataset_sources=[s for s in meta.get("dataset_sources", []) if s] + [s for s in a.dataset if s not in meta.get("dataset_sources", [])],
                    kernel_sources=[s for s in meta.get("kernel_sources", []) if s] + a.kernel)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    (d / "REGION.json").write_text(json.dumps({"base": a.base, "alone": a.alone, "spec": spec, "weights": weights, "parity": a.parity,
                                               "limit_h": limit_h, "n_ids": len(ids)}, indent=1))
    print(f"wrote {d} | {len(regions)} region(s), rounds {[len(r['rounds']) for r in regions]} | passes {spec['passes']} | "
          f"weights {weights} | push: kaggle kernels push -p {d}")


if __name__ == "__main__":
    main()
