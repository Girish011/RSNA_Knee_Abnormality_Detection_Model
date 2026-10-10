import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.agent import activity, genome as G, members
from rsna_knee.constants import LABEL_COLS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_ensemble_kernel as BEK  # noqa: E402
import submit_gate as SG  # noqa: E402


@pytest.fixture
def msandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(activity, "ACTIVITY", tmp_path / "activity.jsonl")
    monkeypatch.setattr(members, "MEMBERS", tmp_path / "members.json")
    monkeypatch.setattr(members, "POLICY", tmp_path / "policy.json")
    journal = [{"id": "p1", "genome": dict(G.BASE, labels="dread", epochs=20)}]
    (tmp_path / "journal.json").write_text(json.dumps(journal))
    monkeypatch.setattr(members, "JOURNAL_FILE", tmp_path / "journal.json")
    launched = []

    def fake_launch(job, arms, hours, **kw):
        launched.append((job, arms, hours))
        return {"pod": "podX", "costPerHr": 0.34, "result": f"u/{job}-out"}

    monkeypatch.setattr(members.launcher, "launch", fake_launch)
    return launched


def test_full_cli_drops_fold_hook_and_sets_seed():
    lab, args = members.full_cli(G.BASE, 1337)
    assert lab == "labels_dread.parquet"
    assert "--fold" not in args and "--folds" not in args
    assert "--seed 1337" in args and "--holdout /workspace/bundle/expert_holdout.json" in args


def test_queue_dedupes_and_launches_one_pod_for_all_arms(msandbox, monkeypatch):
    job = members.queue("p1", [42, 1337], ["dread", "blend_v2"])
    assert len(job["arms"]) == 4 and job["approved"] is (job["est_usd"] <= 10)
    job["approved"] = True
    m = members.load(); m["queue"][0]["approved"] = True; members.save(m)
    with pytest.raises(SystemExit):
        members.queue("p1", [42], ["dread"])  # already queued
    two = [{"desiredStatus": "RUNNING"}, {"desiredStatus": "RUNNING"}]
    monkeypatch.setattr(members.runpod, "list_pods", lambda: two)
    assert members.tick(loop_active=False) is True and not msandbox  # MAX_PODS already running: wait
    monkeypatch.setattr(members.runpod, "list_pods", lambda: [])
    assert members.tick(loop_active=True) is True and len(msandbox) == 1   # loop pod (1) + this = 2
    assert len(msandbox[0][1]) == 4                                        # all 4 arms on one pod
    assert all("--fold " not in a[2] for a in msandbox[0][1])
    assert members.load()["active"]["pod"] == "podX"


def _fake_kernel(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": ["pass\n"]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (base / "b.ipynb").write_text(json.dumps(nb))
    (base / "kernel-metadata.json").write_text(json.dumps(
        {"id": "x/y", "id_no": 1, "code_file": "b.ipynb", "dataset_sources": ["a/b", ""]}))
    return base


def _run_tail(kdir, work, leg_src, pub):
    nb = json.loads(next(kdir.glob("*.ipynb")).read_text())
    pub.to_csv(work / "submission.csv", index=False)
    ns = {}
    for c in nb["cells"][:1] + nb["cells"][-3:]:
        src = "".join(c["source"])
        if "OURS 2/3" in src:
            assert repr(BEK.LEGS["v1"].read_text()) in src
            src = src.replace(repr(BEK.LEGS["v1"].read_text()), repr(leg_src))
        src = src.replace("/kaggle/working", str(work))
        exec(compile(src, "cell", "exec"), ns)
    return pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})


def test_ensemble_kernel_blends_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(BEK, "BASE", _fake_kernel(tmp_path))
    kdir = BEK.build("ens-t", ["abc123"], 0.3, out_root=tmp_path / "k")
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert meta["id"] == f"{BEK.USER}/ens-t" and meta["enable_internet"] is False
    assert "girishbose/rsna-knee-rp-n-abc123-out" in meta["dataset_sources"] and "" not in meta["dataset_sources"]
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(40)]
    pub = pd.DataFrame(rng.random((40, 12)), columns=LABEL_COLS)
    pub.insert(0, "StudyInstanceUID", ids)
    work = tmp_path / "w"
    work.mkdir()
    ours = pub.copy()
    ours[LABEL_COLS] = rng.random((40, 12))
    ours.to_csv(tmp_path / "ours_src.csv", index=False)
    good_leg = (f"import os, shutil; shutil.copy({str(tmp_path / 'ours_src.csv')!r}, os.environ['OURS_OUT'])")
    out = _run_tail(kdir, work, good_leg, pub)
    expect = 0.7 * pub[LABEL_COLS].rank(pct=True).to_numpy() + 0.3 * ours[LABEL_COLS].rank(pct=True).to_numpy()
    assert np.allclose(out[LABEL_COLS].to_numpy(), expect)
    out2 = _run_tail(kdir, work, "import sys; sys.exit(3)", pub)  # leg fails -> public stands
    assert np.allclose(out2[LABEL_COLS].to_numpy(), pub[LABEL_COLS].to_numpy())


def test_per_finding_tails_blend_only_listed_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(BEK, "BASE", _fake_kernel(tmp_path))
    tails = [BEK.parse_tail("PF OA|aaa111|0.4"), BEK.parse_tail("Synovitis;Contusion|aaa111,bbb222|0.25")]
    kdir = BEK.build("ens-tail", [], 0.0, out_root=tmp_path / "k", tails=tails)
    ens = json.loads((kdir / "ENSEMBLE.json").read_text())
    assert ens["tails"]["PF OA"] == [["aaa111"], 0.4] and ens["members"] == ["aaa111", "bbb222"]
    rng = np.random.default_rng(1)
    ids = [f"s{i}" for i in range(40)]
    pub = pd.DataFrame(rng.random((40, 12)), columns=LABEL_COLS)
    pub.insert(0, "StudyInstanceUID", ids)
    work = tmp_path / "w"
    work.mkdir()
    m0, m1 = pub.copy(), pub.copy()
    m0[LABEL_COLS] = rng.random((40, 12))
    m1[LABEL_COLS] = rng.random((40, 12))
    allm = pub.copy()
    allm[LABEL_COLS] = (m0[LABEL_COLS].rank(pct=True) + m1[LABEL_COLS].rank(pct=True)) / 2
    for nm, f in (("m0", m0), ("m1", m1), ("all", allm)):
        f.iloc[::-1].to_csv(tmp_path / f"{nm}.csv", index=False)  # leg order != public order
    leg = ("import os, shutil; o = os.environ['OURS_OUT']; "
           f"shutil.copy({str(tmp_path / 'all.csv')!r}, o); "
           f"shutil.copy({str(tmp_path / 'm0.csv')!r}, o[:-4] + '_m0.csv'); "
           f"shutil.copy({str(tmp_path / 'm1.csv')!r}, o[:-4] + '_m1.csv')")
    out = _run_tail(kdir, work, leg, pub)
    r = lambda s: s.rank(pct=True).to_numpy()  # noqa: E731
    assert np.allclose(out["PF OA"], 0.6 * r(pub["PF OA"]) + 0.4 * r(m0["PF OA"]))
    assert np.allclose(out["Synovitis"], 0.75 * r(pub["Synovitis"]) + 0.25 * r(allm["Synovitis"]))
    strong = [c for c in LABEL_COLS if c not in ("PF OA", "Synovitis", "Contusion")]
    assert np.allclose(out[strong].to_numpy(), pub[strong].to_numpy())  # untouched raw values
    with pytest.raises(SystemExit):
        BEK.parse_tail("Nope|aaa111|0.3")
    with pytest.raises(SystemExit):
        BEK.build("x", [], 0.0, out_root=tmp_path / "k", tails=tails + [BEK.parse_tail("PF OA|bbb222|0.2")])
    assert BEK.parse_tail("weak5|aaa111|0.3")[0] == BEK.WEAK5


def test_submit_gate_refuses_duplicates_and_daily_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(BEK, "BASE", _fake_kernel(tmp_path))
    kdir = BEK.build("ens-g", ["abc123"], 0.2, out_root=tmp_path / "k")
    fp = SG.fingerprint(kdir)
    assert SG.check(kdir, [], 0) == []
    assert any("duplicate" in e for e in SG.check(kdir, [{"fingerprint": fp, "ref": 1}], 0))
    assert any("daily cap" in e for e in SG.check(kdir, [], 5))
    kdir2 = BEK.build("ens-g2", ["abc123"], 0.3, out_root=tmp_path / "k")
    assert SG.fingerprint(kdir2) != fp


def test_warm_start_member_args(msandbox):
    lab, args = members.full_cli(G.BASE, 42, init="v5", epochs=6)
    assert "--epochs 6" in args and "--init_model /workspace/input/init/raptor_ft_coatnet_v5_full_swa.pt" in args
    job = members.queue("p1", [42], ["dread"], init="v5", epochs=6)
    assert job["arms"][0]["init"] == "v5" and job["arms"][0]["genome"]["epochs"] == 6
    assert job["arms"][0]["tag"] != members.member_tag(job["arms"][0]["genome"], 42)
    with pytest.raises(SystemExit):
        members.queue("p1", [42], ["dread"], init="nope")


def test_label_weights_member_is_its_own_recipe(msandbox):
    g = dict(G.BASE, label_weights="weak5x2.5")
    assert G.validate(g) == [] or all(e.startswith(("epochs=",)) for e in G.validate(g))
    _, args = members.full_cli(g, 42, init="v5", epochs=6)
    assert "--label_weights 1,1,1,2.5,1,2.5,2.5,1,2.5,1,2.5,1" in args
    assert "--label_weights" not in members.full_cli(G.BASE, 42)[1]
    assert members.member_tag(g, 42) != members.member_tag(G.BASE, 42)
    job = members.queue("p1", [42], ["dread"], init="v5", epochs=6, label_weights="weak5x2.5")
    assert job["arms"][0]["genome"]["label_weights"] == "weak5x2.5"
    with pytest.raises(SystemExit):
        members.queue("p1", [7], ["dread"], label_weights="nope")
    assert "unknown" not in " ".join(G.validate(dict(G.BASE, label_weights="nope")))
    assert any("label_weights" in e for e in G.validate(dict(G.BASE, label_weights="nope")))


def test_swap_v5_rewires_both_arms_and_roots(tmp_path):
    arms = ('ARMS = [\n    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt", "w": 0.6},\n'
            '    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt", "w": 0.1}]\n')
    nb = {"cells": [{"cell_type": "code", "source": "ASSET_ROOTS = ['/kaggle/input/x']\n"},
                    {"cell_type": "code", "source": "names=['raptor_ft_coatnet_v5_full_swa.pt']\n"},
                    {"cell_type": "code", "source": arms}]}
    out = BEK.swap_v5(nb, "raptor_ft_mabc.pt", "girishbose/rsna-knee-rp-m-1-out")
    assert "'/kaggle/input/datasets/girishbose/rsna-knee-rp-m-1-out'" in out["cells"][0]["source"]
    assert out["cells"][2]["source"].count("raptor_ft_mabc.pt") == 2
    assert "coatnet_v5" in out["cells"][1]["source"]  # preflight keeps the public v5 requirement
    with pytest.raises(SystemExit):
        BEK.swap_v5({"cells": [{"cell_type": "code", "source": "x=1"}]}, "a.pt", "u/d")


def test_observe_waits_for_final_snapshot(msandbox, monkeypatch, tmp_path):
    monkeypatch.setattr(members, "RESULTS", tmp_path / "res")
    m = members.load()
    m["running"] = [{"job": "m-x", "pod": "p", "result": "u/m-x-out", "launched": 0, "est_hours": 1,
                   "price_hr": 0.34, "cost_logged": True,
                   "arms": [{"tag": "a1", "genome": G.BASE, "seed": 1}, {"tag": "a2", "genome": G.BASE, "seed": 2}]}]
    monkeypatch.setattr(members.runpod, "get_pod", lambda pid: None)
    status = {"text": "uploading partial after a1"}

    def fake_collect(a):
        (tmp_path / "res" / "m-x").mkdir(parents=True, exist_ok=True)
        (tmp_path / "res" / "m-x" / "STATUS").write_text(status["text"])
        tags = ["a1"] if "partial" in status["text"] else ["a1", "a2"]
        return {t: {"genome": G.BASE, "seed": 1, "gold_dev": 0.9, "holdout_auc": 0.93} for t in tags}

    monkeypatch.setattr(members, "_collect", fake_collect)
    members.observe(m)
    assert m["running"] and not m["done"]                     # partial snapshot: keep waiting
    status["text"] = "uploading (all arms finished)"
    members.observe(m)
    assert not m["running"] and set(m["done"]) == {"a1", "a2"}


def test_swap_v5_refuses_two_different_members():
    nb = {"cells": [{"cell_type": "code", "source": "ASSET_ROOTS = []\n"},
                    {"cell_type": "code", "source": '"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt"'}]}
    with pytest.raises(SystemExit, match="share one model"):
        BEK.swap_v5(nb, ["raptor_ft_ma.pt", "raptor_ft_mb.pt"], ["u/d1", "u/d1"])


def test_eval_job_ships_inits_and_eval_flag(msandbox):
    m = members.load()
    m["done"]["mx"] = {"dataset": "u/rsna-knee-soup-x", "weights": "raptor_ft_mx.pt", "genome": G.BASE}
    members.save(m)
    job = members.queue_eval(["v5", "mx"])
    assert members.load()["queue"][0]["job"] == job["job"] and len(job["arms"]) == 2
    members._launch(members.load(), job)
    (_, arms, _), = msandbox
    assert all("--eval_only --init_model /workspace/input/init/" in a[2] for a in arms)
    assert "--fold " not in arms[0][2]
    with pytest.raises(SystemExit):
        members.queue_eval(["v5", "mx"])  # already queued → refused


def test_bundle_writes_inits(tmp_path, monkeypatch):
    from rsna_knee.agent import launcher
    monkeypatch.setattr(launcher, "REPO", launcher.REPO)
    d = launcher.build_bundle("t-inits", [("a", "labels_dread.parquet",
                                           "--init_model /workspace/input/init/raptor_ft_coatnet_v5_full_swa.pt")],
                              [("raptor_ft_s.pt", "u/soup", "soup/raptor_ft_s.pt")])
    lines = (d / "inits.txt").read_text().splitlines()
    import shutil; shutil.rmtree(d)
    assert "raptor_ft_s.pt|u/soup|soup/raptor_ft_s.pt" in lines and any(l.startswith("raptor_ft_coatnet_v5") for l in lines)


def test_fold_job_five_arms_with_fold_hook(msandbox):
    job = members.queue_folds("p1", "dread", init="v5", epochs=6)
    assert [a["fold"] for a in job["arms"]] == [0, 1, 2, 3, 4]
    members._launch(members.load(), job)
    (_, arms, _), = msandbox
    for k, (_, _, args) in enumerate(arms):
        assert f"--folds 5 --fold {k} " in args and "--init_model" in args and "--epochs 6" in args
        assert "--holdout /workspace/bundle/expert_holdout.json" in args
    with pytest.raises(SystemExit):
        members.queue_folds("p1", "dread", init="v5", epochs=6)


def test_two_member_jobs_run_in_parallel(msandbox, monkeypatch):
    for seed, lw in ((1, None), (2, "weak5x2.5"), (3, "weak5x4")):   # 3 recipes (seed cap is per recipe)
        members.queue("p1", [seed], ["dread"], label_weights=lw)
    monkeypatch.setattr(members.runpod, "list_pods", lambda: [])
    monkeypatch.setattr(members.runpod, "get_pod", lambda pid: {"desiredStatus": "RUNNING"})
    monkeypatch.setattr(members, "_observe_one", lambda m, a: None)
    members.tick(loop_active=False)
    members.tick(loop_active=False)
    members.tick(loop_active=False)   # third must wait: MAX_PODS=2
    m = members.load()
    assert len(msandbox) == 2 and len(m["running"]) == 2 and len(m["queue"]) == 1
    assert m["active"]["job"] == m["running"][0]["job"]


def test_auto_self_train_gate(msandbox):
    job = members.queue_folds("p1", "dread", init="v5", epochs=6)
    m = members.load()
    m["queue"].append({"job": "m-other", "arms": [], "approved": True})
    fail = lambda **kw: {"macro": {"report": .88, "image": .93, "fused": .89}, "passed": False,  # noqa: E731
                         "name": None, "n_fused": 0}
    assert members.auto_self_train(m, job, fail)["passed"] is False
    assert [j["job"] for j in m["queue"]][-1] == "m-other" and len(m["queue"]) == 2
    ok = lambda **kw: {"macro": {"report": .88, "image": .93, "fused": .94}, "passed": True,  # noqa: E731
                       "name": "fused_x", "n_fused": 4349}
    import json as _j
    reg = members.G.label_registry
    members.G.label_registry = lambda: {**reg(), "fused_x": {"file": "labels_fused_x.parquet"}}
    try:
        members.auto_self_train(m, job, ok)
    finally:
        members.G.label_registry = reg
    first = m["queue"][0]
    assert {a["genome"]["labels"] for a in first["arms"]} == {"fused_x"} and len(first["arms"]) == 2
    assert all(a["init"] == "v5" for a in first["arms"])


def test_refold_and_after_dependency(msandbox, monkeypatch):
    job = members.queue_folds("p1", "dread", init="v5", epochs=6)
    m = members.load()
    m["queue"] = []
    for a in job["arms"]:
        m["done"][a["tag"]] = {**a, "job": job["job"]}
    members.save(m)
    fix = members.refold(job["job"], [0, 1])
    assert fix["job"] == job["job"] + "-fix" and [a["fold"] for a in fix["arms"]] == [0, 1]
    assert fix["oof_base"] == job["job"] and len(members.load()["done"]) == 3
    m = members.load()
    m["queue"].append({"job": "m-dep", "arms": [], "approved": True, "after": fix["job"]})
    members.save(m)
    monkeypatch.setattr(members.runpod, "list_pods", lambda: [])
    monkeypatch.setattr(members, "_observe_one", lambda m, a: None)
    members.tick(loop_active=False)          # launches the fix job
    members.tick(loop_active=False)          # m-dep must wait: its dependency is running
    assert [r["job"] for r in members.load()["running"]] == [fix["job"]]
    with pytest.raises(SystemExit):
        members.refold(job["job"], [0, 1])   # arms no longer in done


def test_k16_members_pin_48gb(msandbox, monkeypatch):
    job = members.queue("p1", [42], ["dread"], init="v5", epochs=6, k=16)
    assert "gpu_class" not in job and job["arms"][0]["genome"]["k"] == 16
    job["gpu_class"] = "48"   # still honoured when set
    seen = {}
    monkeypatch.setattr(members.launcher, "launch",
                        lambda j, arms, h, **kw: seen.update(kw, args=arms[0][2]) or
                        {"pod": "p", "costPerHr": 0.33, "result": "u/x"})
    members._launch(members.load(), job)
    assert seen["gpu_class"] == "48" and "--k 16 " in seen["args"] and "--bs 4 " in seen["args"]
    with pytest.raises(SystemExit):
        members.queue("p1", [42], ["dread"], k=32)


def test_policy_blocks_fold_jobs(msandbox, tmp_path):
    (tmp_path / "policy.json").write_text('{"fold_jobs_closed": true, "why": "closed"}')
    with pytest.raises(SystemExit, match="CLOSED"):
        members.queue_folds("p1", "dread")


def test_user_terminated_pod_is_not_relaunched(msandbox, monkeypatch):
    from rsna_knee.agent import loop as L
    job = members.queue("p1", [42], ["dread"])
    m = members.load(); m["queue"] = []
    a = {**job, "pod": "px", "result": "u/r", "launched": __import__("time").time(), "price_hr": 0.3}
    m["running"] = [a]; members.save(m)
    monkeypatch.setattr(members.runpod, "get_pod", lambda pid: {"desiredStatus": "RUNNING"})
    monkeypatch.setattr(L, "_probe_pod_progress", lambda pid: {"reachable": True})
    monkeypatch.setattr(L, "_running_pod_unhealthy", lambda a: None)
    m = members.load(); members.observe(m)
    assert m["running"][0]["seen_alive"] is True
    monkeypatch.setattr(members.runpod, "get_pod", lambda pid: None)   # user terminated it
    monkeypatch.setattr(members, "_collect", lambda a: {})
    members.observe(m)
    assert not any(j["job"] == job["job"] for j in m["queue"])      # not requeued as a phantom


def test_ops_file_edits_apply_at_tick(msandbox, monkeypatch, tmp_path):
    monkeypatch.setattr(members, "OPS", tmp_path / "ops.jsonl")
    job = members.queue("p1", [42], ["dread"])
    m = members.load(); m["queue"] = []
    m["running"] = [{**job, "pod": "old", "launched": 0}, {**job, "pod": "new", "launched": 0}]
    members.save(m)
    members.request_op({"op": "drop_running", "job": job["job"]})
    members.request_op({"op": "requeue", "job_spec": {**job, "retries": 3}})
    monkeypatch.setattr(members.runpod, "list_pods", lambda: [{"desiredStatus": "RUNNING"}] * 2)
    monkeypatch.setattr(members, "_observe_one", lambda m, a: None)
    members.tick(loop_active=False)
    m = members.load()
    assert m["running"] == [] and m["queue"][0]["retries"] == 3 and not (tmp_path / "ops.jsonl").exists()


def test_seed_cap_counts_queued_and_running_arms(msandbox):
    m = members.load()
    g = dict(G.clamp(G.BASE), labels="dread")
    m["done"]["mdone"] = {"tag": "mdone", "genome": g, "seed": 7, "init": None}
    members.save(m)
    job = members.queue("p1", [1, 2, 3], ["dread"])   # 1 done -> room for 1 more
    assert len(job["arms"]) == 1
    with pytest.raises(SystemExit, match="cap"):
        members.queue("p1", [4], ["dread"])           # 1 done + 1 queued = cap
    members.queue("p1", [4], ["dread"], label_weights="weak5x2.5")  # other recipe: allowed


def test_strategist_gpu_jobs_are_proposals_under_policy(msandbox, tmp_path):
    from rsna_knee.agent import strategist
    assert not strategist.gpu_propose_only()
    members.POLICY.write_text(json.dumps({"strategist_gpu": "propose"}))
    assert strategist.gpu_propose_only()
    job = members.queue("p1", [1], ["dread"], auto_approve=False)
    assert job["approved"] is False
    members.tick(loop_active=False)
    assert members.load()["running"] == []   # unapproved proposal never launches


def test_swap_arms_and_coat_shift(tmp_path):
    cells = ['ASSET_ROOTS = [\n',
             'ARMS = [{"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt"},\n'
             ' {"name": "x", "file": "raptor_ft_coatnet_v10_full.pt"}]\n',
             "_coatnet_weight = {l: 0.60 for l in L}\n_coatnet_weight.update({'ACL': 0.75, 'X': 1.0})\n"
             "_blend_output = _blend_transformer.copy()\n"]
    nb = {"cells": [{"cell_type": "code", "source": c} for c in cells]}
    nb2, dss = BEK.swap_arms(nb, {"v8": "abc123"})
    assert dss == ["girishbose/rsna-knee-rp-n-abc123-out"]
    assert '"file": "raptor_ft_abc123.pt"' in nb2["cells"][1]["source"]
    assert "raptor_ft_coatnet_v10_full.pt" in nb2["cells"][1]["source"]
    nb3 = BEK.shift_coat_weight(nb, 0.15)
    ns = {"L": ["ACL", "Y", "X"], "_blend_transformer": pd.DataFrame()}
    exec(nb3["cells"][2]["source"], ns)
    assert ns["_coatnet_weight"] == {"ACL": 0.9, "Y": 0.75, "X": 1.0}


def test_swap_rad_heads_repins_sha_and_paths():
    old = "a" * 64
    cell = (f"_RAD_E13_HEADS_SHA256 = '{old}'\nby_name = {{\n    'v52_e11_heads.pt': [\n"
            "        '/kaggle/input/x/v52_e11_heads.pt',\n    ],\n}\n")
    nb = {"cells": [{"cell_type": "code", "source": cell}]}
    nb2, dss = BEK.swap_rad_heads(nb, {"e13": ("me/rad-ours", "b" * 64)})
    t = nb2["cells"][0]["source"]
    assert dss == ["me/rad-ours"] and "b" * 64 in t and old not in t
    assert t.index("/kaggle/input/datasets/me/rad-ours/v52_e11_heads.pt") < t.index("/kaggle/input/x/")
    with pytest.raises(SystemExit):
        BEK.swap_rad_heads(nb, {"ref": ("me/rad-ours", "b" * 64)})
