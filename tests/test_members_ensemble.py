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
            assert repr(BEK.LEG.read_text()) in src
            src = src.replace(repr(BEK.LEG.read_text()), repr(leg_src))
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
    m["active"] = {"job": "m-x", "pod": "p", "result": "u/m-x-out", "launched": 0, "est_hours": 1,
                   "price_hr": 0.34, "cost_logged": True,
                   "arms": [{"tag": "a1", "genome": G.BASE, "seed": 1}, {"tag": "a2", "genome": G.BASE, "seed": 2}]}
    monkeypatch.setattr(members.runpod, "get_pod", lambda pid: None)
    status = {"text": "uploading partial after a1"}

    def fake_collect(a):
        (tmp_path / "res" / "m-x").mkdir(parents=True, exist_ok=True)
        (tmp_path / "res" / "m-x" / "STATUS").write_text(status["text"])
        tags = ["a1"] if "partial" in status["text"] else ["a1", "a2"]
        return {t: {"genome": G.BASE, "seed": 1, "gold_dev": 0.9, "holdout_auc": 0.93} for t in tags}

    monkeypatch.setattr(members, "_collect", fake_collect)
    members.observe(m)
    assert m["active"] is not None and not m["done"]          # partial snapshot: keep waiting
    status["text"] = "uploading (all arms finished)"
    members.observe(m)
    assert m["active"] is None and set(m["done"]) == {"a1", "a2"}


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
    for seeds in ([1], [2], [3]):
        members.queue("p1", seeds, ["dread"])
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
