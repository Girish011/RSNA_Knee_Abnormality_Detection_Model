import json
import random

import numpy as np
import pandas as pd
import pytest

from rsna_knee.agent import activity, genome as G, loop, planner
from rsna_knee.agent.journal import ROOT_ID, Journal
from rsna_knee.constants import LABEL_COLS


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(activity, "ACTIVITY", tmp_path / "activity.jsonl")
    monkeypatch.setattr(loop, "JOURNAL", tmp_path / "journal.json")
    monkeypatch.setattr(loop, "STATE", tmp_path / "loop_state.json")
    monkeypatch.setattr(loop, "HOLDOUT", tmp_path / "holdout.json")
    monkeypatch.setattr(loop, "RESULTS", tmp_path / "results")
    ids = np.array([f"s{i}" for i in range(200)])
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.integers(0, 2, (200, 12)), index=ids, columns=LABEL_COLS)
    monkeypatch.setattr(loop, "reference", lambda: (ref, pd.DataFrame(True, index=ids, columns=LABEL_COLS)))
    launched = []

    def fake_launch(job, arms, hours):
        launched.append((job, arms))
        return {"pod": f"pod-{len(launched)}", "costPerHr": 0.34, "result": f"u/{job}-out"}

    monkeypatch.setattr(loop.launcher, "launch", fake_launch)
    monkeypatch.setattr(loop.runpod, "get_pod", lambda pid: (_ for _ in ()).throw(RuntimeError("404")))
    monkeypatch.setattr(loop.runpod, "terminate_pod", lambda pid: None)
    monkeypatch.setattr(loop.runpod, "list_pods", lambda: [])

    def fake_download(slug, dest, tags):
        if "broken" in slug:
            return True
        nid = tags[0]
        (dest / nid).mkdir(parents=True, exist_ok=True)
        noise = rng.normal(0, 1.0, (200, 12))
        np.savez(dest / nid / f"raptor_oof_{nid}_fold0.npz", pred=ref.to_numpy() + noise, ids=ids)
        (dest / nid / f"raptor_ft_{nid}.json").write_text(json.dumps({"best_gold_auc": 0.91}))
        return True

    monkeypatch.setattr(loop, "download_results", fake_download)
    return launched


def test_genome_base_valid_and_cli():
    assert G.validate(G.BASE) == []
    labels, args = G.to_cli(G.BASE, "/b")
    assert labels == "labels_dread.parquet"
    assert "--fold 0" in args and "--holdout /b/expert_holdout.json" in args
    bad = dict(G.BASE, epochs=40, corpus="dense80")
    errs = G.validate(bad)
    assert any("epochs" in e for e in errs) and any("dense80" in e for e in errs)


def test_planner_rejects_then_accepts(tmp_path):
    j = Journal()
    parent = j.add(ROOT_ID, dict(G.BASE), plan="base")
    j.record(parent.id, proxy_auc=0.8, gold_auc=0.9)
    cold = tmp_path / "cold.md"
    cold.write_text("- fact one\n")
    replies = [
        {"type": "train", "plan": "two changes", "genome": dict(G.BASE, labels="blend_v2", k=16)},
        {"type": "train", "plan": "unfreeze everything", "genome": dict(G.BASE, freeze_blocks=1)},
        {"type": "train", "plan": "swap teacher", "genome": dict(G.BASE, labels="blend_v2"), "expected_gain": 0.002},
    ]
    calls = []

    def llm(system, user, model):
        calls.append(user)
        return replies[len(calls) - 1], 0.001

    p = planner.propose(j, parent, "primary", [], budget_left=290, cold_start=cold, llm=llm)
    assert p.type == "train" and p.genome["labels"] == "blend_v2" and p.attempts == 3
    assert "REJECTED" in calls[1] and "unfreeze" in calls[2]


def test_loop_cycle(sandbox):
    launched = sandbox
    loop.tick(llm=lambda *a, **k: ({}, 0.0), rng=random.Random(0))
    assert len(launched) == 1 and loop.load_state()["active"]["nodes"]

    def llm(system, user, model):
        return {"type": "train", "plan": "blend teacher", "genome": dict(G.BASE, labels="blend_v2")}, 0.002

    loop.tick(llm=llm, rng=random.Random(0))
    j = loop.load_journal()
    online = [n for n in j.nodes.values() if n.kind == "online"]
    assert len(online) == 2
    assert online[0].valid and 0.5 < online[0].proxy_auc < 1.0
    assert len(launched) == 2 and loop.load_state()["active"]["nodes"] == [online[1].id]
    kinds = [e["kind"] for e in activity.read()]
    assert "cost" in kinds and "gate" in kinds and "plan" in kinds


def test_infra_failure_relaunches_then_fails(sandbox):
    launched = sandbox
    loop.tick(llm=lambda *a, **k: ({}, 0.0), rng=random.Random(0))
    nid = loop.load_state()["active"]["nodes"][0]
    for attempt in (1, 2, 3):
        s = loop.load_state()
        s["active"]["result"] = "u/broken-out"
        loop.save_state(s)
        loop.tick(llm=lambda *a, **k: ({"type": "stop", "why": "test"}, 0.0), rng=random.Random(0))
    assert [j for j, _ in launched] == [f"n-{nid}", f"n-{nid}-r1", f"n-{nid}-r2"]
    assert loop.load_journal().nodes[nid].status == "failed"


def test_expensive_proposal_waits_for_approval(sandbox, monkeypatch):
    launched = sandbox
    monkeypatch.setattr(G, "est_hours", lambda g, gpu_factor=1.6: 40.0)
    loop.tick(llm=lambda *a, **k: ({}, 0.0), rng=random.Random(0))
    s = loop.load_state()
    assert launched == [] and s["pending"]["type"] == "train"
    s["pending"]["approved"] = True
    loop.save_state(s)
    loop.tick(rng=random.Random(0))
    assert len(launched) == 1
