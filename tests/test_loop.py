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
    monkeypatch.setattr(loop.members, "MEMBERS", tmp_path / "members.json")
    ids = np.array([f"s{i}" for i in range(200)])
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.integers(0, 2, (200, 12)), index=ids, columns=LABEL_COLS)
    monkeypatch.setattr(loop, "reference", lambda: (ref, pd.DataFrame(True, index=ids, columns=LABEL_COLS)))
    launched = []

    def fake_launch(job, arms, hours, **kwargs):
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
    assert G.BASE["k"] == 8 and G.BASE["arch"] == "coatnet384"
    labels, args = G.to_cli(G.BASE, "/b")
    assert labels == "labels_dread.parquet"
    assert "--fold 0" in args and "--holdout /b/expert_holdout.json" in args
    bad = dict(G.BASE, epochs=40)
    errs = G.validate(bad)
    assert any("epochs" in e for e in errs)
    assert G.validate(dict(G.BASE, corpus="dense80")) == []
    assert G.validate(dict(G.BASE, corpus="raptor44")) == []
    assert any("k=" in e or "banned" in e for e in G.validate(dict(G.BASE, k=12)))
    assert any("k_eval" in e or "banned" in e for e in G.validate(dict(G.BASE, k_eval=42)))
    assert G.clamp({"corpus": "dense80", "labels": "dread", "arch": "coatnet384",
                    "epochs": 16, "k": 16, "k_eval": 42, "bb_lr": 3e-5, "head_lr": 1e-3,
                    "wd": 0.02, "freeze_blocks": 0})["k"] == 8
    assert G.validate(dict(G.BASE, arch="convnext_b384"))  # not on MENU


def test_planner_rejects_then_accepts(tmp_path):
    j = Journal()
    parent = j.add(ROOT_ID, dict(G.BASE), plan="base")
    j.record(parent.id, proxy_auc=0.8, gold_auc=0.9)
    cold = tmp_path / "cold.md"
    cold.write_text("- fact one\n")
    replies = [
        {"type": "train", "plan": "two changes", "genome": dict(G.BASE, labels="blend_v2", epochs=12)},
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


def test_loop_cycle(sandbox, monkeypatch):
    launched = sandbox
    monkeypatch.setattr(loop.runpod, "get_pod", lambda pid: {"desiredStatus": "RUNNING"})
    loop.tick(llm=lambda *a, **k: ({}, 0.0), rng=random.Random(0))
    assert len(launched) == 1 and loop.load_state()["active"]["nodes"]
    # Simulate finished pod: gone from API, long elapsed, results downloadable.
    s = loop.load_state()
    s["active"]["launched"] = __import__("time").time() - 10_000
    loop.save_state(s)
    monkeypatch.setattr(loop.runpod, "get_pod", lambda pid: (_ for _ in ()).throw(RuntimeError("404")))

    def llm(system, user, model):
        return {"type": "train", "plan": "blend teacher", "genome": dict(G.BASE, labels="blend_v2")}, 0.002

    loop.tick(llm=llm, rng=random.Random(0))
    j = loop.load_journal()
    online = [n for n in j.nodes.values() if n.kind == "online"]
    assert online[0].valid and 0.5 < online[0].proxy_auc < 1.0
    assert len(launched) >= 2  # baseline + next train
    kinds = [e["kind"] for e in activity.read()]
    assert "cost" in kinds and "gate" in kinds


def test_infra_failure_relaunches_then_fails(sandbox):
    launched = sandbox
    loop.tick(llm=lambda *a, **k: ({}, 0.0), rng=random.Random(0))
    nid = loop.load_state()["active"]["nodes"][0]
    for attempt in (1, 2, 3):
        s = loop.load_state()
        s["active"]["result"] = "u/broken-out"
        loop.save_state(s)
        loop.tick(llm=lambda *a, **k: ({"type": "stop", "why": "test"}, 0.0), rng=random.Random(0))
    assert [j for j, _ in launched[:3]] == [f"n-{nid}", f"n-{nid}-r1", f"n-{nid}-r2"]
    assert loop.load_journal().nodes[nid].status == "failed"


def test_deepseek_train_proposal_launches(sandbox, monkeypatch):
    """Idle ticks go to DeepSeek; a train tool/JSON proposal must start a pod."""
    launched = sandbox
    from rsna_knee.agent.journal import ROOT_ID
    from rsna_knee.agent import genome as G
    j = loop.load_journal()
    node = j.add(ROOT_ID, dict(G.BASE), plan="base")
    j.record(node.id, proxy_auc=0.92, gold_auc=0.88)
    j.save()
    monkeypatch.setattr(G, "label_registry", lambda: {
        "dread": {"file": "labels_dread.parquet"},
        "blend_v2": {"file": "labels_llm_blend_v2.parquet"},
        "relabel_8c56cdf0": {"file": "labels_relabel_8c56cdf0.parquet"},
    })
    g = dict(G.BASE, labels="relabel_8c56cdf0")
    s = loop.load_state()
    s.update({"relabels": [{"name": "relabel_8c56cdf0", "registered": True, "kept": ["Fracture"]}],
              "active": None, "pending": None, "next_plan_after": 0, "paused": False})
    loop.save_state(s)
    loop.tick(llm=lambda *a, **k: ({
        "type": "train", "plan": "train unused relabel", "genome": g, "why": "unused labels",
        "expected_gain": 0.94}, 0.0), rng=__import__("random").Random(0))
    assert any("relabel_8c56cdf0" in str(arms) for _, arms in launched)
    assert loop.load_state()["active"]["nodes"]


def test_dream_parent_stop_still_launches_train(sandbox):
    """Elite/UCT can select a dream blend; stop-fallback must use an online genome."""
    launched = sandbox
    j = loop.load_journal()
    a = j.add(ROOT_ID, dict(G.BASE), plan="a")
    j.record(a.id, proxy_auc=0.92, gold_auc=0.88)
    b = j.add(a.id, dict(G.BASE, epochs=12), plan="b")
    j.record(b.id, proxy_auc=0.93, gold_auc=0.89)
    dream = j.add(a.id, {"blend": [a.id, b.id]}, plan="dream: rank blend", kind="dream",
                  refs=[b.id], operator="aggregate")
    j.record(dream.id, proxy_auc=0.94)
    j.save()
    s = loop.load_state()
    s.update({"active": None, "pending": None, "next_plan_after": 0, "paused": False,
              "relabels": []})
    loop.save_state(s)
    # DeepSeek stop → host safety-net must still launch from an online genome.
    loop.tick(llm=lambda *a, **k: ({"type": "stop", "why": "stuck"}, 0.0), rng=random.Random(0))
    assert launched, "expected a safety-net train"
    assert loop.load_state()["active"] is not None
    assert any("safety-net" in (e.get("msg") or "") for e in activity.read())


def test_relabel_count_unique_registered_only():
    assert loop._relabel_count({"relabels": [
        {"name": "a", "registered": True},
        {"name": "a", "registered": True},
        {"name": "b", "registered": False},
        {"name": "c", "registered": True},
    ]}) == 2


def test_max_relabel_forces_menu_train(sandbox, monkeypatch):
    launched = sandbox
    j = loop.load_journal()
    parent = j.add(ROOT_ID, dict(G.BASE), plan="base")
    j.record(parent.id, proxy_auc=0.92, gold_auc=0.88)
    j.save()
    monkeypatch.setattr(loop, "_unused_label_sets", lambda *a, **k: [])
    s = loop.load_state()
    s.update({
        "active": None, "pending": None, "next_plan_after": 0, "paused": False,
        "relabels": [{"name": f"relabel_{i}", "registered": True, "kept": ["Fracture"]}
                     for i in range(loop.MAX_RELABELS)],
    })
    loop.save_state(s)
    loop.tick(llm=lambda *a, **k: ({
        "type": "relabel", "plan": "more", "labels": ["Fracture"], "rule": "both",
        "base": "dread", "why": "noise"}, 0.0), rng=random.Random(0))
    assert launched and loop.load_state()["active"] is not None
    assert any("max-relabel" in (e.get("msg") or "") for e in activity.read())


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
