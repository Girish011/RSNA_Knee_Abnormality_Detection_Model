import json

import pytest

from rsna_knee.agent import activity, genome as G, members, strategist
from rsna_knee.agent.journal import ROOT_ID, Journal


@pytest.fixture
def ssandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(activity, "ACTIVITY", tmp_path / "activity.jsonl")
    monkeypatch.setattr(members, "MEMBERS", tmp_path / "members.json")
    monkeypatch.setattr(strategist, "SUBMITS", tmp_path / "submits.json")
    monkeypatch.setattr(strategist, "analyze", lambda j, r: {"stub": True})
    j = Journal()
    parent = j.add(ROOT_ID, dict(G.BASE, corpus="dense80"), plan="base")
    j.record(parent.id, proxy_auc=0.93, gold_auc=0.9)
    (tmp_path / "journal.json").write_text(json.dumps([{"id": parent.id, "genome": parent.genome}]))
    monkeypatch.setattr(members, "JOURNAL_FILE", tmp_path / "journal.json")
    return j, parent, tmp_path


def _scripted(calls):
    """Fake run_tools: invoke the given (tool, args) through the real handlers."""
    def run(system, prompt, tools, handlers, **kw):
        names = {t["function"]["name"] for t in tools}
        trace = []
        for name, args in calls:
            assert name in names
            trace.append({"tool": name, "args": args, "result": handlers[name](**args)})
        return {"done": True}, 0.01, trace
    return run


def test_research_rate_limit_and_daily_cap(ssandbox, monkeypatch):
    j, parent, tmp = ssandbox
    state = {}
    assert strategist.research(state, j, tmp, llm_tools=_scripted([]), now=1000.0) is True
    assert strategist.research(state, j, tmp, llm_tools=_scripted([]), now=1001.0) is False
    # relabel budget exhausted: thinking turns still run
    monkeypatch.setattr(strategist, "llm_spent_today", lambda events=None, kinds=strategist.RELABEL_KINDS:
                        99.0 if kinds == strategist.RELABEL_KINDS else 0.1)
    assert strategist.research({}, j, tmp, llm_tools=_scripted([]), now=5000.0) is True
    # thinking budget exhausted: skip
    monkeypatch.setattr(strategist, "llm_spent_today", lambda events=None, kinds=strategist.RELABEL_KINDS: 99.0)
    assert strategist.research({}, j, tmp, llm_tools=_scripted([]), now=9000.0) is False


def test_closed_axes_refused_label_change_queued(ssandbox):
    j, parent, tmp = ssandbox
    trace_holder = {}

    def run(system, prompt, tools, handlers, **kw):
        bad = handlers["queue_next_train"](parent=parent.id, genome=dict(parent.genome, bb_lr=5e-5),
                                           plan="lr", why="x")
        good = handlers["queue_next_train"](parent=parent.id, genome=dict(parent.genome, labels="blend_v2"),
                                            plan="labels", why="x")
        again = handlers["queue_next_train"](parent=parent.id, genome=dict(parent.genome, labels="dread", corpus="raptor44"),
                                             plan="second", why="x")
        trace_holder.update(bad=bad, good=good, again=again)
        return {}, 0.0, []

    strategist.research({}, j, tmp, llm_tools=run, now=1.0)
    assert trace_holder["bad"]["ok"] is False and "closed axis" in trace_holder["bad"]["errors"][0]
    assert trace_holder["good"]["ok"] is True
    assert j.nodes[trace_holder["good"]["node"]].status == "pending"
    assert trace_holder["again"]["ok"] is False  # one pending fold-0 train at a time


def test_relabel_respects_caps_and_runs(ssandbox, monkeypatch):
    j, parent, tmp = ssandbox
    ran = []
    monkeypatch.setattr(strategist.labeler, "estimate", lambda labels, rule, model: (4000, 20.0))
    state = {}
    strategist.research(state, j, tmp, now=1.0, llm_tools=_scripted(
        [("request_relabel", {"labels": ["Synovitis"], "rule": "all", "base": "dread", "model": "pro", "why": "w"})]))
    monkeypatch.setattr(strategist.labeler, "estimate", lambda labels, rule, model: (4000, 2.0))
    monkeypatch.setattr(strategist.labeler, "run", lambda labels, rule, base, **kw: ran.append(kw) or {
        "name": "relabel_x", "kept": ["Synovitis"], "audit": {}, "cells_changed": 10, "cost_usd": 1.9,
        "registered": True})
    strategist.research(state, j, tmp, now=10_000.0, llm_tools=_scripted(
        [("request_relabel", {"labels": ["Synovitis"], "rule": "all", "base": "dread", "model": "pro", "why": "w"})]))
    assert len(ran) == 1 and ran[0]["model"] == strategist.labeler.PRO
    assert state["relabels"][0]["name"] == "relabel_x"


def test_queue_member_job_via_tool(ssandbox):
    j, parent, tmp = ssandbox
    out = {}

    def run(system, prompt, tools, handlers, **kw):
        out["r"] = handlers["queue_member_job"](node=parent.id, seeds=[7], labels=["dread"], init="v5",
                                                epochs=6, why="diversity")
        return {}, 0.0, []

    strategist.research({}, j, tmp, llm_tools=run, now=1.0)
    assert out["r"]["ok"] is True and members.load()["queue"][0]["arms"][0]["init"] == "v5"
