"""DeepSeek tool-calling planner: launch_runpod_train auto-queues trains."""

from __future__ import annotations

import json

from rsna_knee.agent import genome as G, planner
from rsna_knee.agent.deepseek import _assistant_echo
from rsna_knee.agent.journal import ROOT_ID, Journal


def _local_run_tools(system, user, tools, handlers, chat_fn):
    """Minimal tool loop using a fake chat_fn (same contract as deepseek.run_tools)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    cost, trace, final = 0.0, [], None
    for _ in range(6):
        msg, c = chat_fn(messages, tools=tools, tool_choice="auto")
        cost += c
        messages.append(_assistant_echo(msg))
        calls = msg.get("tool_calls") or []
        if not calls:
            raw = (msg.get("content") or "").strip()
            if raw:
                try:
                    final = json.loads(raw)
                except json.JSONDecodeError:
                    final = {"type": "stop", "why": raw[:200]}
            break
        for tc in calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = json.loads(fn.get("arguments") or "{}")
            result = handlers[name](**args)
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": json.dumps(result)})
    return final, cost, trace


def test_propose_tools_launch_runpod(monkeypatch, tmp_path):
    j = Journal()
    parent = j.add(ROOT_ID, dict(G.BASE), plan="base")
    j.record(parent.id, proxy_auc=0.92, gold_auc=0.88)
    cold = tmp_path / "cold.md"
    cold.write_text("- dread baseline proxy 0.9215\n")

    rounds = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "check status then launch",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_runpod_status", "arguments": "{}"},
            }],
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "idle; launch blend_v2",
            "tool_calls": [{
                "id": "c2", "type": "function",
                "function": {
                    "name": "launch_runpod_train",
                    "arguments": json.dumps({
                        "plan": "swap teacher to blend_v2",
                        "genome": dict(G.BASE, labels="blend_v2"),
                        "expected_gain": 0.002,
                        "why": "unused label set",
                    }),
                },
            }],
        },
        {"role": "assistant", "content": '{"type":"train","ok":true}'},
    ]
    n = {"i": 0}

    def fake_chat(messages, **kw):
        msg = rounds[n["i"]]
        n["i"] += 1
        return msg, 0.001

    monkeypatch.setattr(
        planner, "run_tools",
        lambda system, user, tools, handlers, **kw: _local_run_tools(system, user, tools, handlers, fake_chat),
    )
    p = planner.propose(j, parent, "primary", [], budget_left=290, cold_start=cold, use_tools=True)
    assert p.type == "train" and p.genome["labels"] == "blend_v2"
    assert p.attempts >= 1
    assert any(t["tool"] == "launch_runpod_train" and t["result"].get("ok") for t in p.tool_trace)


def test_launch_tool_rejects_bad_genome(monkeypatch, tmp_path):
    j = Journal()
    parent = j.add(ROOT_ID, dict(G.BASE), plan="base")
    j.record(parent.id, proxy_auc=0.9, gold_auc=0.88)
    cold = tmp_path / "cold.md"
    cold.write_text("- fact\n")

    rounds = [
        {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "bad", "type": "function",
                "function": {
                    "name": "launch_runpod_train",
                    "arguments": json.dumps({
                        "plan": "unfreeze", "genome": dict(G.BASE, freeze_blocks=1),
                        "why": "try unfreeze",
                    }),
                },
            }],
        },
        {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "ok", "type": "function",
                "function": {
                    "name": "launch_runpod_train",
                    "arguments": json.dumps({
                        "plan": "labels=blend_v2", "genome": dict(G.BASE, labels="blend_v2"),
                        "why": "unused label set", "expected_gain": 0.001,
                    }),
                },
            }],
        },
        {"role": "assistant", "content": "{}"},
    ]
    n = {"i": 0}

    def fake_chat(messages, **kw):
        msg = rounds[n["i"]]
        n["i"] += 1
        return msg, 0.0

    monkeypatch.setattr(
        planner, "run_tools",
        lambda system, user, tools, handlers, **kw: _local_run_tools(system, user, tools, handlers, fake_chat),
    )
    p = planner.propose(j, parent, "primary", [], budget_left=100, cold_start=cold)
    assert p.type == "train" and p.genome["labels"] == "blend_v2"
    assert p.tool_trace[0]["result"]["ok"] is False
    assert p.tool_trace[1]["result"]["ok"] is True
