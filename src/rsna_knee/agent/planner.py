"""LLM planner: propose the next node as a genome diff (or a relabel request), then validate.

MLEvolve's planner/coder split, with the coder removed: the output is a typed genome that
``genome.to_cli`` compiles, so a bad proposal is rejected by validation instead of crashing
a rented GPU. DeepSeek uses native tool calls: ``launch_runpod_train`` auto-queues a pod
(host starts it after budget/approval gates); validation errors are returned as tool results
so the model can fix and re-call within one planning turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rsna_knee.agent import genome as G
from rsna_knee.agent.deepseek import FLASH, chat_json, run_tools
from rsna_knee.agent.journal import ROOT_ID, Journal, Node, genome_hash
from rsna_knee.agent.memory import BANS, check_bans, journal_records, load_cold_start, retrieve
from rsna_knee.constants import LABEL_COLS

MAX_DIFF = {"primary": 1, "intra_branch": 2, "cross_branch": 2, "aggregate": 3}
RELABEL_RULES = ["pilk_unk", "disagree", "both"]

SYSTEM = """You are the planner of an autonomous ML search for the RSNA knee MRI competition
(macro ROC AUC over 12 findings, hidden test is expert-read images, reports exist only in training).
Campaign target: push fold-0 proxy AUC to at least 0.958 (top-10 band). Do not stop while budget
remains and unused label sets or untried genome fields exist. You cannot write code: you either
change fields of a typed genome or request a relabel of weak label cells.

TOOL POLICY (mandatory):
- Whenever a train is the right next step, you MUST call launch_runpod_train. Do not only describe
  a train in text — the host only starts a GPU when that tool returns ok. The host auto-triggers
  the RunPod pod after budget and approval gates.
- Call get_runpod_status first if you are unsure whether a pod is already running (one pod at a time).
- Prefer launch_runpod_train over stop while unused LABEL REGISTRY entries or menu fields remain.
- Use request_relabel only for weak findings with evidence; use stop_search only when every menu
  field and unused label set is exhausted.
Never propose unfreeze, train-longer schedules past the menu, thin-cache work, or reports at test time."""

FORMAT = """Use tools. Prefer this order: (1) train any unused LABEL REGISTRY entry as a one-field
labels= change via launch_runpod_train; (2) geometry k / epochs within menu; (3) targeted relabel.
Never invent genome fields. After a successful tool call you may reply with a short JSON summary
{"type":"train"|"relabel"|"stop","ok":true} or empty content."""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_runpod_status",
            "description": "Read whether a RunPod job is already active / pending approval. "
                           "Call before launch_runpod_train if a pod might already be running.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch_runpod_train",
            "description": "Queue a CoAtNet train on RunPod for a full typed genome. Call this "
                           "whenever a train is needed. On ok the host auto-starts the pod "
                           "(subject to one-pod, budget, and >$10 approval gates).",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "One sentence plan"},
                    "genome": {"type": "object", "description": "Full genome dict from the menu"},
                    "expected_gain": {"type": "number", "description": "Expected proxy AUC after train"},
                    "why": {"type": "string", "description": "Evidence from memory / registry"},
                },
                "required": ["plan", "genome", "why"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_relabel",
            "description": "Request an LLM relabel of weak label cells (not a GPU train).",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "rule": {"type": "string", "enum": RELABEL_RULES},
                    "base": {"type": "string", "description": "Registry label set name"},
                    "why": {"type": "string"},
                },
                "required": ["plan", "labels", "rule", "base", "why"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_search",
            "description": "Stop only if every menu field and unused label set is exhausted.",
            "parameters": {
                "type": "object",
                "properties": {"why": {"type": "string"}},
                "required": ["why"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class Proposal:
    type: str
    plan: str = ""
    genome: dict = field(default_factory=dict)
    expected_gain: float = 0.0
    why: str = ""
    labels: list[str] = field(default_factory=list)
    rule: str = ""
    base: str = ""
    cost_usd: float = 0.0
    attempts: int = 0
    tool_trace: list = field(default_factory=list)


def _node_line(n: Node) -> str:
    return (f"{n.id} [{n.kind}] proxy={n.proxy_auc} gold_dev={n.gold_auc} status={n.status} "
            f"genome={json.dumps(n.genome, sort_keys=True)} plan={n.plan!r}")


def build_prompt(journal: Journal, parent: Node, operator: str, refs: list[str],
                 budget_left: float, cold_start: Path, target: float = 0.958,
                 runpod_hint: str = "") -> str:
    top = sorted(journal.valid_nodes(), key=lambda n: n.proxy_auc, reverse=True)[:6]
    parent_genome = parent.genome or G.BASE
    query = f"{operator} {json.dumps(parent_genome)} {parent.plan}"
    mem = retrieve(query, load_cold_start(cold_start) + journal_records(journal), k=8)
    reg = {k: v.get("audit", {}) for k, v in G.label_registry().items()}
    used_labels = sorted({n.genome.get("labels") for n in journal.nodes.values()
                          if n.kind == "online" and n.genome.get("labels")})
    unused = [k for k in reg if k not in used_labels]
    best_proxy = top[0].proxy_auc if top else None
    gap = (None if best_proxy is None else round(target - best_proxy, 4))
    lines = [
        f"TARGET: fold-0 proxy AUC >= {target} (current best {best_proxy}; gap {gap}). Keep proposing until target or budget is gone.",
        f"PROTOCOL (fixed, do not change): {G.PROTOCOL}; proxy = macro AUC of fold-0 OOF predictions vs a fixed "
        f"reference label set on ~870 non-expert studies; gold_dev = 29 expert studies (veto only, sd ~0.02-0.04).",
        f"GENOME MENU: {json.dumps(G.MENU)}",
        f"LABEL REGISTRY (name -> audit on expert-dev): {json.dumps(reg)}",
        f"UNUSED LABEL SETS (prefer training these next): {unused or 'none'}",
        f"FINDINGS: {LABEL_COLS}",
        f"PARENT NODE: {_node_line(parent) if parent.id != ROOT_ID else 'root; base genome ' + json.dumps(G.BASE)}",
        f"OPERATOR: {operator} (change at most {MAX_DIFF.get(operator, 1)} genome field(s) vs the parent)",
        "REFERENCE NODES: " + ("; ".join(_node_line(journal.nodes[r]) for r in refs) or "none"),
        "BEST NODES: " + ("; ".join(_node_line(n) for n in top) or "none yet"),
        "MEMORY (measured facts; do not repeat kills):\n- " + "\n- ".join(mem),
        "BANNED: " + "; ".join(f"{b.rule}: {b.why}" for b in BANS),
        f"BUDGET LEFT: ${budget_left:.0f}; one 16-epoch CoAtNet train costs about $1-4.",
    ]
    if runpod_hint:
        lines.append(f"RUNPOD HINT: {runpod_hint}")
    lines.append(FORMAT)
    return "\n".join(lines)


def check(p: Proposal, journal: Journal, parent: Node, operator: str) -> list[str]:
    errs = []
    if p.type == "stop":
        return errs
    if p.type == "relabel":
        bad = [l for l in p.labels if l not in LABEL_COLS]
        if not p.labels or bad:
            errs.append(f"relabel labels must be a non-empty subset of {LABEL_COLS}; bad {bad}")
        if p.rule not in RELABEL_RULES:
            errs.append(f"rule must be one of {RELABEL_RULES}")
        if p.base not in G.label_registry():
            errs.append(f"base {p.base!r} not in registry")
    elif p.type == "train":
        errs += G.validate(p.genome)
        if not errs:
            d = G.diff(parent.genome or G.BASE, p.genome)
            limit = MAX_DIFF.get(operator, 1)
            if parent.id != ROOT_ID and not d:
                errs.append("genome identical to parent")
            if len(d) > limit:
                errs.append(f"{operator} allows {limit} changed field(s); you changed {sorted(d)}")
            if genome_hash(p.genome) in journal.nodes:
                errs.append("this exact genome was already run; read its result instead")
    else:
        errs.append("type must be train, relabel or stop")
    hits = check_bans(f"{p.plan} {p.why} {json.dumps(p.genome)}")
    if p.type == "relabel":
        hits = [b for b in hits if b.rule != "reports_at_test"]
    errs += [f"banned ({b.rule}): {b.why}" for b in hits]
    return errs


def _default_status() -> dict:
    return {"active": None, "pending": None, "note": "status probe not wired; assume idle unless told otherwise"}


def propose(journal: Journal, parent: Node, operator: str, refs: list[str], *, budget_left: float,
            cold_start: Path, model: str = FLASH, tries: int = 3, llm: Callable | None = None,
            target: float = 0.958, status_fn: Callable[[], dict] | None = None,
            use_tools: bool = True) -> Proposal:
    """Propose the next action. Default path uses DeepSeek tool calls so trains go through
    ``launch_runpod_train`` (host auto-starts the pod). Pass ``llm=`` for the legacy JSON
    path used by unit tests.
    """
    if llm is not None or not use_tools:
        return _propose_json(journal, parent, operator, refs, budget_left=budget_left,
                             cold_start=cold_start, model=model, tries=tries,
                             llm=llm or chat_json, target=target)
    return _propose_tools(journal, parent, operator, refs, budget_left=budget_left,
                          cold_start=cold_start, model=model, target=target,
                          status_fn=status_fn or _default_status)


def _propose_json(journal: Journal, parent: Node, operator: str, refs: list[str], *,
                  budget_left: float, cold_start: Path, model: str, tries: int, llm,
                  target: float) -> Proposal:
    # Legacy JSON-only format for tests / offline fakes.
    fmt = (
        'Reply with exactly one of:\n'
        '{"type": "train", "plan": "<one sentence>", "genome": {<full genome>}, '
        '"expected_gain": <float, proxy AUC>, "why": "<evidence from memory>"}\n'
        '{"type": "relabel", "plan": "<one sentence>", "labels": [<finding names>], '
        '"rule": "pilk_unk"|"disagree"|"both", "base": "<registry label set>", "why": "<evidence>"}\n'
        '{"type": "stop", "why": "<only if every menu field and unused label set is exhausted>"}\n'
        "Prefer: (1) train any unused LABEL REGISTRY entry as a one-field labels= change; "
        "(2) geometry k / epochs within menu; (3) targeted relabel of weak findings. Never invent fields."
    )
    prompt = build_prompt(journal, parent, operator, refs, budget_left, cold_start, target=target)
    prompt = prompt.replace(FORMAT, fmt)
    cost, feedback = 0.0, ""
    errs: list[str] = []
    for i in range(tries):
        reply, c = llm(SYSTEM, prompt + feedback, model=model)
        cost += c
        p = Proposal(type=str(reply.get("type", "")), plan=str(reply.get("plan", "")),
                     genome=dict(reply.get("genome") or {}), why=str(reply.get("why", "")),
                     expected_gain=float(reply.get("expected_gain") or 0.0),
                     labels=list(reply.get("labels") or []), rule=str(reply.get("rule", "")),
                     base=str(reply.get("base", "")))
        errs = check(p, journal, parent, operator)
        if not errs:
            p.cost_usd, p.attempts = cost, i + 1
            return p
        feedback = "\n\nYOUR LAST PROPOSAL WAS REJECTED:\n- " + "\n- ".join(errs) + "\nFix it and reply again."
    return Proposal(type="stop", why=f"no valid proposal after {tries} tries: {errs}",
                    cost_usd=cost, attempts=tries)


def _propose_tools(journal: Journal, parent: Node, operator: str, refs: list[str], *,
                   budget_left: float, cold_start: Path, model: str, target: float,
                   status_fn: Callable[[], dict]) -> Proposal:
    prompt = build_prompt(journal, parent, operator, refs, budget_left, cold_start, target=target,
                          runpod_hint="Call get_runpod_status if needed, then launch_runpod_train "
                                      "to auto-trigger the pod when a train is required.")
    box: dict[str, Any] = {"proposal": None, "attempts": 0}

    def get_runpod_status() -> dict:
        try:
            return {"ok": True, **(status_fn() or {})}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def launch_runpod_train(plan: str = "", genome: dict | None = None, why: str = "",
                            expected_gain: float = 0.0, **_extra) -> dict:
        box["attempts"] += 1
        p = Proposal(type="train", plan=str(plan), genome=dict(genome or {}), why=str(why),
                     expected_gain=float(expected_gain or 0.0))
        errs = check(p, journal, parent, operator)
        if errs:
            return {"ok": False, "errors": errs, "hint": "Fix genome/plan and call launch_runpod_train again."}
        box["proposal"] = p
        return {"ok": True, "queued": True,
                "message": "Host will auto-start RunPod after one-pod / budget / approval gates."}

    def request_relabel(plan: str = "", labels: list | None = None, rule: str = "",
                        base: str = "", why: str = "", **_extra) -> dict:
        box["attempts"] += 1
        p = Proposal(type="relabel", plan=str(plan), labels=list(labels or []), rule=str(rule),
                     base=str(base), why=str(why))
        errs = check(p, journal, parent, operator)
        if errs:
            return {"ok": False, "errors": errs}
        box["proposal"] = p
        return {"ok": True, "queued": True, "message": "Relabel queued (no GPU)."}

    def stop_search(why: str = "", **_extra) -> dict:
        box["attempts"] += 1
        box["proposal"] = Proposal(type="stop", why=str(why) or "model called stop_search")
        return {"ok": True, "stopped": True}

    handlers = {
        "get_runpod_status": get_runpod_status,
        "launch_runpod_train": launch_runpod_train,
        "request_relabel": request_relabel,
        "stop_search": stop_search,
    }
    _final, cost, trace = run_tools(SYSTEM, prompt, TOOLS, handlers, model=model, max_rounds=6)
    p = box["proposal"]
    if p is None:
        return Proposal(type="stop",
                        why="no tool accepted a proposal (call launch_runpod_train when a train is needed)",
                        cost_usd=cost, attempts=max(1, box["attempts"]), tool_trace=trace)
    p.cost_usd, p.attempts, p.tool_trace = cost, max(1, box["attempts"]), trace
    return p
