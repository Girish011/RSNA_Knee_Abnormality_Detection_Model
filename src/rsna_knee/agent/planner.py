"""LLM planner: propose the next node as a genome diff (or a relabel request), then validate.

MLEvolve's planner/coder split, with the coder removed: the output is a typed genome that
``genome.to_cli`` compiles, so a bad proposal is rejected by validation instead of crashing
a rented GPU. Validation errors are fed back to the model for up to ``tries`` attempts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rsna_knee.agent import genome as G
from rsna_knee.agent.deepseek import FLASH, chat_json
from rsna_knee.agent.journal import ROOT_ID, Journal, Node, genome_hash
from rsna_knee.agent.memory import BANS, check_bans, journal_records, load_cold_start, retrieve
from rsna_knee.constants import LABEL_COLS

MAX_DIFF = {"primary": 1, "intra_branch": 2, "cross_branch": 2, "aggregate": 3}
RELABEL_RULES = ["pilk_unk", "disagree", "both"]

SYSTEM = """You are the planner of an autonomous ML search for the RSNA knee MRI competition
(macro ROC AUC over 12 findings, hidden test is expert-read images, reports exist only in training).
You choose the NEXT experiment. You cannot write code: you either change fields of a typed genome
or request a relabel of weak label cells with an LLM report reader. Reply with JSON only."""

FORMAT = """Reply with exactly one of:
{"type": "train", "plan": "<one sentence>", "genome": {<full genome>}, "expected_gain": <float, proxy AUC>, "why": "<evidence from memory>"}
{"type": "relabel", "plan": "<one sentence>", "labels": [<finding names>], "rule": "pilk_unk"|"disagree"|"both", "base": "<registry label set>", "why": "<evidence>"}
{"type": "stop", "why": "<reason no experiment is worth its cost>"}"""


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


def _node_line(n: Node) -> str:
    return (f"{n.id} [{n.kind}] proxy={n.proxy_auc} gold_dev={n.gold_auc} status={n.status} "
            f"genome={json.dumps(n.genome, sort_keys=True)} plan={n.plan!r}")


def build_prompt(journal: Journal, parent: Node, operator: str, refs: list[str],
                 budget_left: float, cold_start: Path) -> str:
    top = sorted(journal.valid_nodes(), key=lambda n: n.proxy_auc, reverse=True)[:6]
    parent_genome = parent.genome or G.BASE
    query = f"{operator} {json.dumps(parent_genome)} {parent.plan}"
    mem = retrieve(query, load_cold_start(cold_start) + journal_records(journal), k=8)
    reg = {k: v.get("audit", {}) for k, v in G.label_registry().items()}
    return "\n".join([
        f"PROTOCOL (fixed, do not change): {G.PROTOCOL}; proxy = macro AUC of fold-0 OOF predictions vs a fixed "
        f"reference label set on ~870 non-expert studies; gold_dev = 29 expert studies (veto only, sd ~0.02-0.04).",
        f"GENOME MENU: {json.dumps(G.MENU)}",
        f"LABEL REGISTRY (name -> audit on expert-dev): {json.dumps(reg)}",
        f"FINDINGS: {LABEL_COLS}",
        f"PARENT NODE: {_node_line(parent) if parent.id != ROOT_ID else 'root; base genome ' + json.dumps(G.BASE)}",
        f"OPERATOR: {operator} (change at most {MAX_DIFF.get(operator, 1)} genome field(s) vs the parent)",
        "REFERENCE NODES: " + ("; ".join(_node_line(journal.nodes[r]) for r in refs) or "none"),
        "BEST NODES: " + ("; ".join(_node_line(n) for n in top) or "none yet"),
        "MEMORY (measured facts; do not repeat kills):\n- " + "\n- ".join(mem),
        "BANNED: " + "; ".join(f"{b.rule}: {b.why}" for b in BANS),
        f"BUDGET LEFT: ${budget_left:.0f}; one 16-epoch CoAtNet train costs about $1-4.",
        FORMAT,
    ])


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


def propose(journal: Journal, parent: Node, operator: str, refs: list[str], *, budget_left: float,
            cold_start: Path, model: str = FLASH, tries: int = 3, llm=chat_json) -> Proposal:
    prompt = build_prompt(journal, parent, operator, refs, budget_left, cold_start)
    cost, feedback = 0.0, ""
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
    return Proposal(type="stop", why=f"no valid proposal after {tries} tries: {errs}", cost_usd=cost, attempts=tries)
