"""DeepSeek strategist: works WHILE the GPU is busy, not only when it is idle.

The idle-tick planner (``planner.propose``) decides the next fold-0 train once per pod. That
left DeepSeek unused for the 3-7 h each pod runs. The strategist runs every
``RESEARCH_EVERY_S`` while a pod (fold-0 or member job) is active and does the CPU / API work
that moves the leaderboard in the ensemble era:

* reads per-finding evidence (``analyze_results``) instead of one macro number;
* re-labels weak findings from the reports (``request_relabel``: any rule incl. ``all``,
  flash or pro, audited on expert-dev by the labeler) — this runs immediately, no GPU;
* keeps the member queue stocked (``queue_member_job``) so the next pod starts the moment
  the slot frees, with no planning gap;
* queues the next fold-0 exploration train (``queue_next_train``) for label/geometry levers
  only — lr / wd / head_lr / epoch-only diffs are a closed axis;
* builds (never pushes) ensemble submit kernels (``build_ensemble_kernel``);
* may fix harness code through the allowlisted coder tools.

Hard limits stay on the host: one pod, $300 campaign cap, >$10 needs the user, no LB pushes or
submits, daily DeepSeek spend cap ``DAILY_LLM_USD``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from rsna_knee.agent import coder, genome as G, labeler, members, runpod
from rsna_knee.agent.activity import BUDGET_USD, RUNS, log, read, spent
from rsna_knee.agent.deepseek import FLASH, run_tools
from rsna_knee.agent.journal import Journal, genome_hash
from rsna_knee.constants import LABEL_COLS

REPO = G.REPO
RESEARCH_EVERY_S = 45 * 60
DAILY_LLM_USD = 15.0           # DeepSeek spend per calendar day (planner + strategist + relabels)
RELABEL_MAX_USD = 12.0         # one relabel job (a full pro re-read of ~4 findings is ~$7-10)
MAX_RELABELS = 12              # unique registered label sets (budget guard, not science)
CLOSED_AXES = {"bb_lr", "head_lr", "wd", "epochs"}
LLM_KINDS = {"plan", "research", "data", "code"}
SUBMITS = RUNS / "submits.json"
V5_GOLD58 = {"ACL": 0.964, "MCL": 0.980, "Medial Meniscus": 0.957, "Lateral Meniscus": 0.877,
             "Medial OA": 0.977, "Lateral OA": 0.867, "PF OA": 0.847, "Effusion": 0.976,
             "Synovitis": 0.797, "Baker's": 0.971, "Contusion": 0.928, "Fracture": 0.917}

SYSTEM = """You are the autonomous research lead for the RSNA knee MRI competition (macro ROC AUC,
12 findings). The trophy is the Kaggle public/final LB: gate 0.958, aim 0.970. Top of board ~0.960.

What is measured (do not relitigate):
- Our single fold-0 CoAtNet = LB 0.920. Public single CoAtNet (full data) = 0.924. Public D4 stack
  (DINO + RadImageNet + 4 CoAtNet families) = 0.943. Submissions are D4 stack + OUR members,
  rank-blended (build_ensemble_kernel). The gap is ensemble strength + label quality.
- lr / wd / head_lr / epochs one-field diffs landed within +-0.003 proxy: CLOSED. Never queue them.
- Members = full-data retrains; public v5 warm start (init=v5, 4-8 epochs) is allowed on members only.
- Weakest findings everywhere: Synovitis, PF OA, Lateral OA, Lateral Meniscus. Labels come from
  LLM reads of radiology reports; better/extra reads of those findings are the cheapest lever.
- The fold-0 proxy is a compass with a known optimism (~-0.017 vs LB). gold_dev/holdout are 29 experts each.

You run WHILE the GPU trains. Each turn, do the highest-value actions, typically 1-3 of:
1) analyze_results first (per-finding evidence).
2) request_relabel for weak findings when a new read could beat the registry (rule 'all' with
   model 'pro' re-reads every study; 'pilk_unk'/'disagree' re-read only uncertain cells).
3) Keep ONE member job queued behind the running pod: diverse, strong recipes (new label sets,
   seeds, init=v5). Do not queue near-duplicates of members already done or queued.
4) queue_next_train only for label/corpus levers (fold-0 compass), never closed axes.
5) build_ensemble_kernel when new members finished and the ledger has no submit with them yet.
Use coder tools only for real harness bugs. End with a short JSON summary of what you did and why.
Never write secrets. You cannot push kernels or submit to the LB — the user approves those."""


def llm_spent_today(events: list[dict] | None = None) -> float:
    today = datetime.now().strftime("%Y-%m-%d")
    return float(sum(e.get("cost_usd") or 0.0 for e in (events if events is not None else read())
                     if e.get("kind") in LLM_KINDS and str(e.get("ts", "")).startswith(today)))


def _per_label(y: np.ndarray, p: np.ndarray, m: np.ndarray | None = None) -> dict[str, float]:
    out = {}
    for j, lab in enumerate(LABEL_COLS):
        sel = m[:, j] if m is not None else np.ones(len(y), bool)
        yy = y[sel, j].astype(int)
        if 0 < yy.sum() < len(yy):
            out[lab] = round(float(roc_auc_score(yy, p[sel, j])), 3)
    return out


def analyze(journal: Journal, results: Path) -> dict[str, Any]:
    """Per-finding evidence for the best online nodes, finished members and the public v5."""
    from rsna_knee.agent import loop as L
    ref, mask = L.reference()
    online = sorted([n for n in journal.valid_nodes() if n.kind == "online"],
                    key=lambda n: n.proxy_auc or 0, reverse=True)[:6]
    rows = {}
    for n in online:
        got = L._oof_for(n.id, results)
        if got is None:
            continue
        pred, ids = got
        keep = np.isin(ids, ref.index)
        r, m = ref.loc[ids[keep]].to_numpy(), mask.loc[ids[keep]].to_numpy()
        rows[n.id] = {"proxy": round(n.proxy_auc or 0, 4), "gold_dev": round(n.gold_auc or 0, 4),
                      "labels": n.genome.get("labels"), "per_finding_proxy": _per_label(r, pred[keep], m)}
    mem = members.load()
    done = {t: {k: r.get(k) for k in ("gold_dev", "holdout_auc", "init", "seed")}
            | {"labels": r["genome"]["labels"], "epochs": r["genome"]["epochs"]}
            for t, r in mem["done"].items()}
    weakest = sorted(V5_GOLD58, key=V5_GOLD58.get)[:4]
    return {
        "online_top": rows,
        "members_done": done,
        "members_queue": [(j["job"], [(a["genome"]["labels"], a["seed"], a.get("init")) for a in j["arms"]])
                          for j in mem["queue"]],
        "members_active": mem["active"] and mem["active"]["job"],
        "public_v5_gold58_per_finding": V5_GOLD58,
        "weakest_findings": weakest,
        "label_registry": {k: v.get("audit", {}) for k, v in G.label_registry().items()},
        "lb_ledger": json.loads(SUBMITS.read_text()) if SUBMITS.exists() else [],
    }


def _check_next_train(genome: dict, journal: Journal, parent_id: str) -> list[str]:
    errs = G.validate(genome)
    if parent_id not in journal.nodes:
        errs.append(f"parent {parent_id} not in journal")
    if errs:
        return errs
    d = G.diff(G.clamp(journal.nodes[parent_id].genome or G.BASE), genome)
    if not d:
        errs.append("identical to parent")
    elif set(d) <= CLOSED_AXES:
        errs.append(f"closed axis: {sorted(d)} (lr/wd/head_lr/epochs diffs landed within ±0.003)")
    if len(d) > 2:
        errs.append(f"max 2 changed fields; got {sorted(d)}")
    if genome_hash(genome) in journal.nodes:
        errs.append("genome already run")
    if any(n.kind == "online" and n.status == "pending" for n in journal.nodes.values()):
        errs.append("a fold-0 train is already pending; wait for it")
    return errs


def handlers(state: dict, journal: Journal, results: Path, actions: list[str]) -> dict:
    def analyze_results(**_):
        return analyze(journal, results)

    def get_runpod_status(**_):
        try:
            bal = runpod.balance().get("clientBalance")
        except Exception:  # noqa: BLE001
            bal = None
        return {"loop_active": state.get("active") and state["active"].get("job"),
                "members_active": members.load()["active"] is not None,
                "runpod_balance_usd": bal, "campaign_spent_usd": round(spent(read()), 2),
                "campaign_cap_usd": BUDGET_USD, "llm_spent_today_usd": round(llm_spent_today(), 2),
                "llm_daily_cap_usd": DAILY_LLM_USD}

    def request_relabel(labels: list | None = None, rule: str = "", base: str = "",
                        model: str = "flash", why: str = "", **_):
        labels = list(labels or [])
        errs = []
        if not labels or any(l not in LABEL_COLS for l in labels):
            errs.append(f"labels must be a non-empty subset of {LABEL_COLS}")
        if rule not in labeler.RULES:
            errs.append(f"rule must be one of {labeler.RULES}")
        if base not in G.label_registry():
            errs.append(f"base {base!r} not in registry {sorted(G.label_registry())}")
        if model not in labeler.MODELS:
            errs.append(f"model must be one of {sorted(labeler.MODELS)}")
        n_reg = len({r["name"] for r in state.get("relabels") or [] if r.get("registered")})
        if n_reg >= MAX_RELABELS:
            errs.append(f"relabel cap {MAX_RELABELS} reached")
        if errs:
            return {"ok": False, "errors": errs}
        n, est = labeler.estimate(labels, rule, labeler.MODELS[model])
        if est > RELABEL_MAX_USD:
            return {"ok": False, "errors": [f"est ${est:.2f} > per-job cap ${RELABEL_MAX_USD}; fewer findings or flash"]}
        if llm_spent_today() + est > DAILY_LLM_USD:
            return {"ok": False, "errors": [f"daily DeepSeek cap ${DAILY_LLM_USD} would be exceeded (est ${est:.2f})"]}
        res = labeler.run(labels, rule, base, model=labeler.MODELS[model], workers=32)
        state.setdefault("relabels", []).append(
            {"name": res["name"], "registered": res["registered"], "kept": res["kept"]})
        log("data", f"relabel {res['name']} ({rule}, {model}, {labels}) by strategist: kept {res['kept']}, "
            f"{res['cells_changed']} cells changed | {why[:120]}", cost_usd=res["cost_usd"], audit=res["audit"])
        actions.append(f"relabel {res['name']} kept={res['kept']}")
        return {"ok": True, **{k: res[k] for k in ("name", "kept", "audit", "cells_changed", "cost_usd", "registered")}}

    def queue_member_job(node: str = "", seeds: list | None = None, labels: list | None = None,
                         init: str | None = None, epochs: int | None = None, why: str = "", **_):
        m = members.load()
        if sum(1 for j in m["queue"]) >= 2:
            return {"ok": False, "errors": ["2 member jobs already queued; wait"]}
        try:
            job = members.queue(node, [int(s) for s in (seeds or [42])], list(labels) if labels else None,
                                m, init=init or None, epochs=int(epochs) if epochs else None)
        except SystemExit as e:
            return {"ok": False, "errors": [str(e)]}
        actions.append(f"member job {job['job']} ({len(job['arms'])} arms, ${job['est_usd']}, "
                       f"approved={job['approved']})")
        log("research", f"strategist queued member job {job['job']}: {why[:160]}", job=job["job"])
        return {"ok": True, "job": job["job"], "arms": [a["tag"] for a in job["arms"]],
                "est_usd": job["est_usd"], "approved": job["approved"],
                "note": "" if job["approved"] else "needs user approval (> $10)"}

    def queue_next_train(parent: str = "", genome: dict | None = None, plan: str = "", why: str = "", **_):
        g = dict(genome or {})
        if errs := _check_next_train(g, journal, parent):
            return {"ok": False, "errors": errs}
        node = journal.add(parent, g, plan=f"[strategist] {plan}", operator="cross_branch")
        actions.append(f"next fold-0 train {node.id} {G.diff(G.clamp(journal.nodes[parent].genome), g)}")
        log("research", f"strategist queued fold-0 train {node.id}: {plan[:160]}", node=node.id)
        return {"ok": True, "node": node.id, "note": "launches when the pod slot is free (after member jobs)"}

    def build_ensemble_kernel(name: str = "", members_list: list | None = None, w: float = 0.2,
                              why: str = "", **_):
        sys.path.insert(0, str(REPO / "scripts"))
        import build_ensemble_kernel as bek
        if not name.startswith("ens-") or not members_list:
            return {"ok": False, "errors": ["name must start with 'ens-' and members_list non-empty"]}
        try:
            d = bek.build(name, [str(x) for x in members_list], float(w))
        except SystemExit as e:
            return {"ok": False, "errors": [str(e)]}
        actions.append(f"built kernel {name} (needs user approval to push/submit)")
        log("research", f"strategist built kernel {d.name} members={members_list} w={w}: {why[:120]}")
        return {"ok": True, "path": str(d), "note": "built only; the user must approve push + submit"}

    return {"analyze_results": analyze_results, "get_runpod_status": get_runpod_status,
            "request_relabel": request_relabel, "queue_member_job": queue_member_job,
            "queue_next_train": queue_next_train, "build_ensemble_kernel": build_ensemble_kernel,
            **coder.handlers()}


def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required,
                       "additionalProperties": False}}}


TOOLS = [
    _fn("analyze_results", "Per-finding AUCs of top online nodes, finished/queued members, public v5, "
        "label registry audits and the LB ledger. Call first.", {}, []),
    _fn("get_runpod_status", "Pods, balances, campaign and daily DeepSeek spend.", {}, []),
    _fn("request_relabel", "Re-read reports for some findings NOW (no GPU), audited on expert-dev; "
        "registers a new label set for kept findings.",
        {"labels": {"type": "array", "items": {"type": "string"}},
         "rule": {"type": "string", "enum": list(labeler.RULES)},
         "base": {"type": "string", "description": "registry label set to modify"},
         "model": {"type": "string", "enum": sorted(labeler.MODELS)},
         "why": {"type": "string"}}, ["labels", "rule", "base", "model", "why"]),
    _fn("queue_member_job", "Queue full-data member retrains (one pod, several arms) from an online node.",
        {"node": {"type": "string"}, "seeds": {"type": "array", "items": {"type": "integer"}},
         "labels": {"type": "array", "items": {"type": "string"}},
         "init": {"type": "string", "enum": ["v5"]}, "epochs": {"type": "integer"},
         "why": {"type": "string"}}, ["node", "seeds", "labels", "why"]),
    _fn("queue_next_train", "Queue the next fold-0 compass train (label/corpus levers; closed axes refused).",
        {"parent": {"type": "string"}, "genome": {"type": "object"}, "plan": {"type": "string"},
         "why": {"type": "string"}}, ["parent", "genome", "plan", "why"]),
    _fn("build_ensemble_kernel", "Build (not push) a D4-stack + our members submit kernel.",
        {"name": {"type": "string"}, "members_list": {"type": "array", "items": {"type": "string"}},
         "w": {"type": "number"}, "why": {"type": "string"}}, ["name", "members_list", "w", "why"]),
] + coder.CODER_TOOLS


def research(state: dict, journal: Journal, results: Path, *, llm_tools=run_tools,
             model: str = FLASH, now: float | None = None) -> bool:
    """One strategist turn if due. Returns True if DeepSeek ran."""
    now = now or time.time()
    if now < float(state.get("next_research_after") or 0):
        return False
    state["next_research_after"] = now + RESEARCH_EVERY_S
    today = llm_spent_today()
    if today >= DAILY_LLM_USD:
        log("note", f"strategist skipped: DeepSeek spend today ${today:.2f} >= cap ${DAILY_LLM_USD}")
        return False
    actions: list[str] = []
    prompt = (f"STATE: {json.dumps(analyze(journal, results), default=str)[:30000]}\n"
              f"MENU: {json.dumps(G.MENU)}\nCLOSED AXES: {sorted(CLOSED_AXES)}\n"
              f"DeepSeek spend today ${today:.2f} of ${DAILY_LLM_USD}.\n"
              "Decide and act with tools now, then reply with a short JSON summary.")
    final, cost, trace = llm_tools(SYSTEM, prompt, TOOLS, handlers(state, journal, results, actions),
                                   model=model, max_rounds=10)
    summary = json.dumps(final, default=str)[:300] if final else ""
    log("research", f"strategist turn: {'; '.join(actions) or 'no action'} | {summary}",
        cost_usd=cost, tools=len(trace or []))
    return True
