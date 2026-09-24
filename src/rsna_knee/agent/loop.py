"""The autonomous search loop: observe -> dream -> plan -> check -> act -> record.

One ``tick`` is idempotent and resumable (all state is on disk), so the loop survives laptop
sleep and can move hosts. Guards:

* one pod at a time; a pod running past ``2 x estimate + 1 h`` is terminated;
* hard campaign cap (``activity.assert_budget``) and a per-action approval gate: any train or
  relabel estimated above ``APPROVAL_USD`` waits for ``loop approve``;
* the loop never submits to the leaderboard and never checks Kaggle kernel status; it only
  downloads a result dataset after the pod that uploads it has terminated;
* the proxy is scored on a fixed fold against reference cells where both public readers agree;
  expert-dev gold is a veto; the expert holdout is written to ``holdout.json`` for human keep
  decisions and never shown to the planner;
* ``agent/runs/STOP`` exits ``run``; ``loop pause`` stops planning but keeps observing.

CLI: ``python -m rsna_knee.agent.loop status|tick|run|approve|reject|pause|resume``.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee.agent import genome as G
from rsna_knee.agent import labeler, launcher, runpod
from rsna_knee.agent.activity import BUDGET_USD, JOURNAL, RUNS, log, read, spent
from rsna_knee.agent.dream import rank_blend
from rsna_knee.agent.journal import ROOT_ID, Journal, genome_hash, genome_hash
from rsna_knee.agent.planner import propose
from rsna_knee.agent.search import Schedule, choose_operator, select
from rsna_knee.constants import LABEL_COLS

REPO = G.REPO
STATE = RUNS / "loop_state.json"
HOLDOUT = RUNS / "holdout.json"
STOP_FILE = RUNS / "STOP"
RESULTS = REPO / "outputs" / "runpod_results"
COLD_START = REPO / "agent" / "memory" / "cold_start.md"
PUB = REPO / "data" / "external" / "public_labels"
APPROVAL_USD = 10.0
VETO_DROP = 0.04
MAX_RELABELS = 4
STOP_COOLDOWN_S = 15 * 60  # short; never park the campaign for hours
PROXY_TARGET = 0.958  # top-10 band from AGENT_PLAN; loop keeps searching until met or budget dies
DOWNLOAD_TRIES = 8
INFRA_RETRIES = 2


# ------------------------------------------------------------------ state

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"paused": False, "active": None, "pending": None, "next_plan_after": 0,
            "relabels": [], "heartbeat": 0}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1, default=str))
    tmp.replace(STATE)


def load_journal() -> Journal:
    return Journal.load(JOURNAL) if JOURNAL.exists() else Journal(JOURNAL)


# ------------------------------------------------------------------ scoring

_REF: tuple[pd.DataFrame, pd.DataFrame] | None = None


def reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(binary reference, mask): cells where pilk and dread agree on the side of 0.5."""
    global _REF
    if _REF is None:
        p = pd.read_csv(PUB / "pilk" / "report_labels_v2.csv").set_index("StudyInstanceUID")[LABEL_COLS]
        d = pd.read_csv(PUB / "dread" / "labels_llm_soft.csv").set_index("StudyInstanceUID")[LABEL_COLS]
        ids = p.index.intersection(d.index)
        pb, db = p.loc[ids] > 0.5, d.loc[ids] > 0.5
        _REF = (pb.astype(int), pb == db)
    return _REF


def proxy_auc(pred: np.ndarray, ids: np.ndarray) -> float:
    ref, mask = reference()
    ids = pd.Index(ids.astype(str))
    keep = ids.isin(ref.index)
    r, m, P = ref.loc[ids[keep]].to_numpy(), mask.loc[ids[keep]].to_numpy(), pred[keep]
    aucs = []
    for j in range(len(LABEL_COLS)):
        y = r[m[:, j], j]
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, P[m[:, j], j]))
    return float(np.mean(aucs))


def gold_macro(npz: Path) -> float | None:
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    y, p = z["truth"].astype(int), z["pred"]
    aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs)) if aucs else None


def oof_file(job_dir: Path, tag: str) -> Path:
    return job_dir / tag / f"raptor_oof_{tag}_fold{G.PROTOCOL['fold']}.npz"


# ------------------------------------------------------------------ observe

def result_files(tag: str) -> list[str]:
    f = G.PROTOCOL["fold"]
    return [f"{tag}/raptor_ft_{tag}.json", f"{tag}/raptor_oof_{tag}_fold{f}.npz",
            f"{tag}/raptor_oof_{tag}_swa_fold{f}.npz", f"{tag}/raptor_holdout_{tag}.npz",
            f"{tag}/train.log"]


def download_results(slug: str, dest: Path, tags: list[str]) -> bool:
    """Fetch only the scoring files by name (whole-dataset downloads fail silently on large sets)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["STATUS", "job.log"] + [f for t in tags for f in result_files(t)]:
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", slug, "-f", name, "-p", str((dest / name).parent)],
                       capture_output=True, text=True)
    return (dest / "STATUS").exists()


def observe(state: dict, journal: Journal) -> None:
    a = state["active"]
    if not a:
        return
    try:
        pod = runpod.get_pod(a["pod"])
    except Exception:
        pod = None
    elapsed_h = (time.time() - a["launched"]) / 3600
    if pod is not None and pod.get("desiredStatus") == "RUNNING":
        if elapsed_h > 2 * a["est_hours"] + 1:
            runpod.terminate_pod(a["pod"])
            log("error", f"pod {a['pod']} exceeded {2 * a['est_hours'] + 1:.1f} h; terminated",
                component="loop", pod=a["pod"])
        else:
            return
    elif pod is not None:
        runpod.terminate_pod(a["pod"])  # stopped after a failed upload; stop volume billing

    if not a.get("cost_logged"):
        log("cost", f"pod {a['pod']} ~{elapsed_h:.2f} h at ${a['price_hr']}/h",
            cost_usd=elapsed_h * a["price_hr"], pod=a["pod"])
        a["cost_logged"], a["gpu_hours"] = True, elapsed_h
    dest = RESULTS / a["job"]
    if not download_results(a["result"], dest, a["nodes"]):
        a["tries"] = a.get("tries", 0) + 1
        if a["tries"] < DOWNLOAD_TRIES:
            return
    if not any(oof_file(dest, n).exists() for n in a["nodes"]) and a.get("infra_retries", 0) < INFRA_RETRIES:
        status = (dest / "STATUS").read_text().strip() if (dest / "STATUS").exists() else "no STATUS"
        log("error", f"job {a['job']} produced no results ({status}); relaunching as infrastructure failure",
            component="loop", pod=a["pod"])
        retries = a.get("infra_retries", 0) + 1
        state["active"] = None
        start_train(state, journal, a["nodes"][0], attempt=retries)
        state["active"]["infra_retries"] = retries
        return
    holdout = json.loads(HOLDOUT.read_text()) if HOLDOUT.exists() else {}
    for nid in a["nodes"]:
        oof = oof_file(dest, nid)
        meta_f = dest / nid / f"raptor_ft_{nid}.json"
        if not oof.exists() or not meta_f.exists():
            journal.record(nid, error=f"no results in {dest} (see job.log)", gpu_hours=a["gpu_hours"])
            log("gate", f"node {nid} FAILED: missing results", node=nid)
            continue
        z = np.load(oof, allow_pickle=True)
        proxy = proxy_auc(z["pred"], z["ids"])
        gold_dev = float(json.loads(meta_f.read_text())["best_gold_auc"])
        parent = journal.nodes[journal.nodes[nid].parent]
        err = ""
        if parent.gold_auc is not None and gold_dev < parent.gold_auc - VETO_DROP:
            err = f"veto: gold_dev {gold_dev:.3f} < parent {parent.gold_auc:.3f} - {VETO_DROP}"
        reward = journal.record(nid, proxy_auc=None if err else proxy, gold_auc=gold_dev,
                                gpu_hours=a["gpu_hours"], error=err)
        journal.nodes[nid].proxy_auc = proxy
        swa = dest / nid / f"raptor_oof_{nid}_swa_fold{G.PROTOCOL['fold']}.npz"
        holdout[nid] = {"holdout_auc": gold_macro(dest / nid / f"raptor_holdout_{nid}.npz"),
                        "swa_proxy": proxy_auc(*[np.load(swa, allow_pickle=True)[k] for k in ("pred", "ids")])
                        if swa.exists() else None}
        log("gate", f"node {nid}: proxy {proxy:.4f} gold_dev {gold_dev:.4f} reward {reward:+.0f}"
            + (f" {err}" if err else ""), node=nid)
    HOLDOUT.write_text(json.dumps(holdout, indent=1))
    state["active"] = None
    dream_blends(journal, dest.parent)


# ------------------------------------------------------------------ dream

def _oof_for(node_id: str, root: Path) -> tuple[np.ndarray, np.ndarray] | None:
    for f in root.glob(f"*/{node_id}/raptor_oof_{node_id}_fold{G.PROTOCOL['fold']}.npz"):
        z = np.load(f, allow_pickle=True)
        return z["pred"], z["ids"].astype(str)
    return None


def dream_blends(journal: Journal, root: Path, top: int = 4) -> None:
    """Free replay: rank-mean blends of pairs among the best online nodes."""
    online = sorted((n for n in journal.valid_nodes() if n.kind == "online"),
                    key=lambda n: n.proxy_auc, reverse=True)[:top]
    for i, a in enumerate(online):
        for b in online[i + 1:]:
            g = {"blend": sorted([a.id, b.id])}
            if genome_hash(g) in journal.nodes:
                continue
            pa, pb = _oof_for(a.id, root), _oof_for(b.id, root)
            if pa is None or pb is None or not np.array_equal(pa[1], pb[1]):
                continue
            node = journal.add(a.id, g, plan=f"dream: rank blend {a.id}+{b.id}", kind="dream",
                               refs=[b.id], operator="aggregate")
            proxy = proxy_auc(rank_blend([pa[0], pb[0]]), pa[1])
            journal.record(node.id, proxy_auc=proxy)
            log("dream", f"blend {a.id}+{b.id}: proxy {proxy:.4f} (singles {a.proxy_auc:.4f}/{b.proxy_auc:.4f})",
                node=node.id)


# ------------------------------------------------------------------ plan + act

def _unused_label_sets(journal: Journal, state: dict) -> list[str]:
    """Registered soft-label files that no finished/pending online node has trained on yet."""
    used = {n.genome.get("labels") for n in journal.nodes.values()
            if n.kind == "online" and n.genome.get("labels")}
    out = []
    for r in state.get("relabels") or []:
        if r.get("registered") and r.get("name") and r["name"] not in used:
            out.append(r["name"])
    for name in G.label_registry():
        if name not in used and name not in ("dread", "blend_v2") and name not in out:
            # only auto-queue registered relabels, not every registry entry
            pass
    return out


def _force_train_labels(state: dict, journal: Journal, parent: Node, labels: str, plan: str) -> None:
    g = dict(parent.genome or G.BASE)
    g["labels"] = labels
    if genome_hash(g) in journal.nodes:
        log("gate", f"skip force-train {labels}: genome already exists")
        return
    node = journal.add(parent.id, g, plan=plan, operator="primary")
    _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(g))


def start_train(state: dict, journal: Journal, node_id: str, attempt: int = 0) -> None:
    g = journal.nodes[node_id].genome
    labels_file, args = G.to_cli(g, "/workspace/bundle")
    hours = G.est_hours(g)
    job = f"n-{node_id}" + (f"-r{attempt}" if attempt else "")
    out = launcher.launch(job, [(node_id, labels_file, args)], hours)
    state["active"] = {"job": job, "pod": out["pod"], "nodes": [node_id], "result": out["result"],
                       "launched": time.time(), "est_hours": hours,
                       "price_hr": float(out.get("costPerHr") or launcher.PRICE_HR)}


def run_relabel(state: dict, prop: dict) -> None:
    res = labeler.run(prop["labels"], prop["rule"], prop["base"])
    state["relabels"].append({"name": res["name"], "registered": res["registered"], "kept": res["kept"]})
    log("data", f"relabel {res['name']} ({prop['rule']}, {prop['labels']}): kept {res['kept']}, "
        f"{res['cells_changed']} cells changed", cost_usd=res["cost_usd"], audit=res["audit"])


def plan(state: dict, journal: Journal, rng: random.Random, llm=None) -> None:
    budget_left = BUDGET_USD - spent(read())
    online = [n for n in journal.nodes.values() if n.kind == "online"]
    if not online:
        node = journal.add(ROOT_ID, dict(G.BASE), plan="baseline: public CoAtNet recipe, dread labels, fold-0 protocol")
        return _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(node.genome))

    # Do not start a second train while an online node is still pending (in-flight / uploading).
    if any(n.kind == "online" and n.status == "pending" for n in journal.nodes.values()):
        log("note", "skip plan: an online node is still pending")
        return
    # Never idle while a registered relabel has not been trained: that is free expected gain.
    unused = _unused_label_sets(journal, state)
    best = journal.best()
    online_valid = [n for n in journal.nodes.values() if n.kind == "online" and n.valid]
    parent = best if best and best.kind == "online" else (
        max(online_valid, key=lambda n: n.proxy_auc or 0.0) if online_valid else None)
    if unused and parent is not None:
        labels = unused[0]
        log("plan", f"auto-queue train on unused labels {labels} (campaign target proxy>={PROXY_TARGET})")
        return _force_train_labels(
            state, journal, parent, labels,
            plan=f"train registered relabel {labels} on best parent (target {PROXY_TARGET})")

    parent, mode = select(journal, min(1.0, 1 - budget_left / BUDGET_USD), Schedule(), rng)
    operator, refs = choose_operator(journal, parent, Schedule())
    kw = {"llm": llm} if llm else {}
    p = propose(journal, parent, operator, refs, budget_left=budget_left, cold_start=COLD_START,
                target=PROXY_TARGET, **kw)
    log("plan", f"[{mode}/{operator}] from {parent.id}: {p.type} - {p.plan or p.why}", cost_usd=p.cost_usd,
        attempts=p.attempts)
    if p.type == "stop":
        # Fall back to a cheap allowed train (k / epochs / head_lr) instead of parking for hours.
        fallback = _fallback_genome(parent.genome or G.BASE, journal)
        if fallback is not None:
            log("plan", f"stop overridden: fallback train {G.diff(parent.genome or G.BASE, fallback)}")
            node = journal.add(parent.id, fallback, plan="fallback after planner stop/ban wall",
                               operator="primary")
            return _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(fallback))
        state["next_plan_after"] = time.time() + STOP_COOLDOWN_S
        log("note", f"planner stop; retry in {STOP_COOLDOWN_S // 60} min (target {PROXY_TARGET} not met)")
        return
    if p.type == "relabel":
        if len(state["relabels"]) >= MAX_RELABELS:
            log("gate", "relabel rejected: MAX_RELABELS reached")
            state["next_plan_after"] = time.time() + 600
            return
        n, est = labeler.estimate(p.labels, p.rule)
        return _gate_or_start(state, journal, {"type": "relabel", "labels": p.labels, "rule": p.rule,
                                               "base": p.base, "calls": n}, None, est)
    node = journal.add(parent.id, p.genome, plan=p.plan, refs=refs, operator=operator)
    _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(p.genome))


def _fallback_genome(parent: dict, journal: Journal) -> dict | None:
    """Deterministic one-field diffs when the LLM only proposes banned levers."""
    for field, values in (("k", [16, 8]), ("epochs", [20, 12]), ("head_lr", [2e-3, 5e-4]),
                          ("bb_lr", [5e-5, 2e-5]), ("labels", ["blend_v2"])):
        for v in values:
            g = dict(parent)
            if g.get(field) == v:
                continue
            g[field] = v
            if G.validate(g) or genome_hash(g) in journal.nodes:
                continue
            return g
    return None


def _gate_or_start(state, journal, item: dict, hours: float | None, est_usd: float | None = None) -> None:
    est = est_usd if est_usd is not None else hours * launcher.PRICE_HR
    item["est_usd"] = round(est, 2)
    if est > APPROVAL_USD:
        state["pending"] = item
        log("gate", f"approval needed: {item} (est ${est:.2f} > ${APPROVAL_USD:.0f})")
        return
    _execute(state, journal, item)


def _execute(state, journal, item: dict) -> None:
    if item["type"] == "train":
        start_train(state, journal, item["node"])
    else:
        run_relabel(state, item)


# ------------------------------------------------------------------ tick / run

def _foreign_pods() -> bool:
    """A hand-launched pod is still running; the loop waits so only one pod bills at a time."""
    try:
        return any(p.get("desiredStatus") == "RUNNING" for p in runpod.list_pods())
    except Exception:
        return True

def tick(llm=None, rng: random.Random | None = None) -> dict:
    state, journal = load_state(), load_journal()
    rng = rng or random.Random(int(time.time()))
    try:
        observe(state, journal)
        if state["active"] or state["paused"] or _foreign_pods():
            return state
        if state["pending"]:
            if state["pending"].get("approved"):
                item, state["pending"] = state["pending"], None
                _execute(state, journal, item)
            return state
        if time.time() >= state.get("next_plan_after", 0):
            plan(state, journal, rng, llm)
    finally:
        state["heartbeat"] = time.time()
        journal.save()
        save_state(state)
    return state


def run(interval: int) -> None:
    log("note", f"loop started (interval {interval}s)")
    while not STOP_FILE.exists():
        try:
            tick()
        except Exception as e:
            log("error", f"loop tick failed: {type(e).__name__}: {str(e)[:300]}", component="loop",
                trace=traceback.format_exc()[-800:])
        time.sleep(interval)
    log("note", "loop stopped by STOP file")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "tick", "run", "approve", "reject", "pause", "resume"])
    ap.add_argument("--interval", type=int, default=600)
    a = ap.parse_args()
    s = load_state()
    if a.cmd == "status":
        j = load_journal()
        print(json.dumps({k: s[k] for k in ("paused", "active", "pending", "relabels")}, indent=1, default=str))
        for n in sorted(j.nodes.values(), key=lambda n: n.created):
            print(f"{n.id:12s} {n.kind:6s} {n.status:8s} proxy={n.proxy_auc} gold_dev={n.gold_auc} {n.plan[:70]}")
        print(f"spent ${spent(read()):.2f} of ${BUDGET_USD:.0f}")
    elif a.cmd == "tick":
        tick()
    elif a.cmd == "run":
        run(a.interval)
    elif a.cmd in ("approve", "reject"):
        if not s["pending"]:
            raise SystemExit("nothing pending")
        log("gate", f"user {a.cmd}d {s['pending']}")
        if a.cmd == "approve":
            s["pending"]["approved"] = True
        else:
            if s["pending"]["type"] == "train":
                j = load_journal()
                j.record(s["pending"]["node"], error="rejected by user")
                j.save()
            s["pending"] = None
        save_state(s)
    else:
        s["paused"] = a.cmd == "pause"
        save_state(s)
        log("note", f"loop {a.cmd}d by user")


if __name__ == "__main__":
    main()
