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
from rsna_knee.agent import labeler, launcher, members, runpod, strategist
from rsna_knee.agent.activity import BUDGET_USD, JOURNAL, RUNS, log, read, spent
from rsna_knee.agent.dream import greedy_rank_blend, rank_blend
from rsna_knee.agent.journal import ROOT_ID, Journal, Node, genome_hash
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
# 2026-09-29: submissions come from full-data members; fold-0 ImageNet trains only re-measure
# what member per-finding AUCs already show. Idle ticks go to the strategist instead.
FOLD0_TRAINS = False
VETO_DROP = 0.04
# Cap on *successful unique* LLM relabel jobs (budget guard). Not a science threshold.
# Rejection must fall through to a menu train — never idle the campaign.
MAX_RELABELS = strategist.MAX_RELABELS
STOP_COOLDOWN_S = 15 * 60  # only when the menu is fully exhausted
# Local search bar only. True goal = Kaggle submit/LB macro AUC ≥ 0.958 (DECISIONS 2026-09-25).
PROXY_TARGET = 0.958
DOWNLOAD_TRIES = 8
KAGGLE_TIMEOUT_S = 180
INFRA_RETRIES = 2
# Kill hung/dead pods early instead of billing until the API notices.
PROXY_MISS_LIMIT = 2          # consecutive observe ticks with unreachable proxy
HEARTBEAT_STALE_S = 20 * 60   # no HEARTBEAT update → treat as dead
HEALTH_GRACE_S = 12 * 60      # skip probe during boot/install/corpus download
BOOT_STALL_S = 90 * 60        # installing/downloading/smoke with no progress past this → kill
DENSE80_BOOT_STALL_S = 5 * 3600  # ~40 GB shard pull; allow several hours before kill


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
        try:
            # A download of a file that never appears hung the loop for 15+ min on 2026-10-01.
            subprocess.run(["kaggle", "datasets", "download", slug, "-f", name, "-p", str((dest / name).parent)],
                           capture_output=True, text=True, timeout=KAGGLE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log("note", f"kaggle download timed out: {slug} {name}")
    return (dest / "STATUS").exists()


def _probe_pod_progress(pod_id: str) -> dict:
    """Read STATUS / HEARTBEAT from the pod HTTP proxy. Never raises."""
    import requests
    base = runpod.proxy_url(pod_id)
    out: dict = {"reachable": False, "status": None, "heartbeat_age_s": None}
    try:
        r = requests.get(f"{base}/STATUS", timeout=12)
        if r.ok:
            out["reachable"] = True
            out["status"] = r.text.strip()[:200]
    except Exception:
        return out
    try:
        r = requests.get(f"{base}/HEARTBEAT", timeout=8)
        if r.ok:
            ts = int(r.text.strip().split()[0])
            out["heartbeat_age_s"] = max(0.0, time.time() - ts)
    except Exception:
        pass
    return out


def _running_pod_unhealthy(a: dict) -> str | None:
    """Return a reason if the RUNNING pod looks dead/hung; else None.

    Legacy pods (no HEARTBEAT) are only flagged when the proxy itself is down —
    STATUS does not change for hours during a healthy train.
    """
    if time.time() - a["launched"] < HEALTH_GRACE_S:
        return None
    probe = _probe_pod_progress(a["pod"])
    if not probe["reachable"]:
        misses = int(a.get("proxy_misses", 0)) + 1
        a["proxy_misses"] = misses
        if misses >= PROXY_MISS_LIMIT:
            return f"proxy unreachable {misses} ticks (dead host / silent terminate)"
        log("note", f"pod {a['pod']} proxy miss {misses}/{PROXY_MISS_LIMIT}", pod=a["pod"])
        return None
    a["proxy_misses"] = 0
    st = (probe.get("status") or "").lower()
    age_s = time.time() - a["launched"]
    # Boot phases are allowed to take a while (dense80 ~40 GB), but not forever.
    if st.startswith(("installing", "downloading", "smoke")):
        limit = DENSE80_BOOT_STALL_S if "dense80" in st else BOOT_STALL_S
        if age_s > limit:
            return f"boot STATUS stuck on {st!r} for {age_s / 60:.0f} min (limit {limit / 60:.0f} min)"
        return None
    hb = probe.get("heartbeat_age_s")
    if hb is None:
        return None  # pre-heartbeat bundles: proxy up is enough
    if hb > HEARTBEAT_STALE_S:
        return f"HEARTBEAT stale ({hb / 60:.0f} min); job hung or crashed"
    return None


def observe(state: dict, journal: Journal) -> None:
    a = state["active"]
    if not a:
        return
    try:
        pod = runpod.get_pod(a["pod"])
    except Exception:
        pod = None
    elapsed_h = (time.time() - a["launched"]) / 3600
    # Phantom / vanished pod (create claimed success then 404, or never rented): relaunch.
    if pod is None and not a.get("cost_logged") and elapsed_h < 0.5:
        log("error", f"pod {a['pod']} gone before billing (phantom/capacity); relaunching",
            component="loop", pod=a["pod"])
        retries = a.get("infra_retries", 0) + 1
        state["active"] = None
        if retries <= INFRA_RETRIES:
            start_train(state, journal, a["nodes"][0], attempt=retries)
            if state.get("active"):
                state["active"]["infra_retries"] = retries
        else:
            for nid in a["nodes"]:
                journal.record(nid, error="pod vanished / no capacity after retries", gpu_hours=0.0)
                log("gate", f"node {nid} FAILED: no capacity", node=nid)
        return
    if pod is not None and pod.get("desiredStatus") == "RUNNING":
        if elapsed_h > 2 * a["est_hours"] + 1:
            runpod.terminate_pod(a["pod"])
            log("error", f"pod {a['pod']} exceeded {2 * a['est_hours'] + 1:.1f} h; terminated",
                component="loop", pod=a["pod"])
        else:
            why = _running_pod_unhealthy(a)
            if why is None:
                return
            log("error", f"pod {a['pod']} unhealthy: {why}; terminating to stop burn",
                component="loop", pod=a["pod"])
            try:
                runpod.terminate_pod(a["pod"])
            except Exception:
                pass
            # No upload expected from a dead host — skip download thrash, infra-relaunch ASAP.
            a["tries"] = DOWNLOAD_TRIES
            # Fall through: log cost, try download once, infra-relaunch if empty.
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


def dream_blends(journal: Journal, root: Path, top: int = 4) -> int:
    """Free replay: rank-mean + greedy per-label blends among best online nodes. Returns # new dreams."""
    online = sorted((n for n in journal.valid_nodes() if n.kind == "online"),
                    key=lambda n: n.proxy_auc, reverse=True)[:top]
    created = 0
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
            created += 1
            log("dream", f"blend {a.id}+{b.id}: proxy {proxy:.4f} (singles {a.proxy_auc:.4f}/{b.proxy_auc:.4f})",
                node=node.id)
    # Greedy per-label weights over top online OOF (CPU) — often beats equal rank-mean.
    packs: list[tuple[Node, np.ndarray, np.ndarray]] = []
    for n in online:
        o = _oof_for(n.id, root)
        if o is not None:
            packs.append((n, o[0], o[1]))
    if len(packs) >= 2:
        ids0 = packs[0][2]
        if all(np.array_equal(ids0, p[2]) for p in packs[1:]):
            ref, mask = reference()
            keep = pd.Index(ids0.astype(str)).isin(ref.index)
            if int(keep.sum()) >= 50:
                y = ref.loc[ids0[keep]].to_numpy().astype(float)
                preds = [p[1][keep] for p in packs]
                g = {"blend": sorted(p[0].id for p in packs), "method": "greedy_rank"}
                if genome_hash(g) not in journal.nodes:
                    try:
                        blend, _, _ = greedy_rank_blend(preds, y)
                    except Exception as e:  # noqa: BLE001
                        log("note", f"greedy dream skipped: {type(e).__name__}: {e}")
                    else:
                        full = np.zeros((len(ids0), len(LABEL_COLS)), dtype=float)
                        full[keep] = blend
                        node = journal.add(packs[0][0].id, g,
                                           plan=f"dream: greedy rank blend {[p[0].id for p in packs]}",
                                           kind="dream",
                                           refs=[p[0].id for p in packs[1:]], operator="aggregate")
                        proxy = proxy_auc(full, ids0)
                        journal.record(node.id, proxy_auc=proxy)
                        created += 1
                        best_single = max(p[0].proxy_auc or 0 for p in packs)
                        log("dream", f"greedy blend {[p[0].id for p in packs]}: "
                            f"proxy {proxy:.4f} (best single {best_single:.4f})", node=node.id)
    return created


# ------------------------------------------------------------------ plan + act

def _relabel_count(state: dict) -> int:
    """Unique successful (registered) relabel jobs — duplicates / failed audits do not burn the cap."""
    return len({r["name"] for r in state.get("relabels") or []
                if r.get("registered") and r.get("name")})


def _train_genome(node: Node | None) -> dict:
    """Strip dream-only keys; snap historical parents (k=12/16) onto the live VRAM-safe MENU."""
    raw = dict((node.genome if node else None) or G.BASE)
    return G.clamp(raw)


def _online_seed(journal: Journal, node: Node | None = None) -> Node | None:
    """Dream elites have no train genome; expand from best valid online (prefer dense80)."""
    online = [n for n in journal.nodes.values() if n.kind == "online" and n.valid
              and "corpus" in (n.genome or {})]
    if not online:
        return None
    if node is not None and node.kind == "online" and node.valid and "corpus" in (node.genome or {}):
        return node
    refs: list[str] = []
    if node is not None:
        refs.extend(node.refs or [])
        blend = (node.genome or {}).get("blend") if isinstance(node.genome, dict) else None
        if isinstance(blend, list):
            refs.extend(str(x) for x in blend)
    ranked_refs = [journal.nodes[r] for r in refs
                   if r in journal.nodes and journal.nodes[r] in online]
    pool = ranked_refs or online
    dense = [n for n in pool if n.genome.get("corpus") == "dense80"]
    return max(dense or pool, key=lambda n: n.proxy_auc or 0.0)


def _unused_label_sets(journal: Journal, state: dict, *, corpus: str | None = None) -> list[str]:
    """Label files not yet trained (done/pending) on the target corpus (default: dense80 if enabled)."""
    want = corpus or ("dense80" if G.CORPORA.get("dense80") else None)
    used = {n.genome.get("labels") for n in journal.nodes.values()
            if n.kind == "online" and n.status in ("done", "pending")
            and n.genome.get("labels")
            and (want is None or n.genome.get("corpus") == want)}
    out: list[str] = []
    for r in state.get("relabels") or []:
        if r.get("registered") and r.get("name") and r["name"] not in used and r["name"] not in out:
            out.append(r["name"])
    for name in ("dread", "blend_v2"):
        if name in G.label_registry() and name not in used and name not in out:
            out.append(name)
    return out


def _force_train_labels(state: dict, journal: Journal, parent: Node, labels: str, plan: str) -> None:
    g = _train_genome(parent)
    g["labels"] = labels
    existing = journal.nodes.get(genome_hash(g))
    if existing is not None:
        if existing.status == "failed":
            existing.status, existing.error, existing.plan = "pending", "", plan
            log("plan", f"retry failed labels={labels} node {existing.id}")
            return start_train(state, journal, existing.id, attempt=1)
        log("gate", f"skip force-train {labels}: genome already exists ({existing.status})")
        return
    node = journal.add(parent.id, g, plan=plan, operator="primary")
    _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(g))


def _force_train_corpus(state: dict, journal: Journal, parent, corpus: str, plan: str) -> None:
    """One-field corpus switch (primary); used to kick dense80 submit-geometry trains."""
    g = _train_genome(parent)
    g["corpus"] = corpus
    if errs := G.validate(g):
        log("gate", f"skip force-train corpus={corpus}: {errs}")
        return
    existing = journal.nodes.get(genome_hash(g))
    if existing is not None:
        if existing.status == "failed":
            existing.status, existing.error, existing.plan = "pending", "", plan
            log("plan", f"retry failed corpus={corpus} node {existing.id}")
            return start_train(state, journal, existing.id, attempt=1)
        log("gate", f"skip force-train corpus={corpus}: genome already exists ({existing.status})")
        return
    node = journal.add(parent.id, g, plan=plan, operator="primary")
    _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(g))


def _force_fallback_train(state: dict, journal: Journal, seed: Node, why: str) -> bool:
    """Queue the next unused menu genome from an online seed. Returns True if a train was queued."""
    base = _train_genome(seed)
    fallback = _fallback_genome(base, journal)
    if fallback is None:
        return False
    log("plan", f"{why}: auto-queue train {G.diff(base, fallback)}")
    node = journal.add(seed.id, fallback, plan=f"{why}: {G.diff(base, fallback)}", operator="primary")
    _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(fallback))
    return True


def start_train(state: dict, journal: Journal, node_id: str, attempt: int = 0) -> None:
    node = journal.nodes[node_id]
    g = dict(node.genome or {})
    if errs := G.validate(g):
        # Historical pending nodes may still carry k=12/16 — clamp once, then re-validate.
        g = G.clamp(g)
        node.genome = g
        if errs2 := G.validate(g):
            node.status, node.error = "failed", f"illegal genome: {errs2}"
            log("gate", f"node {node_id} FAILED: illegal genome {errs2}")
            return
        log("note", f"clamped genome on {node_id} before launch: {errs} → k={g.get('k')} k_eval={g.get('k_eval')}")
    labels_file, args = G.to_cli(g, "/workspace/bundle")
    hours = G.est_hours(g)
    job = f"n-{node_id}" + (f"-r{attempt}" if attempt else "")
    corpus = g.get("corpus", "raptor44")
    out = launcher.launch(job, [(node_id, labels_file, args)], hours, corpus=corpus)
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

    # Free CPU replay before any GPU rent (blends / greedy weights over saved OOF logits).
    n_dreams = dream_blends(journal, RESULTS)
    if n_dreams:
        log("dream", f"pre-plan replay created {n_dreams} dream node(s); GPU only for label/lr/epoch diffs")

    # Orphaned pending online node (launch claimed then vanished): resume that train.
    orphans = [n for n in journal.nodes.values()
               if n.kind == "online" and n.status == "pending"]
    if orphans and not state.get("active"):
        if _foreign_pods():
            log("note", "skip orphan resume: another pod is already RUNNING")
            return
        n = orphans[0]
        if G.validate(n.genome or {}):
            clamped = G.clamp(n.genome or {})
            if G.validate(clamped):
                n.status, n.error = "failed", f"illegal genome under VRAM ban: {G.validate(n.genome or {})}"
                log("gate", f"drop orphan {n.id}: {n.error}")
                return
            n.genome = clamped
            log("note", f"clamped orphan {n.id} to k={clamped.get('k')} before resume")
        log("plan", f"resume orphaned pending node {n.id}")
        start_train(state, journal, n.id, attempt=int((state.get("_orphan_retries") or 0)) + 1)
        state["_orphan_retries"] = int(state.get("_orphan_retries") or 0) + 1
        return
    # Do not start a second train while an online node is still pending *and* a job is active.
    if orphans and state.get("active"):
        log("note", "skip plan: an online node is still pending")
        return

    if _foreign_pods():
        log("note", "skip plan: foreign/hand-launched pod still RUNNING")
        return

    seed = _online_seed(journal)
    if seed is None:
        log("gate", "no valid online seed; cannot plan a train")
        return

    # DeepSeek owns the idle tick (coder-first + launch). Host only safety-nets if it still stops.
    parent, mode = select(journal, min(1.0, 1 - budget_left / BUDGET_USD), Schedule(), rng)
    seed = _online_seed(journal, parent) or seed
    operator, refs = choose_operator(journal, parent, Schedule())
    kw = {"llm": llm} if llm else {}
    if not llm:
        kw["status_fn"] = lambda: {
            "active": state.get("active"),
            "pending": state.get("pending"),
            "foreign_running": _foreign_pods(),
            "relabel_slots_left": max(0, MAX_RELABELS - _relabel_count(state)),
            "unused_dense80_labels": _unused_label_sets(journal, state),
            "coder": "CODER-FIRST: write_repo_file on agent/runpod/tests then run_agent_tests then launch",
            "policy": ("VRAM: k=8 k_eval=24 only; arch=coatnet384; dream/replay already ran this tick; "
                       "lr/wd/head_lr/epochs diffs are CLOSED; launch label/corpus levers or relabel weak findings"),
        }
    p = propose(journal, seed, operator, refs, budget_left=budget_left, cold_start=COLD_START,
                target=PROXY_TARGET, **kw)
    log("plan", f"[{mode}/{operator}] from {seed.id} (selected {parent.id}): {p.type} - {p.plan or p.why}",
        cost_usd=p.cost_usd, attempts=p.attempts, tools=len(getattr(p, "tool_trace", []) or []))
    if p.type == "stop":
        log("note", "DeepSeek stop/no-launch after retry; host safety-net menu train")
        if _force_fallback_train(state, journal, seed, "safety-net after DeepSeek"):
            return
        state["next_plan_after"] = time.time() + STOP_COOLDOWN_S
        log("note", f"planner stop; menu exhausted; retry in {STOP_COOLDOWN_S // 60} min "
            f"(target {PROXY_TARGET} not met)")
        return
    if p.type == "relabel":
        if _relabel_count(state) >= MAX_RELABELS:
            log("gate", "relabel rejected: MAX_RELABELS reached; forcing menu train")
            if _force_fallback_train(state, journal, seed, "max-relabel fallback"):
                return
            state["next_plan_after"] = time.time() + STOP_COOLDOWN_S
            return
        n, est = labeler.estimate(p.labels, p.rule)
        return _gate_or_start(state, journal, {"type": "relabel", "labels": p.labels, "rule": p.rule,
                                               "base": p.base, "calls": n}, None, est)
    node = journal.add(seed.id, p.genome, plan=p.plan, refs=refs, operator=operator)
    _gate_or_start(state, journal, {"type": "train", "node": node.id}, G.est_hours(p.genome))


def _fallback_genome(parent: dict, journal: Journal) -> dict | None:
    """Next unused one-field menu diff from a full online genome (never a dream blend)."""
    base = G.clamp(parent)
    label_opts = [x for x in G.label_registry() if x != base.get("labels")]
    # Prefer unused labels + lr/epoch (GPU-needed). Never open k / k_eval / arch (VRAM + settled).
    # lr / wd / head_lr / epochs are a closed axis (DECISIONS 2026-09-29): labels / corpus only.
    for field, values in (("corpus", ["dense80"]), ("labels", label_opts)):
        for v in values:
            g = dict(base)
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
        # Full-data submission members take the next free pod slot before any new fold-0 train.
        try:
            members_busy = members.tick(loop_active=bool(state["active"]))
        except Exception as e:  # noqa: BLE001 — never let the member track stall the loop
            members_busy = False
            log("error", f"members tick failed: {type(e).__name__}: {str(e)[:200]}", component="members")
        if state["active"] or state["paused"] or members_busy or _foreign_pods():
            # GPU busy: DeepSeek strategist does CPU/API work (labels, member queue, kernels).
            if not state["paused"] and llm is None:
                try:
                    strategist.research(state, journal, RESULTS)
                except Exception as e:  # noqa: BLE001 — never stall pod monitoring
                    log("error", f"strategist failed: {type(e).__name__}: {str(e)[:200]}",
                        component="strategist", trace=traceback.format_exc()[-600:])
            return state
        if state["pending"]:
            if state["pending"].get("approved"):
                item, state["pending"] = state["pending"], None
                _execute(state, journal, item)
            return state
        if time.time() >= state.get("next_plan_after", 0):
            if FOLD0_TRAINS or llm is not None:
                plan(state, journal, rng, llm)
            else:
                # Member-track era: idle GPU → strategist queues the next full-data member job.
                strategist.research(state, journal, RESULTS)
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
