"""Submission members: full-data retrains of scored genomes, several arms on ONE pod.

Fold-0 online nodes are the search compass (train on 80%, score OOF). What goes into a
Kaggle submission should be trained on every non-expert study, in several seeds / label
sets, so the ensemble leg is stronger and more diverse than one fold-0 model. Running all
arms on one pod pays the ~40 GB dense80 pull once instead of once per model.

    python -m rsna_knee.agent.members queue --node 5bc0a0f45d75 --seeds 42,1337 --labels dread
    python -m rsna_knee.agent.members status

The queue is launched by the loop tick (``loop.tick`` → ``members.tick``) as soon as no pod
is running, so the one-pod rule holds. While a member job is queued or active the loop does
not plan new fold-0 trains. Results land in ``outputs/runpod_results/<job>/`` and in
``agent/runs/members.json``; weights stay in the private Kaggle result dataset, which the
ensemble kernel (``scripts/build_ensemble_kernel.py``) attaches directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from rsna_knee.agent import genome as G
from rsna_knee.agent import launcher, runpod
from rsna_knee.agent.activity import RUNS, log

MEMBERS = RUNS / "members.json"
RESULTS = G.REPO / "outputs" / "runpod_results"
JOURNAL_FILE = RUNS / "journal.json"
FULL_DATA_FACTOR = 1.25   # 4349 vs 3479 training studies
APPROVAL_USD = 10.0
# Public v5 full-SWA CoAtNet (CC0, same arch/head; 486/486 tensors load). It was trained on all
# studies, so fold-0 proxy would be leaky — warm starts live on the member track only and are
# judged on gold_dev / holdout / LB.
INITS = {"v5": "/workspace/input/init/raptor_ft_coatnet_v5_full_swa.pt"}
DOWNLOAD_TRIES = 8


def load() -> dict:
    if MEMBERS.exists():
        return json.loads(MEMBERS.read_text())
    return {"queue": [], "active": None, "done": {}}


def save(m: dict) -> None:
    MEMBERS.write_text(json.dumps(m, indent=1, default=str))


def full_cli(genome: dict, seed: int, bundle_dir: str = "/workspace/bundle",
             init: str | None = None, epochs: int | None = None) -> tuple[str, str]:
    """Trainer args for a full-data arm: the fold-0 protocol args minus the fold hook.

    The expert dev/holdout split is kept so checkpoint selection and the holdout read stay
    identical to the online protocol (experts are never trained on either way).
    """
    labels_file, args = G.to_cli(genome, bundle_dir)
    args = re.sub(r"--folds \d+ --fold \d+ --fold_file \S+ ", "", args)
    args = re.sub(r"--seed \d+", f"--seed {int(seed)}", args)
    if epochs:
        args = re.sub(r"--epochs \d+", f"--epochs {int(epochs)}", args)
    if init:
        args += f" --init_model {INITS[init]}"
    assert "--fold " not in args and "--folds" not in args, args
    return labels_file, args


def est_hours(genome: dict, n_arms: int) -> float:
    return (G.est_hours(genome) - 0.5) * FULL_DATA_FACTOR * n_arms + 1.0


def member_tag(genome: dict, seed: int, init: str | None = None) -> str:
    key = {**genome, "seed": seed, "fit": "full", **({"init": init} if init else {})}
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode())
    return "m" + h.hexdigest()[:11]


def _node_genome(node_id: str) -> dict:
    for n in json.loads(JOURNAL_FILE.read_text()):
        if n["id"] == node_id:
            return dict(n["genome"])
    raise SystemExit(f"node {node_id} not in journal")


def queue(node_id: str, seeds: list[int], labels: list[str] | None = None,
          m: dict | None = None, init: str | None = None, epochs: int | None = None) -> dict:
    """Queue one pod job whose arms are (seed × label set) full-data retrains of ``node_id``."""
    m = m if m is not None else load()
    if init and init not in INITS:
        raise SystemExit(f"init {init!r} not in {sorted(INITS)}")
    base = G.clamp(_node_genome(node_id))
    if epochs:
        base["epochs"] = int(epochs)
    arms, seen = [], {t for j in m["queue"] for t in (a["tag"] for a in j["arms"])} | set(m["done"])
    for lab in labels or [base["labels"]]:
        g = dict(base, labels=lab)
        if errs := [e for e in G.validate(g) if not (epochs and e.startswith("epochs="))]:
            raise SystemExit(f"illegal genome for labels={lab}: {errs}")
        for s in seeds:
            tag = member_tag(g, s, init)
            if tag in seen:
                continue
            seen.add(tag)
            arms.append({"tag": tag, "genome": g, "seed": int(s), "parent": node_id, "init": init})
    if not arms:
        raise SystemExit("every requested member already exists or is queued")
    hours = est_hours(base, len(arms))
    job = {"job": "m-" + hashlib.sha1("".join(a["tag"] for a in arms).encode()).hexdigest()[:10],
           "arms": arms, "est_hours": round(hours, 2), "est_usd": round(hours * launcher.PRICE_HR, 2),
           "corpus": base["corpus"], "approved": hours * launcher.PRICE_HR <= APPROVAL_USD,
           "queued": time.time()}
    m["queue"].append(job)
    save(m)
    log("plan", f"members queued {job['job']}: {len(arms)} full-data arms from {node_id} "
        f"(~{hours:.1f} h, ~${job['est_usd']})" + ("" if job["approved"] else " — needs approve"),
        job=job["job"], arms=[a["tag"] for a in arms])
    return job


def queue_eval(targets: list[str], m: dict | None = None) -> dict:
    """Queue an eval-only job: score checkpoints ("v5" or finished member / soup tags) with the
    member protocol (dense80 windows, gold-dev 29, holdout 29) so they compare 1:1 with members."""
    m = m if m is not None else load()
    base = G.clamp(dict(G.BASE, corpus="dense80"))
    arms, inits = [], []
    for t in targets:
        if t == "v5":
            fname, ds, path = "raptor_ft_coatnet_v5_full_swa.pt", "dreaddevelopment/raptor-knee-maxspan", \
                "raptor_ft_coatnet_v5_full_swa.pt"
        elif t in m["done"]:
            r = m["done"][t]
            fname, ds, path = r["weights"], r["dataset"], r.get("path") or f"{t}/{r['weights']}"
        else:
            raise SystemExit(f"unknown eval target {t}")
        inits.append((fname, ds, path))
        tag = "e" + hashlib.sha1(f"eval|{t}|{fname}".encode()).hexdigest()[:11]
        pending = {x["tag"] for j in m["queue"] + ([m["active"]] if m["active"] else []) for x in j["arms"]}
        if tag in m["done"] or tag in pending:
            continue
        arms.append({"tag": tag, "genome": base, "seed": 42, "parent": t, "eval_of": t,
                     "init_file": fname})
    if not arms:
        raise SystemExit("every requested eval already exists")
    hours = 1.0 + 0.2 * len(arms)
    job = {"job": "e-" + hashlib.sha1("".join(a["tag"] for a in arms).encode()).hexdigest()[:10],
           "arms": arms, "inits": inits, "est_hours": round(hours, 2),
           "est_usd": round(hours * launcher.PRICE_HR, 2), "corpus": "dense80", "approved": True,
           "queued": time.time()}
    m["queue"].insert(0, job)   # evals are cheap and unblock decisions: run first
    save(m)
    log("plan", f"eval job queued {job['job']}: {targets} (~${job['est_usd']})", job=job["job"])
    return job


def busy(m: dict | None = None) -> bool:
    """True while a member job is active or queued-and-approved (the loop must not plan)."""
    m = m if m is not None else load()
    return bool(m["active"]) or any(j.get("approved") for j in m["queue"])


def _launch(m: dict, job: dict) -> None:
    arms = []
    for a in job["arms"]:
        if a.get("eval_of"):
            lab, args = full_cli(a["genome"], a["seed"])
            args += f" --eval_only --init_model /workspace/input/init/{a['init_file']}"
            arms.append((a["tag"], lab, args))
        else:
            arms.append((a["tag"], *full_cli(a["genome"], a["seed"], init=a.get("init"),
                                             epochs=a["genome"]["epochs"])))
    out = launcher.launch(job["job"] + (f"-r{job['retries']}" if job.get("retries") else ""),
                          arms, job["est_hours"], corpus=job["corpus"],
                          inits=[tuple(x) for x in job.get("inits") or []])
    m["active"] = {**job, "pod": out["pod"], "result": out["result"], "launched": time.time(),
                   "price_hr": float(out.get("costPerHr") or launcher.PRICE_HR)}


def _collect(a: dict) -> dict[str, dict]:
    """Download per-arm metadata (not weights) and summarise gold/holdout."""
    from rsna_knee.agent import loop as L
    dest = RESULTS / a["job"]
    tags = [x["tag"] for x in a["arms"]]
    L.download_results(a["result"], dest, tags)
    out = {}
    for x in a["arms"]:
        meta = dest / x["tag"] / f"raptor_ft_{x['tag']}.json"
        if not meta.exists():
            continue
        j = json.loads(meta.read_text())
        out[x["tag"]] = {**x, "job": a["job"], "dataset": a["result"],
                         "weights": f"raptor_ft_{x['tag']}.pt",
                         "gold_dev": j.get("best_gold_auc"), "swa_gold_dev": j.get("swa_gold_auc"),
                         "holdout_auc": L.gold_macro(dest / x["tag"] / f"raptor_holdout_{x['tag']}.npz"),
                         "finished": time.time()}
    return out


def observe(m: dict) -> None:
    a = m["active"]
    if not a:
        return
    from rsna_knee.agent import loop as L
    try:
        pod = runpod.get_pod(a["pod"])
    except Exception:
        pod = None
    elapsed_h = (time.time() - a["launched"]) / 3600
    if pod is None and elapsed_h < 0.5 and not a.get("cost_logged"):
        log("error", f"member pod {a['pod']} vanished before billing; requeueing {a['job']}",
            component="members", pod=a["pod"])
        job = {k: a[k] for k in ("job", "arms", "est_hours", "est_usd", "corpus", "approved", "queued")
               if k in a} | ({"inits": a["inits"]} if a.get("inits") else {})
        job["retries"] = a.get("retries", 0) + 1
        m["active"] = None
        if job["retries"] <= L.INFRA_RETRIES:
            m["queue"].insert(0, job)
        else:
            log("gate", f"member job {job['job']} DROPPED after {job['retries']} phantom pods; "
                "requeue with `members queue` if still wanted", component="members", job=job["job"])
        return
    if pod is not None and pod.get("desiredStatus") == "RUNNING":
        if elapsed_h <= 2 * a["est_hours"] + 1 and L._running_pod_unhealthy(a) is None:
            return
        log("error", f"member pod {a['pod']} unhealthy or over time ({elapsed_h:.1f} h); terminating",
            component="members", pod=a["pod"])
        runpod.terminate_pod(a["pod"])
    elif pod is not None:
        runpod.terminate_pod(a["pod"])  # stopped after failed upload: stop volume billing
    if not a.get("cost_logged"):
        log("cost", f"member pod {a['pod']} ~{elapsed_h:.2f} h at ${a['price_hr']}/h",
            cost_usd=elapsed_h * a["price_hr"], pod=a["pod"])
        a["cost_logged"] = True
    got = _collect(a)
    st_file = RESULTS / a["job"] / "STATUS"
    final = st_file.exists() and "all arms finished" in st_file.read_text()  # snapshot predates "done"
    # Multi-arm pods push a partial version after each arm; Kaggle serves the previous version
    # for a few minutes after the final push. Do not finalize on a partial snapshot.
    if (not got or (not final and len(got) < len(a["arms"]))) and a.get("tries", 0) + 1 < DOWNLOAD_TRIES:
        a["tries"] = a.get("tries", 0) + 1
        log("note", f"member job {a['job']}: {len(got)}/{len(a['arms'])} arms visible, STATUS not final; "
            f"retry {a['tries']}/{DOWNLOAD_TRIES}", job=a["job"])
        return
    m["done"].update(got)
    missing = [x["tag"] for x in a["arms"] if x["tag"] not in got]
    for tag, r in got.items():
        log("gate", f"member {tag} ({r['genome']['labels']}, seed {r['seed']}): gold_dev "
            f"{r['gold_dev']} holdout {r['holdout_auc']}", member=tag, job=a["job"])
    if missing:
        log("error", f"member job {a['job']} missing arms {missing}", component="members")
    m["active"] = None


def tick(loop_active: bool) -> bool:
    """Observe/launch member jobs. Returns True when the loop must not plan this tick."""
    m = load()
    try:
        observe(m)
        if not m["active"] and not loop_active and m["queue"]:
            job = next((j for j in m["queue"] if j.get("approved")), None)
            if job is not None:
                try:
                    if not any(p.get("desiredStatus") == "RUNNING" for p in runpod.list_pods()):
                        m["queue"].remove(job)
                        _launch(m, job)
                except Exception as e:  # noqa: BLE001 — capacity/budget: keep it queued
                    if job not in m["queue"]:
                        m["queue"].insert(0, job)
                    log("error", f"member launch {job['job']} failed: {type(e).__name__}: {str(e)[:200]}",
                        component="members")
        return busy(m)
    finally:
        save(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["queue", "eval", "status", "approve", "drop"])
    ap.add_argument("--targets", default="", help="eval: comma list of 'v5' / member or soup tags")
    ap.add_argument("--node")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--labels", default=None, help="comma list; default = the node's labels")
    ap.add_argument("--job")
    ap.add_argument("--init", default=None, help="warm start: v5 (public full-SWA CoAtNet)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (warm starts: 4-8)")
    a = ap.parse_args()
    m = load()
    if a.cmd == "eval":
        print(json.dumps(queue_eval([t for t in a.targets.split(",") if t], m), indent=1, default=str))
        return
    if a.cmd == "queue":
        if not a.node:
            raise SystemExit("--node required")
        j = queue(a.node, [int(s) for s in a.seeds.split(",")],
                  a.labels.split(",") if a.labels else None, m, init=a.init, epochs=a.epochs)
        print(json.dumps(j, indent=1, default=str))
    elif a.cmd in ("approve", "drop"):
        for j in list(m["queue"]):
            if j["job"] == a.job:
                if a.cmd == "approve":
                    j["approved"] = True
                else:
                    m["queue"].remove(j)
                save(m)
                log("gate", f"user {a.cmd} member job {a.job}")
                return
        raise SystemExit(f"{a.job} not queued")
    else:
        print(json.dumps({"active": m["active"] and {k: m["active"][k] for k in ("job", "pod", "est_hours")},
                          "queue": [(j["job"], len(j["arms"]), j["est_usd"], j.get("approved")) for j in m["queue"]]},
                         indent=1, default=str))
        for tag, r in m["done"].items():
            print(f"{tag} {r['genome']['labels']:18s} seed={str(r['seed']):<5} init={r.get('init')} "
                  f"ep={r['genome']['epochs']} gold_dev={r['gold_dev']} "
                  f"holdout={r['holdout_auc']} {r['dataset']}/{r['weights']}")


if __name__ == "__main__":
    main()
