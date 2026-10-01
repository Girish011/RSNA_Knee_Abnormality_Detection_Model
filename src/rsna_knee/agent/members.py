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
import subprocess
import time
from pathlib import Path

from rsna_knee.agent import genome as G
from rsna_knee.agent import launcher, runpod
from rsna_knee.agent.activity import RUNS, log

MEMBERS = RUNS / "members.json"
POLICY = RUNS / "policy.json"   # user switches the agent cannot override (e.g. fold_jobs_closed)
RESULTS = G.REPO / "outputs" / "runpod_results"
JOURNAL_FILE = RUNS / "journal.json"
FULL_DATA_FACTOR = 1.25   # 4349 vs 3479 training studies
APPROVAL_USD = 10.0
# Public v5 full-SWA CoAtNet (CC0, same arch/head; 486/486 tensors load). It was trained on all
# studies, so fold-0 proxy would be leaky — warm starts live on the member track only and are
# judged on gold_dev / holdout / LB.
INITS = {"v5": "/workspace/input/init/raptor_ft_coatnet_v5_full_swa.pt"}
DOWNLOAD_TRIES = 8
# User 2026-09-30: up to 2 pods at once (counts every RUNNING pod on the account).
MAX_PODS = 2
# Phantom pods (created, then gone before billing) cost $0 but 10 min each; community capacity
# produced 3 in a row several times on 2026-09-30 and dropped the key fused-label job.
MEMBER_INFRA_RETRIES = 6
SEEDS_PER_RECIPE = 2   # 2- and 4-seed soups scored the same (0.9364 / 0.9365)


def _running(m: dict) -> list[dict]:
    """Running member jobs. ``m["active"]`` mirrors the first one for older readers."""
    if "running" not in m:   # one-time migration from the single-slot format
        m["running"] = [m["active"]] if m.get("active") else []
    return m["running"]


def load() -> dict:
    m = json.loads(MEMBERS.read_text()) if MEMBERS.exists() else {"queue": [], "active": None, "done": {}}
    _running(m)
    return m


def save(m: dict) -> None:
    run = _running(m)
    m["active"] = run[0] if run else None
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
    if genome.get("k", 8) > 8:
        # Keep images per forward = bs*k = 64 (as at k=8, bs=8). bs=8 x k=16 = 128 images of 384^2
        # overflowed PyTorch 32-bit indexing ("canUse32BitIndexMath") on 2026-09-30.
        args = re.sub(r"--bs \d+", f"--bs {max(1, 64 // int(genome['k']))}", args)
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
          m: dict | None = None, init: str | None = None, epochs: int | None = None,
          k: int | None = None) -> dict:
    """Queue one pod job whose arms are (seed × label set) full-data retrains of ``node_id``."""
    m = m if m is not None else load()
    if init and init not in INITS:
        raise SystemExit(f"init {init!r} not in {sorted(INITS)}")
    base = G.clamp(_node_genome(node_id))
    if epochs:
        base["epochs"] = int(epochs)
    if k:
        if int(k) not in (8, 12, 16):
            raise SystemExit("k must be 8, 12 or 16")
        base["k"] = int(k)
    arms, seen = [], {t for j in m["queue"] for t in (a["tag"] for a in j["arms"])} | set(m["done"])
    for lab in labels or [base["labels"]]:
        g = dict(base, labels=lab)
        if errs := [e for e in G.validate(g) if not (epochs and e.startswith("epochs="))
                    and not (k and e.startswith("k="))]:
            raise SystemExit(f"illegal genome for labels={lab}: {errs}")
        same = sum(1 for r in m["done"].values()
                   if r.get("fold") is None and not r.get("eval_of") and not r.get("soup_of")
                   and (r.get("genome") or {}).get("labels") == lab
                   and (r.get("genome") or {}).get("k", 8) == g["k"] and r.get("init") == init)
        if same >= SEEDS_PER_RECIPE:
            # 4-seed and 2-seed soups scored the same (0.9365 / 0.9364): more seeds of one recipe
            # do not move the ensemble. DeepSeek re-queued seed jobs repeatedly on 2026-09-30.
            raise SystemExit(f"{lab} (k={g['k']}, init={init}) already has {same} members; "
                             f"cap {SEEDS_PER_RECIPE} — change the recipe instead of adding seeds")
        for s in seeds:
            tag = member_tag(g, s, init)
            if tag in seen:
                continue
            seen.add(tag)
            arms.append({"tag": tag, "genome": g, "seed": int(s), "parent": node_id, "init": init})
    if not arms:
        raise SystemExit("every requested member already exists or is queued")
    hours = est_hours(base, len(arms)) * (base["k"] / 8 if base["k"] > 8 else 1)
    job = {"job": "m-" + hashlib.sha1("".join(a["tag"] for a in arms).encode()).hexdigest()[:10],
           "arms": arms, "est_hours": round(hours, 2), "est_usd": round(hours * launcher.PRICE_HR, 2),
           "corpus": base["corpus"], "approved": hours * launcher.PRICE_HR <= APPROVAL_USD,
           "queued": time.time(), **({"gpu_class": "48"} if base["k"] > 16 else {})}
    m["queue"].append(job)
    save(m)
    log("plan", f"members queued {job['job']}: {len(arms)} full-data arms from {node_id} "
        f"(~{hours:.1f} h, ~${job['est_usd']})" + ("" if job["approved"] else " — needs approve"),
        job=job["job"], arms=[a["tag"] for a in arms])
    return job


def fold_cli(genome: dict, fold: int, seed: int, init: str | None, epochs: int | None,
             bundle_dir: str = "/workspace/bundle") -> tuple[str, str]:
    """Protocol 5-fold arm (train folds != k, predict fold k OOF) with optional warm start."""
    labels_file, args = G.to_cli(genome, bundle_dir)
    args = re.sub(r"--fold \d+", f"--fold {int(fold)}", args)
    args = re.sub(r"--seed \d+", f"--seed {int(seed)}", args)
    if epochs:
        args = re.sub(r"--epochs \d+", f"--epochs {int(epochs)}", args)
    if init:
        args += f" --init_model {INITS[init]}"
    return labels_file, args


def queue_folds(node_id: str, labels: str, init: str | None = "v5", epochs: int | None = 6,
                seed: int = 42, m: dict | None = None) -> dict:
    """5-fold OOF job: every non-expert train study gets an out-of-fold image prediction.
    Feeds expert-anchored label fusion (scripts/fuse_labels.py), i.e. self-training."""
    m = m if m is not None else load()
    pol = POLICY
    if pol.exists() and json.loads(pol.read_text()).get("fold_jobs_closed"):
        raise SystemExit("5-fold / self-training rounds are CLOSED by user policy (agent/runs/policy.json): "
                         + json.loads(pol.read_text()).get("why", ""))
    info = G.label_registry().get(labels, {})
    if "n_fused" in info and info["n_fused"] < 0.95 * 4349:
        raise SystemExit(f"{labels} fused only {info['n_fused']} of 4349 studies (partial fold set); "
                         "wait for the *_full fusion before another self-training round")
    g = dict(G.clamp(_node_genome(node_id)), labels=labels)
    if epochs:
        g["epochs"] = int(epochs)
    if errs := [e for e in G.validate(g) if not (epochs and e.startswith("epochs="))]:
        raise SystemExit(f"illegal genome: {errs}")
    arms = []
    for k in range(G.PROTOCOL["folds"]):
        key = json.dumps({**g, "fold": k, "seed": seed, "init": init, "fit": "oof"}, sort_keys=True)
        arms.append({"tag": "f" + hashlib.sha1(key.encode()).hexdigest()[:11], "genome": g, "seed": seed,
                     "parent": node_id, "init": init, "fold": k})
    pending = {x["tag"] for j in m["queue"] + _running(m) for x in j["arms"]}
    if all(a["tag"] in m["done"] or a["tag"] in pending for a in arms):
        raise SystemExit("this 5-fold OOF set already exists or is queued")
    per_arm = (G.est_hours(g) - 0.5)
    hours = per_arm * len(arms) + 1.0
    job = {"job": "f-" + hashlib.sha1("".join(a["tag"] for a in arms).encode()).hexdigest()[:10],
           "arms": arms, "est_hours": round(hours, 2), "est_usd": round(hours * launcher.PRICE_HR, 2),
           "corpus": g["corpus"], "approved": hours * launcher.PRICE_HR <= APPROVAL_USD,
           "queued": time.time(), "kind": "oof"}
    m["queue"].append(job)
    save(m)
    log("plan", f"5-fold OOF job queued {job['job']} ({labels}, init={init}, {epochs} ep; ~${job['est_usd']})",
        job=job["job"])
    return job


def refold(job_id: str, folds: list[int], m: dict | None = None) -> dict:
    """Re-run failed folds of a finished OOF job. The new job is named ``<job>-fix`` so the fusion
    glob ``<job>*`` sees all five folds; its completion re-runs self-training on the full set."""
    m = m if m is not None else load()
    arms = [dict(r, tag=t) for t, r in m["done"].items() if r.get("job") == job_id and r.get("fold") in folds]
    if len(arms) != len(folds):
        raise SystemExit(f"{job_id}: found folds {[a['fold'] for a in arms]} in done, wanted {folds}")
    for a in arms:
        m["done"].pop(a["tag"])
    keep = ("tag", "genome", "seed", "parent", "init", "fold")
    arms = [{k: a.get(k) for k in keep} for a in arms]
    hours = (G.est_hours(arms[0]["genome"]) - 0.5) * len(arms) + 1.0
    job = {"job": f"{job_id}-fix", "arms": arms, "est_hours": round(hours, 2),
           "est_usd": round(hours * launcher.PRICE_HR, 2), "corpus": arms[0]["genome"]["corpus"],
           "approved": True, "queued": time.time(), "kind": "oof", "oof_base": job_id}
    m["queue"].insert(0, job)
    save(m)
    log("plan", f"refold {job['job']}: re-running folds {folds} of {job_id} (~${job['est_usd']})", job=job["job"])
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
        pending = {x["tag"] for j in m["queue"] + _running(m) for x in j["arms"]}
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
    return bool(_running(m)) or any(j.get("approved") for j in m["queue"])


def _launch(m: dict, job: dict) -> None:
    arms = []
    for a in job["arms"]:
        if a.get("fold") is not None:
            arms.append((a["tag"], *fold_cli(a["genome"], a["fold"], a["seed"], a.get("init"),
                                             a["genome"]["epochs"])))
            continue
        if a.get("eval_of"):
            lab, args = full_cli(a["genome"], a["seed"])
            args += f" --eval_only --init_model /workspace/input/init/{a['init_file']}"
            arms.append((a["tag"], lab, args))
        else:
            arms.append((a["tag"], *full_cli(a["genome"], a["seed"], init=a.get("init"),
                                             epochs=a["genome"]["epochs"])))
    out = launcher.launch(job["job"] + (f"-r{job['retries']}" if job.get("retries") else ""),
                          arms, job["est_hours"], corpus=job["corpus"],
                          inits=[tuple(x) for x in job.get("inits") or []],
                          gpu_class=job.get("gpu_class"))
    _running(m).append({**job, "pod": out["pod"], "result": out["result"], "launched": time.time(),
                        "price_hr": float(out.get("costPerHr") or launcher.PRICE_HR)})
    m["active"] = _running(m)[0]


def _collect(a: dict) -> dict[str, dict]:
    """Download per-arm metadata (not weights) and summarise gold/holdout."""
    from rsna_knee.agent import loop as L
    dest = RESULTS / a["job"]
    tags = [x["tag"] for x in a["arms"]]
    L.download_results(a["result"], dest, tags)
    for x in a["arms"]:
        if x.get("fold") is not None:
            t, k = x["tag"], x["fold"]
            for name in (f"{t}/raptor_oof_{t}_fold{k}.npz", f"{t}/raptor_oof_{t}_swa_fold{k}.npz",
                         f"{t}/raptor_gold_{t}.npz"):
                (dest / name).parent.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(["kaggle", "datasets", "download",
                                    a["result"], "-f", name, "-p", str((dest / name).parent)],
                                   capture_output=True, text=True, timeout=L.KAGGLE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    log("note", f"kaggle download timed out: {a['result']} {name}")
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
    for a in list(_running(m)):
        _observe_one(m, a)
    m["active"] = _running(m)[0] if _running(m) else None


def _finish(m: dict, a: dict) -> None:
    if a in _running(m):
        _running(m).remove(a)
    m["active"] = m["running"][0] if m["running"] else None   # never re-adopt a finished job


def _observe_one(m: dict, a: dict) -> None:
    from rsna_knee.agent import loop as L
    try:
        pod = runpod.get_pod(a["pod"])
    except Exception:
        pod = None
    elapsed_h = (time.time() - a["launched"]) / 3600
    if pod is not None and pod.get("desiredStatus") == "RUNNING" and not a.get("seen_alive"):
        # A pod whose own HTTP proxy answered once was real: if it later 404s it ended or was
        # terminated by hand — never a phantom to relaunch (2026-09-30: user-terminated pod was
        # relaunched as "-r2").
        if L._probe_pod_progress(a["pod"]).get("reachable"):
            a["seen_alive"] = True
    if pod is None and elapsed_h < 0.5 and not a.get("cost_logged") and not a.get("seen_alive"):
        log("error", f"member pod {a['pod']} vanished before billing; requeueing {a['job']}",
            component="members", pod=a["pod"])
        job = {k: a[k] for k in ("job", "arms", "est_hours", "est_usd", "corpus", "approved", "queued")
               if k in a} | ({"inits": a["inits"]} if a.get("inits") else {})
        job["retries"] = a.get("retries", 0) + 1
        _finish(m, a)
        closed = POLICY.exists() and json.loads(POLICY.read_text()).get("fold_jobs_closed")
        if a.get("kind") == "oof" and closed:
            log("gate", f"not relaunching {job['job']}: 5-fold rounds closed by policy", job=job["job"])
        elif job["retries"] <= MEMBER_INFRA_RETRIES:
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
    _finish(m, a)
    if a.get("kind") == "oof" and not missing:
        base = a.get("oof_base") or a["job"]
        have = {r.get("fold") for r in m["done"].values()
                if (r.get("job") or "").startswith(base) and r.get("fold") is not None}
        oof_ok = all(list((RESULTS).glob(f"{base}*/{t}/raptor_oof_{t}_fold{r['fold']}.npz"))
                     for t, r in m["done"].items()
                     if (r.get("job") or "").startswith(base) and r.get("fold") is not None)
        if have == set(range(G.PROTOCOL["folds"])) and oof_ok:
            auto_self_train(m, a)
        else:
            log("error", f"self-training skipped for {base}: folds {sorted(have)} / OOF files complete={oof_ok}; "
                f"use `members refold --job {base} --folds ...`", component="members")


FUSED_SEEDS = [42, 1337]


def _load_fusion():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fuse_labels", G.REPO / "scripts" / "fuse_labels.py")
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    return fl.run_fusion


def auto_self_train(m: dict, a: dict, run_fusion=None) -> dict | None:
    """Self-training step (host, deterministic): 5-fold OOF landed → fuse report + image labels
    on expert-dev, gate on expert-holdout, and on PASS queue members on the fused labels.
    No LLM judgement: the gate is a measured number. Failures are logged, never raised."""
    try:
        base = a["arms"][0]["genome"]["labels"]
        oof_job = a.get("oof_base") or a["job"]
        name = "fused_" + oof_job.split("-", 1)[1][:8] + ("_full" if a.get("oof_base") else "")
        r = (run_fusion or _load_fusion())(oof_job=oof_job, write=name, base=base)
    except Exception as e:  # noqa: BLE001
        log("error", f"self-training fusion for {a['job']} failed: {type(e).__name__}: {str(e)[:200]}",
            component="members")
        return None
    mac = r["macro"]
    log("gate", f"self-training {a['job']}: holdout macro report {mac['report']:.3f} image {mac['image']:.3f} "
        f"fused {mac['fused']:.3f} → {'PASS' if r['passed'] else 'FAIL'}"
        + (f"; registered {r['name']} ({r['n_fused']} studies)" if r["name"] else ""), job=a["job"])
    if not r["name"]:
        return r
    arm = a["arms"][0]
    try:
        queue(arm["parent"], FUSED_SEEDS, [r["name"]], m, init=arm.get("init"),
              epochs=arm["genome"]["epochs"])
        m["queue"].insert(0, m["queue"].pop())   # fused members jump the queue
    except SystemExit as e:
        log("error", f"self-training queue for {r['name']} refused: {e}", component="members")
    return r


OPS = RUNS / "members_ops.jsonl"


def request_op(op: dict) -> None:
    """Queue a state edit for the loop to apply at its next tick (never races a running tick)."""
    with OPS.open("a") as f:
        f.write(json.dumps(op) + "\n")


def apply_ops(m: dict) -> None:
    if not OPS.exists():
        return
    lines = OPS.read_text().splitlines()
    OPS.unlink()
    for line in lines:
        op = json.loads(line)
        if op.get("op") == "drop_running":
            before = len(_running(m))
            m["running"] = [r for r in _running(m)
                            if not (r["job"] == op["job"] and (not op.get("pod") or r["pod"] == op["pod"]))]
            log("gate", f"op drop_running {op['job']} {op.get('pod', '')}: removed {before - len(m['running'])}")
        elif op.get("op") == "drop_queued":
            m["queue"] = [j for j in m["queue"] if j["job"] != op["job"]]
            log("gate", f"op drop_queued {op['job']}")
        elif op.get("op") == "requeue":
            job = dict(op["job_spec"])
            m["queue"].insert(0, job)
            log("gate", f"op requeue {job['job']} (retries={job.get('retries')})")
    m["active"] = _running(m)[0] if _running(m) else None


def tick(loop_active: bool) -> bool:
    """Observe/launch member jobs. Returns True when the loop must not plan this tick."""
    m = load()
    try:
        apply_ops(m)
        observe(m)
        if len(_running(m)) < MAX_PODS and m["queue"]:
            waiting = {j["job"] for j in m["queue"]} | {r["job"] for r in _running(m)}
            job = next((j for j in m["queue"] if j.get("approved") and j.get("after") not in waiting), None)
            if job is not None:
                try:
                    live = sum(p.get("desiredStatus") == "RUNNING" for p in runpod.list_pods())
                    live = max(live, len(_running(m)) + int(loop_active))
                    if live < MAX_PODS:
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
    ap.add_argument("cmd", choices=["queue", "eval", "folds", "refold", "status", "approve", "drop"])
    ap.add_argument("--folds", default="", help="refold: comma list of fold indices")
    ap.add_argument("--targets", default="", help="eval: comma list of 'v5' / member or soup tags")
    ap.add_argument("--node")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--labels", default=None, help="comma list; default = the node's labels")
    ap.add_argument("--job")
    ap.add_argument("--init", default=None, help="warm start: v5 (public full-SWA CoAtNet)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (warm starts: 4-8)")
    ap.add_argument("--k", type=int, default=None, help="train windows per study; >8 pins to 48 GB GPUs")
    a = ap.parse_args()
    m = load()
    if a.cmd == "refold":
        print(json.dumps(refold(a.job, [int(x) for x in a.folds.split(",")], m), indent=1, default=str)[:800])
        return
    if a.cmd == "folds":
        print(json.dumps(queue_folds(a.node, a.labels, init=a.init or None, epochs=a.epochs, m=m),
                         indent=1, default=str)[:1500])
        return
    if a.cmd == "eval":
        print(json.dumps(queue_eval([t for t in a.targets.split(",") if t], m), indent=1, default=str))
        return
    if a.cmd == "queue":
        if not a.node:
            raise SystemExit("--node required")
        j = queue(a.node, [int(s) for s in a.seeds.split(",")],
                  a.labels.split(",") if a.labels else None, m, init=a.init, epochs=a.epochs, k=a.k)
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
        print(json.dumps({"running": [{k: r[k] for k in ("job", "pod", "est_hours")} for r in _running(m)],
                          "queue": [(j["job"], len(j["arms"]), j["est_usd"], j.get("approved")) for j in m["queue"]]},
                         indent=1, default=str))
        for tag, r in m["done"].items():
            print(f"{tag} {r['genome']['labels']:18s} seed={str(r['seed']):<5} init={r.get('init')} "
                  f"ep={r['genome']['epochs']} gold_dev={r['gold_dev']} "
                  f"holdout={r['holdout_auc']} {r['dataset']}/{r['weights']}")


if __name__ == "__main__":
    main()
