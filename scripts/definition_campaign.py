#!/usr/bin/env python3
"""Isolated $30 campaign. plan/dry-run are local; preflight only reads providers.

launch is an explicit paid action. Unknown launch outcomes retain their reservation and
block further launches; status reconciles by the unique pod name, never relaunches.
Historic campaign costs are not mixed into this authorized $30 ledger.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
import zlib

import requests
from rsna_knee.agent import runpod

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "agent/runs/definition_campaign.json"
LIMIT = 30.0
MAX_GPU_HR = .75
# Deliberately above published running volume/container rates; also covers a short
# stopped recovery interval. No network volume or separately retained storage is used.
STORAGE_GB_MONTH = .30
VOLUME_GB, DISK_GB = 140, 30
STORAGE_HR = (VOLUME_GB + DISK_GB) * STORAGE_GB_MONTH / (28 * 24)
RATE_CEILING = MAX_GPU_HR + STORAGE_HR
RESERVE_MARGIN = .20  # API termination delay / rounded billing; never trained against
USER = "girishbose"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
ACTIVE = {"reserved", "creating", "unknown", "running", "stopped", "terminate_pending"}
BUNDLE_SOURCES = (
    "runpod/definition_reader.py", "runpod/definition_geometry.py", "runpod/definition_job.py",
    "runpod/definition_job.sh", "runpod/fetch_dense80.py", "data/raw/train.csv", "data/raw/train_series.csv",
    "data/processed/definition_v1.npz", "data/processed/definition_v1.manifest.json",
    "work/definition_init/convnext_tiny.fb_in1k.pt", "work/definition_init/convnext_tiny.fb_in1k.provenance.json",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def ledger(path: Path | None = None):
    path = path or STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {"schema": 1, "limit_usd": LIMIT, "jobs": []}
        if state["limit_usd"] != LIMIT:
            raise RuntimeError("campaign limit changed; refusing")
        try:
            yield state
        finally:
            atomic_json(path, state)


def committed(state: dict) -> float:
    # Closed jobs retain full reservation until authoritative invoice reconciliation.
    # Estimated elapsed cost is informative only and never releases budget.
    return sum(float(j["reserved_usd"]) for j in state["jobs"])


def reserve(state: dict, cap: float, **details) -> dict:
    if state.get("launch_hold"):
        raise RuntimeError("campaign launch hold active; resolve and document the recorded startup failure before spending")
    if not math.isfinite(cap) or not .5 <= cap <= 3.0:
        raise ValueError("pilot reservation must be $0.50..$3.00")
    if any(j["status"] in ACTIVE for j in state["jobs"]):
        raise RuntimeError("unresolved campaign pod exists; status/collect before another launch")
    if committed(state) + cap > LIMIT + 1e-9:
        raise RuntimeError("$30 campaign reservation ceiling exceeded")
    job = {"job": "def-" + uuid.uuid4().hex[:12], "status": "reserved", "reserved_usd": cap,
           "created": time.time(), **details}
    quote_ceiling = float(details.get("quoted_gpu_hr", MAX_GPU_HR))
    if not math.isfinite(quote_ceiling) or not 0 < quote_ceiling <= MAX_GPU_HR:
        raise ValueError("invalid reservation GPU quote")
    job["accepted_gpu_hr_ceiling"] = quote_ceiling
    job["all_in_hr_ceiling"] = quote_ceiling + STORAGE_HR
    job["max_wall_seconds"] = min(4 * 3600, int((cap - RESERVE_MARGIN) / job["all_in_hr_ceiling"] * 3600))
    state["jobs"].append(job)
    return job


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def artifact_check() -> dict:
    processed = REPO / "data/processed"
    npz, manifest = processed / "definition_v1.npz", processed / "definition_v1.manifest.json"
    m = json.loads(manifest.read_text())
    if sha(npz) != m["artifact"]["sha256"]:
        raise RuntimeError("definition labels hash mismatch")
    if m.get("forbidden_image_teacher_inputs_used") is not False:
        raise RuntimeError("definition labels provenance not cleared")
    if m.get("excluded_expert_count") != 58:
        raise RuntimeError("expected all 58 expert studies excluded from train")
    for f in ["runpod/definition_reader.py", "runpod/definition_geometry.py", "runpod/definition_job.py",
              "runpod/definition_job.sh", "runpod/fetch_dense80.py", "data/raw/train.csv", "data/raw/train_series.csv"]:
        if not (REPO / f).is_file():
            raise FileNotFoundError(f)
    init = REPO / "work/definition_init/convnext_tiny.fb_in1k.pt"
    provenance = json.loads(init.with_suffix(".provenance.json").read_text())
    if sha(init) != provenance["sha256"]:
        raise RuntimeError("pretrained encoder source hash mismatch")
    return {"labels_sha256": sha(npz), "manifest_sha256": sha(manifest), "n_studies": m["n_studies"]}


def quote(gpu: str, cloud: str) -> dict:
    query = "query($id: String!){ gpuTypes(input:{id:$id}) { id lowestPrice(input:{gpuCount:1,secureCloud:" + ("true" if cloud == "SECURE" else "false") + "}) { uninterruptablePrice } } }"
    # Inventory responses can briefly alternate between a price and null. Retry
    # only this read, never a paid create; absence after three reads still blocks.
    for attempt in range(3):
        r = requests.post(runpod.GRAPHQL, headers=runpod._h(), json={"query": query, "variables": {"id": gpu}}, timeout=30)
        if not r.ok:
            raise RuntimeError(f"price quote HTTP {r.status_code}")
        payload = r.json()
        try:
            price = float(payload["data"]["gpuTypes"][0]["lowestPrice"]["uninterruptablePrice"])
            break
        except (KeyError, IndexError, TypeError, ValueError):
            if attempt == 2:
                raise RuntimeError("provider quote unavailable; no launch") from None
            time.sleep(1)
    if not math.isfinite(price) or not 0 < price <= MAX_GPU_HR:
        raise RuntimeError("quoted GPU rate exceeds $0.75/h ceiling or is invalid")
    return {"gpu": gpu, "cloud": cloud, "quoted_gpu_hr": price, "storage_hr_reserve": STORAGE_HR,
            "quoted_at": time.time(), "ceiling_all_in_hr": RATE_CEILING}


def preflight(gpu: str, cloud: str, cap: float, campaign_state: dict | None = None) -> dict:
    art = artifact_check()
    pods = runpod.list_pods()
    # The account is shared with unrelated projects. Only this ledger's pod IDs
    # and unique job names belong to this campaign; never manage foreign pods.
    if campaign_state is None:
        campaign_state = json.loads(STATE.read_text()) if STATE.exists() else {"jobs": []}
    own_ids = {j["pod"] for j in campaign_state["jobs"] if j.get("pod")}
    own_names = {name for j in campaign_state["jobs"] for name in (j.get("job"), j.get("pod_name")) if name}
    if any(p.get("id") in own_ids or p.get("name") in own_names for p in pods):
        raise RuntimeError("campaign has a pod; inspect/resolve before another paid pilot")
    bal = runpod.balance()
    balance = float(bal["clientBalance"])
    if not math.isfinite(balance) or balance < cap + .25:
        raise RuntimeError("unknown account balance or balance below reserved cap plus $0.25")
    return {**art, **quote(gpu, cloud), "balance_usd": balance,
            "balance_note": "Shared account liquidity only; balance changes and global spend are not campaign cost.",
            "unrelated_pods_ignored": len(pods)}


def bundle(job: dict, modes: list[str], epochs: int, resume_from: Path | None = None) -> Path:
    d = REPO / "work/definition_bundles" / job["job"]
    d.mkdir(parents=True, exist_ok=False)
    for f in BUNDLE_SOURCES:
        shutil.copy2(REPO / f, d / Path(f).name)
    if resume_from:
        for mode in modes:
            p = resume_from / mode / f"definition_{mode}_resume.pt"
            if not p.is_file():
                raise FileNotFoundError(p)
            shutil.copy2(p, d / f"resume_{mode}.pt")
    notice = ("Private competition training bundle. Competition data retains Kaggle/RSNA terms; report-derived labels retain "
              "source obligations. Original source paths and hashes are in definition_v1.manifest.json. Public pretrained "
              "ConvNeXt/ImageNet weights retain upstream timm and original model notices; official source https://huggingface.co/timm/convnext_tiny.fb_in1k . This dataset metadata's 'other' "
              "license is not a relicensing of any third-party source. No image-trained competition teacher inputs.\n")
    (d / "NOTICE.txt").write_text(notice)
    atomic_json(d / "job.json", {"job": job["job"], "modes": modes, "epochs": epochs,
                                "labels_sha256": job["labels_sha256"]})
    files = {p.name: sha(p) for p in d.iterdir() if p.is_file()}
    atomic_json(d / "bundle_manifest.json", {"sha256": files})
    atomic_json(d / "dataset-metadata.json", {"title": job["job"] + "-bundle", "id": f"{USER}/{job['job']}-bundle",
                                             "licenses": [{"name": "other"}]})
    return d


def kaggle(args: list[str], timeout=600):
    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError("Kaggle operation failed (provider output withheld; inspect locally)")
    return r


def verify_prepared_bundle(job: dict, modes: list[str], epochs: int, resume_from: Path | None) -> Path:
    """Only reuse a failed *pre-create* attempt; uncertain paid creates cannot retry."""
    if (job["status"] != "precreate_failed" or float(job["reserved_usd"]) != 0
            or any(job.get(k) is not None for k in ("pod", "launch_ts", "pod_name", "deadline"))):
        raise RuntimeError("prepared job is not an unlaunched precreate failure")
    d = REPO / "work/definition_bundles" / job["job"]
    receipt = json.loads((d / "bundle_manifest.json").read_text())["sha256"]
    for name, digest in receipt.items():
        p = d / name
        if not p.resolve().is_relative_to(d.resolve()) or sha(p) != digest:
            raise RuntimeError("prepared bundle receipt mismatch")
    for source in BUNDLE_SOURCES:
        name = Path(source).name
        if receipt.get(name) != sha(REPO / source):
            raise RuntimeError(f"prepared bundle is stale: {name}; rebuild/version it before launch")
    config = json.loads((d / "job.json").read_text())
    if config != {"job": job["job"], "modes": modes, "epochs": epochs,
                  "labels_sha256": sha(REPO / "data/processed/definition_v1.npz")}:
        raise RuntimeError("prepared bundle job configuration differs")
    for mode in modes:
        name = f"resume_{mode}.pt"
        expected = sha(resume_from / mode / f"definition_{mode}_resume.pt") if resume_from else None
        if receipt.get(name) != expected:
            raise RuntimeError("prepared bundle resume inputs differ")
    return d


def reserve_prepared(state: dict, job: dict, cap: float, details: dict) -> dict:
    # Reuse the identity only after reserve() has checked every campaign constraint.
    fresh = reserve(state, cap, **details)
    state["jobs"].pop()
    identity = {"job": job["job"], "created": job["created"]}
    retries = job.get("precreate_retries", 0) + 1
    job.clear(); job.update(fresh); job.update(identity)
    job["precreate_retries"] = retries
    return job


def wait_bundle_ready(job: dict, directory: Path, max_seconds: float = 180) -> str:
    """Kaggle can return errors while processing a newly created/versioned dataset."""
    slug = f"{USER}/{job['job']}-bundle"
    expected = sha(directory / "bundle_manifest.json")
    deadline = time.monotonic() + max_seconds
    with tempfile.TemporaryDirectory(prefix="definition-bundle-check-") as tmp:
        while time.monotonic() < deadline:
            try:
                r = kaggle(["datasets", "status", slug], timeout=min(30, max(1, deadline-time.monotonic())))
                if "ready" in r.stdout.lower():
                    kaggle(["datasets", "download", slug, "-f", "bundle_manifest.json", "-p", tmp, "--force", "-q"],
                           timeout=min(30, max(1, deadline-time.monotonic())))
                    remote = Path(tmp) / "bundle_manifest.json"
                    if remote.is_file():
                        if sha(remote) != expected:
                            raise ValueError("remote bundle manifest differs; GPU not created")
                        return expected
            except (RuntimeError, subprocess.TimeoutExpired):
                pass  # Read-only retry, bounded by the wall-clock deadline.
            time.sleep(min(5, max(0, deadline-time.monotonic())))
    raise RuntimeError("bundle readiness/manifest verification timed out; GPU not created")


def compact_startup(source: str) -> tuple[list[str], dict]:
    """Transport exact supervisor bytes with a stdlib-only integrity check."""
    raw = source.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    wrapper = ("import base64,zlib,hashlib;"
               f"_s=zlib.decompress(base64.b64decode({encoded!r}));"
               f"assert hashlib.sha256(_s).hexdigest()=={digest!r},'supervisor source checksum mismatch';"
               "exec(compile(_s,'<definition-supervisor>','exec'),{'__name__':'__main__'})")
    return ["python", "-u", "-c", wrapper], {
        "encoding": "zlib9-base64", "source_sha256": digest, "source_bytes": len(raw),
        "wrapper_sha256": hashlib.sha256(wrapper.encode()).hexdigest(), "wrapper_bytes": len(wrapper.encode())}


class PodCreationError(RuntimeError):
    """Only the allowlisted receipt is safe to surface; never stringify raw replies."""
    def __init__(self, receipt: dict):
        self.receipt = receipt
        super().__init__("pod creation " + receipt["outcome"] + "; " + json.dumps(receipt, sort_keys=True))


_PROVIDER_CODES = frozenset({
    "GRAPHQL_PARSE_FAILED", "GRAPHQL_VALIDATION_FAILED", "BAD_USER_INPUT", "UNAUTHENTICATED",
    "FORBIDDEN", "NOT_FOUND", "INTERNAL_SERVER_ERROR", "INSUFFICIENT_FUNDS", "GPU_UNAVAILABLE",
    "RESOURCE_EXHAUSTED", "TOO_MANY_REQUESTS"})
_PROVIDER_MESSAGES = (
    (r"cannot query field|unknown argument|unknown type|field .+ is not defined by type|variable .+ got invalid value|required type .+ was not provided|syntax error",
     "graphql_schema_or_input_validation"),
    (r"dockerargs.{0,100}(?:length|long|limit|max)|(?:length|long|limit|max).{0,100}dockerargs|payload too large|request entity too large",
     "request_or_startup_argument_size_limit"),
    (r"no (?:longer any )?(?:instances|gpus|resources).{0,100}available|insufficient capacity|out of stock|not enough capacity",
     "requested_capacity_unavailable"),
    (r"insufficient (?:funds|balance|credits)|not enough (?:funds|credits)|balance.{0,40}(?:low|insufficient)",
     "insufficient_account_balance"),
    (r"unauthorized|unauthenticated|invalid api key|invalid token|authentication", "authentication_rejected"),
    (r"forbidden|permission denied|access denied", "access_denied"),
    (r"rate limit|too many requests", "rate_limited"),
)


def provider_creation_receipt(response, payload, env_values=()) -> dict:
    """Canonical categories only: provider messages may echo credentials or inputs.

    Only GraphQL parse/schema validation with no resolver path and no data proves
    preexecution rejection. HTTP errors, capacity messages and missing IDs alone
    do not prove whether the mutation allocated a resource.
    """
    status = getattr(response, "status_code", None)
    receipt = {"outcome": "unknown", "http_status": status if isinstance(status, int) else None}
    errors = payload.get("errors", []) if isinstance(payload, dict) else []
    errors = errors if isinstance(errors, list) else []
    safe, definite = [], bool(errors) and not payload.get("data")
    for item in errors:
        item = item if isinstance(item, dict) else {}
        extensions = item.get("extensions")
        code = extensions.get("code") if isinstance(extensions, dict) else None
        code = code if isinstance(code, str) and code in _PROVIDER_CODES else "UNRECOGNIZED"
        # Redact known environment values before classification; never emit the
        # resulting message itself, including unknown tokens or signed URLs.
        message = item.get("message", "")
        message = message if isinstance(message, str) else ""
        for value in sorted((str(v) for v in env_values if v), key=len, reverse=True):
            message = message.replace(value, "[redacted]")
        categories = [label for pattern, label in _PROVIDER_MESSAGES if re.search(pattern, message, re.I)]
        safe.append({"code": code, "messages": categories or ["provider_message_withheld"]})
        definite = definite and code in {"GRAPHQL_PARSE_FAILED", "GRAPHQL_VALIDATION_FAILED"} and not item.get("path")
    if definite:
        receipt["outcome"] = "definite_preexecution_rejection"
    receipt["graphql_errors"] = safe
    if response is not None:
        # Preserve complete evidence locally, never in the ledger or user output.
        raw = getattr(response, "content", None)
        if not isinstance(raw, bytes):
            raw = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else b""
        receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
        private = REPO / "work/definition_private/provider_responses"
        try:
            private.mkdir(parents=True, exist_ok=True, mode=0o700); private.chmod(0o700)
            path = private / (uuid.uuid4().hex + ".response")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            receipt["private_response_file"] = str(path)
        except OSError:
            receipt["private_response_saved"] = False
    return receipt


def create_deadline_pod(spec: dict, deadline: float) -> dict:
    """Provider-side expiry survives host failure. Never retry this mutation automatically."""
    expiry = datetime.fromtimestamp(deadline, timezone.utc).isoformat().replace("+00:00", "Z")
    inp = {"name": spec["name"], "imageName": spec["imageName"], "cloudType": spec["cloudType"],
           "gpuTypeId": spec["gpuTypeIds"][0], "gpuCount": 1, "volumeInGb": VOLUME_GB,
           "containerDiskInGb": DISK_GB, "volumeMountPath": "/workspace", "ports": "8000/http",
           "minVcpuCount": 8, "minMemoryInGb": 32, "minDownload": 500, "minUpload": 100,
           "terminateAfter": expiry,
           "env": [{"key": k, "value": v} for k, v in spec["env"].items()]}
    # GraphQL dockerArgs is parsed as shell arguments; shlex.quote preserves embedded source.
    import shlex
    inp["dockerArgs"] = shlex.join(spec["dockerStartCmd"])
    # Live Pod output does not expose terminateAfter. Input was schema-validated;
    # do not request a nonexistent output field (that rejects creation entirely).
    q = "mutation($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id costPerHr desiredStatus } }"
    try:
        r = requests.post(runpod.GRAPHQL, headers=runpod._h(), json={"query": q, "variables": {"input": inp}}, timeout=60)
    except requests.RequestException:
        # Transport exception text can include request headers or URLs.
        raise PodCreationError({"outcome": "unknown", "http_status": None,
                                "transport_error": "provider_request_failed"}) from None
    try:
        payload = r.json()
    except ValueError:
        payload = None
    data = payload.get("data") if isinstance(payload, dict) else None
    p = data.get("podFindAndDeployOnDemand") if isinstance(data, dict) else None
    if not r.ok or not isinstance(p, dict) or not p.get("id"):
        raise PodCreationError(provider_creation_receipt(r, payload, spec["env"].values())) from None
    p["_creation_receipt"] = {"outcome": "acknowledged", "http_status": getattr(r, "status_code", None)}
    return p


def launch(args):
    # Keep lock across provider creation; persist reservation BEFORE any external mutation.
    with ledger() as st:
        prepared = getattr(args, "prepared_job", None)
        if prepared:
            matches = [j for j in st["jobs"] if j["job"] == prepared]
            if len(matches) != 1:
                raise RuntimeError("prepared job must match exactly one ledger entry")
            j = matches[0]
            d = verify_prepared_bundle(j, args.modes.split(","), args.epochs, args.resume_from)
        pf = preflight(args.gpu, args.cloud, args.cap, campaign_state=st)
        j = reserve_prepared(st, j, args.cap, pf) if prepared else reserve(st, args.cap, **pf)
        atomic_json(STATE, st)
        try:
            if not prepared:
                d = bundle(j, args.modes.split(","), args.epochs, args.resume_from)
                kaggle(["datasets", "create", "-p", str(d), "-q"])
            # Wait for bundle to be usable BEFORE allocating a GPU.
            j["bundle_manifest_sha256"] = wait_bundle_ready(j, d)
        except Exception:
            j["status"] = "precreate_failed"; j["reserved_usd"] = 0.0
            raise
        j["status"] = "creating"; j["launch_ts"] = time.time()
        j["deadline"] = j["launch_ts"] + j["max_wall_seconds"]
        j["pod_name"] = j["job"]; j["result"] = f"{USER}/{j['job']}-out"
        atomic_json(STATE, st)
        status_token = secrets.token_urlsafe(32)
        private = REPO / "work/definition_private"; private.mkdir(parents=True, exist_ok=True)
        token_file = private / f"{j['job']}.token"
        token_file.write_text(status_token); token_file.chmod(0o600)
        j["status_token_file"] = str(token_file)
        atomic_json(STATE, st)
        env = {"KAGGLE_API_TOKEN": Path("~/.kaggle/access_token").expanduser().read_text().strip(),
               "DEF_RUNPOD_API_KEY": runpod.secret("RUNPOD_API_KEY"), "DEF_BUNDLE": f"{USER}/{j['job']}-bundle",
               "DEF_RESULT": j["result"], "DEF_DEADLINE": str(j["deadline"]), "DEF_MAX_GPU_HR": str(j["accepted_gpu_hr_ceiling"]),
               "DEF_STATUS_TOKEN": status_token}
        startup, j["startup_transport"] = compact_startup((d / "definition_job.py").read_text())
        atomic_json(STATE, st)
        spec = {"name": j["pod_name"], "imageName": IMAGE, "computeType": "GPU", "gpuCount": 1,
                "cloudType": args.cloud, "gpuTypeIds": [args.gpu], "interruptible": False,
                "containerDiskInGb": DISK_GB, "volumeInGb": VOLUME_GB, "volumeMountPath": "/workspace",
                "ports": ["8000/http"], "minRAMPerGPU": 32, "minVCPUPerGPU": 8,
                "env": env, "dockerStartCmd": startup}
        try:
            p = create_deadline_pod(spec, j["deadline"])  # exactly once
            j["provider_creation_receipt"] = p.pop("_creation_receipt", {"outcome": "acknowledged"})
            j["pod"] = p["id"]; j["status"] = "running"
            atomic_json(STATE, st)  # persist id before inspecting anything else
            j["provider_deadline_requested"] = datetime.fromtimestamp(j["deadline"], timezone.utc).isoformat()
            j["provider_creation_acknowledged"] = True
            j["provider_deadline_readback"] = "unavailable: live Pod GraphQL output omits terminateAfter"
            # A future REST schema may expose it; lack of readback is recorded, not invented.
            try:
                live = runpod.get_pod(j["pod"])
                if live.get("terminateAfter"):
                    expires = live["terminateAfter"]
                    if abs(datetime.fromisoformat(expires.replace("Z", "+00:00")).timestamp() - j["deadline"]) > 2:
                        raise ValueError("provider expiry differs")
                    j["provider_deadline_readback"] = expires
            except requests.RequestException:
                pass
            price = float(p.get("costPerHr") or runpod.get_pod(p["id"]).get("costPerHr"))
            j["observed_gpu_hr"] = price
            if not math.isfinite(price) or not 0 < price <= j["accepted_gpu_hr_ceiling"]:
                raise RuntimeError("observed pod rate exceeds cap or is unknown")
        except Exception as exc:
            if isinstance(exc, PodCreationError):
                j["provider_creation_receipt"] = exc.receipt
            if j.get("pod"):
                j["status"] = "terminate_pending"
                try:
                    runpod.terminate_pod(j["pod"])
                except Exception:
                    pass
            else:
                j["status"] = "unknown"
            diagnostic = json.dumps(j["provider_creation_receipt"], sort_keys=True) if isinstance(exc, PodCreationError) else "provider diagnostic unavailable"
            raise RuntimeError("launch uncertain/rejected; reservation retained; run status, never blindly relaunch; " + diagnostic) from None
        print(json.dumps({k: j[k] for k in ("job", "pod", "deadline", "reserved_usd", "observed_gpu_hr", "result")}, indent=2))


def status():
    pods = runpod.list_pods()  # failure leaves ledger unchanged; never assume absence on exception
    now = time.time()
    with ledger() as st:
        for j in st["jobs"]:
            if j["status"] not in ACTIVE and j["status"] != "reconciled_absent":
                continue
            matches = [p for p in pods if p["id"] == j.get("pod") or p.get("name") == j.get("pod_name")]
            if len(matches) > 1:
                raise RuntimeError("multiple matching pods; inspect immediately")
            if matches:
                p = matches[0]; j["pod"] = p["id"]
                if j["status"] == "reconciled_absent" or j.get("reappeared_after_absence_review") is not None:
                    # A manually reviewed absence permits a new reservation, but
                    # never permits an old delayed create to keep billing.
                    j["status"] = "terminate_pending"
                    j.setdefault("reappeared_after_absence_review", now)
                    runpod.terminate_pod(j["pod"])
                    continue
                price = p.get("costPerHr")
                if price is not None:
                    j["observed_gpu_hr"] = float(price)
                j["status"] = "running" if p.get("desiredStatus") == "RUNNING" else "stopped"
                if j.get("status_token_file"):
                    try:
                        token = Path(j["status_token_file"]).read_text().strip()
                        r = requests.get(runpod.proxy_url(j["pod"]) + "/STATUS.json", headers={"Authorization": "Bearer " + token}, timeout=10)
                        if r.ok:
                            j["progress"] = r.json()
                    except (OSError, ValueError, requests.RequestException):
                        pass
                if now >= j.get("deadline", now + 1):
                    runpod.terminate_pod(j["pod"])
                    j["status"] = "terminate_pending"
            elif j["status"] == "reconciled_absent":
                continue  # Preserve the manual review and its retained reservation.
            elif j.get("pod"):
                j["status"] = "terminated"; j["last_seen_gone"] = now
            else:
                # Unknown creates without id require positive provider evidence/manual review.
                j["status"] = "unknown"
            elapsed = max(0, now - j.get("launch_ts", now))
            j["elapsed_cost_estimate_usd"] = round(elapsed / 3600 * (j.get("observed_gpu_hr", MAX_GPU_HR) + STORAGE_HR), 4)
        print(json.dumps({"ceiling_usd": LIMIT, "reserved_cumulative_usd": committed(st),
                          "available_to_reserve_usd": round(LIMIT - committed(st), 2),
                          "launch_hold": st.get("launch_hold"),
                          "cost_note": "Elapsed estimates are not invoices; full reservations remain charged conservatively.",
                          "jobs": st["jobs"]}, indent=2))


def collect(job_id: str):
    with ledger() as st:
        j = next(j for j in st["jobs"] if j["job"] == job_id)
        result = j["result"]
    dest = REPO / "outputs/definition_campaign" / job_id
    dest.mkdir(parents=True, exist_ok=True)
    kaggle(["datasets", "download", result, "-p", str(dest), "--unzip", "--force", "-q"], timeout=900)
    # Kaggle dir-mode zip stores each mode directory as a separate archive.
    for archive in dest.glob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            root = dest if all(i.filename.startswith(archive.stem + "/") for i in z.infolist()) else dest / archive.stem
            for info in z.infolist():
                if not (root / info.filename).resolve().is_relative_to(dest.resolve()):
                    raise RuntimeError("unsafe result archive path")
            z.extractall(root)
    manifest = json.loads((dest / "result_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        p = dest / name
        if not p.resolve().is_relative_to(dest.resolve()) or sha(p) != digest:
            raise RuntimeError("downloaded result hash/path mismatch")
    with ledger() as st:
        j = next(j for j in st["jobs"] if j["job"] == job_id)
        j["collected"] = time.time(); j["result_receipt"] = manifest
    print(f"verified results: {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan", "dry-run", "preflight", "launch", "status", "collect"])
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--gpu", default=runpod.GPU_4090)
    ap.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="COMMUNITY")
    ap.add_argument("--modes", choices=["control", "definition", "control,definition"], default="control,definition")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--job")
    ap.add_argument("--prepared-job", help="reuse an already uploaded bundle after a precreate failure; never re-creates its dataset")
    ap.add_argument("--resume-from", type=Path, help="verified collected result directory containing mode/resume.pt files")
    a = ap.parse_args()
    if a.command in ("plan", "dry-run"):
        s = {"jobs": []}; j = reserve(s, a.cap)
        print(json.dumps({"campaign_usd": LIMIT, "pilot_cap_usd": a.cap, "gpu_hr_ceiling": MAX_GPU_HR,
                          "storage_hr_reserve": STORAGE_HR, "max_wall_seconds": j["max_wall_seconds"],
                          "modes": a.modes, "epochs": a.epochs, "corpus": "dense512", "max_active_pods": 1,
                          "note": "No launch. Paid launch requires valid artifacts, current quote, no unresolved campaign pod and sufficient shared-account liquidity; unrelated pods are ignored."}, indent=2))
    elif a.command == "preflight": print(json.dumps(preflight(a.gpu, a.cloud, a.cap), indent=2))
    elif a.command == "launch": launch(a)
    elif a.command == "status": status()
    elif a.command == "collect": collect(a.job)


if __name__ == "__main__":
    main()
