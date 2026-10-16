#!/usr/bin/env python3
"""Submit ledger + gate: the only sanctioned path from a built kernel to a competition submit.

    python scripts/submit_gate.py check  --kernel outputs/kernels/ens-d4-5bc0-w20
    python scripts/submit_gate.py submit --kernel outputs/kernels/ens-d4-5bc0-w20 --version 1 \
        --why "D4 + 5bc0 w=0.2: diversity leg" --i-approve
    python scripts/submit_gate.py record --ref 56643440 --score 0.920
    python scripts/submit_gate.py adopt  --kernel outputs/kernels/cpu-latoa-tf --version 1 --ref 57130001 --why "..."
        # a submission made on the website (API locked / user clicked): ledger row with its fingerprint, no Kaggle call
    python scripts/submit_gate.py ledger

Refuses when: no ``--i-approve`` (LB submits are user-approved), 5 submits already today (UTC,
counted from Kaggle once, not polled), or the kernel's fingerprint (code + members + weight)
was already submitted — two identical 0.920 submits burned quota on 2026-09-28. Every submit
is written to ``agent/runs/submits.json`` with its fingerprint and reason.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "agent" / "runs" / "submits.json"
COMP = "rsna-knee-abnormality-detection"
DAILY = 5
KAGGLE = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))


def load() -> list[dict]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def save(rows: list[dict]) -> None:
    LEDGER.write_text(json.dumps(rows, indent=1))


def fingerprint(kernel_dir: Path) -> str:
    """Hash of code + metadata datasets + ENSEMBLE.json (members, w); ignores titles/ids."""
    meta = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    h = hashlib.sha256()
    h.update((kernel_dir / meta["code_file"]).read_bytes())
    h.update(json.dumps(sorted(meta.get("dataset_sources", []))).encode())
    ens = kernel_dir / "ENSEMBLE.json"
    if ens.exists():
        h.update(ens.read_bytes())
    return h.hexdigest()[:16]


def today_count_kaggle() -> int | None:
    """Submits made today (UTC) per Kaggle; None if the CLI call fails."""
    try:
        r = subprocess.run([KAGGLE, "competitions", "submissions", COMP, "--csv"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    today = datetime.now(timezone.utc).date().isoformat()
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    return sum(1 for row in rows if str(row.get("date", "")).startswith(today))


def latest_ref(msg: str) -> int | None:
    """Ref of the submission just made (newest row whose description matches), so scores can be recorded."""
    try:
        r = subprocess.run([KAGGLE, "competitions", "submissions", COMP, "--csv"],
                           capture_output=True, text=True, timeout=120)
        for row in csv.DictReader(io.StringIO(r.stdout)):
            if row.get("description", "").strip() == msg.strip():
                return int(row["ref"])
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        pass
    return None


def kaggle_submit(kernel_id: str, version: int, msg: str) -> tuple[bool, str]:
    """ONE in-process submit call. On a refusal returns Kaggle's status, rate-limit headers and body: the CLI prints
    only the status line (13 Oct: a 400 whose reason was "non-versioned datasets" and a 429 could not be told apart)."""
    import socket
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    socket.setdefaulttimeout(120)      # 16 Oct: the Mac slept mid-request and the call hung for 4 h
    api = KaggleApi()
    api.authenticate()
    try:
        r = api.competition_submit_code("submission.csv", msg, COMP, kernel_id, version, quiet=True)
        return True, f"submitted: {r}"
    except requests.exceptions.HTTPError as e:
        resp = e.response
        hdr = {k: v for k, v in resp.headers.items() if k.lower() == "retry-after" or "ratelimit" in k.lower()}
        return False, f"{resp.status_code} {resp.reason} | {hdr} | {' '.join(resp.text.split())[:600]}"


def check(kernel_dir: Path, rows: list[dict], kaggle_today: int | None) -> list[str]:
    errs = []
    fp = fingerprint(kernel_dir)
    dup = [r for r in rows if r.get("fingerprint") == fp]
    if dup:
        errs.append(f"duplicate of submit ref {dup[0].get('ref')} ({dup[0].get('kernel')}) — same code/members/w")
    today = datetime.now(timezone.utc).date().isoformat()
    ledger_today = sum(1 for r in rows if r.get("date", "").startswith(today))
    n = max(ledger_today, kaggle_today or 0)
    if n >= DAILY:
        errs.append(f"{n} submits already today (UTC); daily cap {DAILY}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "submit", "record", "ledger", "adopt"])
    ap.add_argument("--kernel", type=Path)
    ap.add_argument("--version", type=int)
    ap.add_argument("--why", default="")
    ap.add_argument("--i-approve", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("--score", type=float)
    a = ap.parse_args()
    rows = load()
    if a.cmd == "ledger":
        for r in rows:
            print(f"{r.get('date', '')[:16]} ref={r.get('ref')} score={r.get('public_lb')} "
                  f"{r.get('kernel')} v{r.get('version')} fp={r.get('fingerprint')} | {r.get('why', '')[:60]}")
        return
    if a.cmd == "record":
        for r in rows:
            if str(r.get("ref")) == str(a.ref):
                r["public_lb"] = a.score
                save(rows)
                print(f"recorded {a.ref} = {a.score}")
                return
        raise SystemExit(f"ref {a.ref} not in ledger")
    if not a.kernel:
        raise SystemExit("--kernel required")
    if a.cmd == "adopt":
        if not (a.ref and a.why and a.version is not None):
            raise SystemExit("adopt needs --ref, --version and --why")
        if any(str(r.get("ref")) == str(a.ref) for r in rows):
            raise SystemExit(f"ref {a.ref} is already in the ledger")
        meta = json.loads((a.kernel / "kernel-metadata.json").read_text())
        rows.append({"date": datetime.now(timezone.utc).isoformat(), "kernel": meta["id"], "version": a.version,
                     "fingerprint": fingerprint(a.kernel), "why": a.why, "ref": int(a.ref), "public_lb": a.score,
                     "ensemble": None, "ts": time.time(), "via": "website"})
        save(rows)
        print(f"adopted {meta['id']} v{a.version} ref {a.ref}")
        return
    errs = check(a.kernel, rows, today_count_kaggle())
    if a.cmd == "check":
        print("OK" if not errs else "\n".join(f"REFUSE: {e}" for e in errs))
        raise SystemExit(1 if errs else 0)
    if not a.i_approve:
        errs.append("LB submits are user-approved: pass --i-approve after the user says yes")
    if not a.why:
        errs.append("--why required: the diagnosed gap-closer this submit tests")
    if a.version is None:
        errs.append("--version required (the COMPLETE kernel version)")
    if errs:
        raise SystemExit("\n".join(f"REFUSE: {e}" for e in errs))
    meta = json.loads((a.kernel / "kernel-metadata.json").read_text())
    msg = f"{meta['title']} | {a.why}"[:200]
    ok, said = kaggle_submit(meta["id"], a.version, msg)
    print(said)
    if not ok:
        raise SystemExit("submit failed")
    rows.append({"date": datetime.now(timezone.utc).isoformat(), "kernel": meta["id"], "version": a.version,
                 "fingerprint": fingerprint(a.kernel), "why": a.why, "ref": latest_ref(msg), "public_lb": None,
                 "ensemble": json.loads((a.kernel / "ENSEMBLE.json").read_text())
                 if (a.kernel / "ENSEMBLE.json").exists() else None, "ts": time.time()})
    save(rows)
    print("ledgered; fill ref/score later with: submit_gate.py record --ref <ref> --score <lb>")


if __name__ == "__main__":
    main()
