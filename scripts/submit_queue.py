#!/usr/bin/env python3
"""Submit the next kernels of a fixed queue, at most CAP per UTC day - ONE gate call per kernel, never a retry.

    python scripts/submit_queue.py [--dry]        # queue: agent/runs/submit_queue.json, log: outputs/submit_queue.log

Why (13 Oct): a restart loop made ~1,400 refused submit calls and Kaggle locked submissions for the whole account
(API and website) for 40 h. This runner is safe to start from a calendar job or by hand, any number of times:
- a kernel already in the ledger (same kernel + version) is skipped; a kernel that Kaggle refused once is never
  tried again by this script (state: agent/runs/submit_queue_state.json);
- a 429 with a SHORT retry-after (<= 15 min; 15 Oct: two submits 7 s apart got "retry-after 295") is waited out
  once and that kernel is tried once more; any other 429 stops the run and sets a HOLD until now + retry-after
  (+ 2 min); while the hold lasts the script makes no call at all. Submits are spaced by GAP seconds;
- any other refusal is logged with Kaggle's text and the run goes on to the next kernel;
- CAP (3, the user's rule of 13 Oct) counts this campaign's ledger rows of the current UTC day.
Queue rows: {"kernel": "outputs/kernels/<dir>", "version": 1, "why": "..."} in the order they should go in.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "agent" / "runs" / "submit_queue.json"
STATE = REPO / "agent" / "runs" / "submit_queue_state.json"
LEDGER = REPO / "agent" / "runs" / "submits.json"
LOG = REPO / "outputs" / "submit_queue.log"
CAP = 3
GAP = 45          # seconds between two submit calls
SHORT = 900       # a retry-after up to this is waited out once, in the run


def say(msg: str) -> None:
    line = f"{datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="say what would be submitted; no Kaggle submit call")
    a = ap.parse_args()
    queue = json.loads(QUEUE.read_text())
    state = json.loads(STATE.read_text()) if STATE.exists() else {"hold_until": 0, "hold_why": "", "refused": {}}
    now = time.time()
    if now < state.get("hold_until", 0):
        say(f"HOLD until {datetime.fromtimestamp(state['hold_until']):%m-%d %H:%M} ({state.get('hold_why', '')}): no call made")
        return 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sent, called = 0, False
    for row in queue:
        d = REPO / row["kernel"]
        kid = json.loads((d / "kernel-metadata.json").read_text())["id"]
        key = f"{kid}:v{row['version']}"
        ledger = json.loads(LEDGER.read_text())       # re-read: the gate appends to it
        n_today = sum(str(r.get("date", "")).startswith(today) for r in ledger)
        if any(r.get("kernel") == kid and r.get("version") == row["version"] for r in ledger):
            continue
        if key in state["refused"]:
            say(f"{key}: refused before ({state['refused'][key][:80]}) - skipped")
            continue
        if n_today >= CAP:
            say(f"cap: {n_today} of {CAP} submissions in the ledger for UTC day {today}; next in the queue: {key}")
            break
        try:
            st = subprocess.run(["kaggle", "kernels", "status", kid], capture_output=True, text=True, timeout=120).stdout
        except subprocess.TimeoutExpired:
            st = "status call timed out"
        if "COMPLETE" not in st:       # a kernel pushed by a calendar job may still be saving: a later run takes it
            say(f"{key}: not COMPLETE yet ({' '.join(st.split())[-40:]}) - skipped this run")
            continue
        if a.dry:
            say(f"DRY would submit {key} (ledger rows today {n_today + sent})")
            sent += 1
            if n_today + sent >= CAP:
                break
            continue
        for attempt in (0, 1):
            if called:
                time.sleep(GAP)
            called = True
            try:
                r = subprocess.run([sys.executable, str(REPO / "scripts" / "submit_gate.py"), "submit", "--kernel", str(d),
                                    "--version", str(row["version"]), "--i-approve", "--why", row["why"]],
                                   capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired:      # outcome unknown: stop; a later run sees it in the ledger or tries again
                say(f"{key}: the gate call timed out (600 s) - run stopped; check the submissions list before the next run")
                return 1
            out = " ".join((r.stdout + " " + r.stderr).split())
            m = re.search(r"retry-after'?:\s*'?(\d+)", out)
            wait = int(m.group(1)) if m else 6 * 3600
            if r.returncode != 0 and "429" in out and wait <= SHORT and attempt == 0:
                say(f"{key}: 429, short retry-after {wait} s - waiting it out, then one more call")
                time.sleep(wait + 20)
                continue
            break
        if r.returncode == 0:
            say(f"{key}: SUBMITTED | {out[:160]}")
            continue
        if "429" in out:
            state["hold_until"], state["hold_why"] = time.time() + wait + 120, f"429 on {key}, retry-after {wait} s"
            STATE.write_text(json.dumps(state, indent=1))
            say(f"{key}: 429, retry-after {wait} s -> HOLD, run stopped | {out[:200]}")
            return 1
        if "REFUSE:" in out and "daily cap" in out:
            say(f"{key}: gate says the daily cap is reached - run stopped | {out[:160]}")
            break
        state["refused"][key] = out[:400]
        STATE.write_text(json.dumps(state, indent=1))
        say(f"{key}: REFUSED, not retried | {out[:400]}")
    say("run done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
