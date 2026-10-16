#!/usr/bin/env python3
"""Log when each submission of a UTC day gets its score: run time on the hidden test = scored - submitted (+-10 min).

    python scripts/score_times.py --day 2026-10-13 [--until-hours 12]      # log: outputs/score_times.log

Why (12 Oct): the 0.956 public notebook gives its second model a budget of 8.3 h; whether a specialist can be added
after it inside the 9 h limit depends on how long the unchanged fork really takes.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "outputs" / "score_times.log"
COMP = "rsna-knee-abnormality-detection"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    help="UTC date of the submissions, e.g. 2026-10-13 (default: the current UTC day)")
    ap.add_argument("--until-hours", type=float, default=12.0)
    a = ap.parse_args()
    seen: set[str] = set()
    t_end = time.time() + a.until_hours * 3600
    while time.time() < t_end:
        try:
            out = subprocess.run(["kaggle", "competitions", "submissions", COMP, "-v"], capture_output=True, text=True, timeout=120).stdout
            rows = [r for r in csv.DictReader(io.StringIO(out)) if str(r.get("date", "")).startswith(a.day)]
        except Exception as e:  # noqa: BLE001
            rows = []
            print(f"read failed: {e}", flush=True)
        now = datetime.now(timezone.utc)
        for r in rows:
            key = r["date"]
            if key in seen or "complete" not in r.get("status", "").lower() and "error" not in r.get("status", "").lower():
                continue
            seen.add(key)
            sub = datetime.fromisoformat(r["date"].replace(" ", "T")[:19]).replace(tzinfo=timezone.utc)
            with LOG.open("a") as f:
                f.write(f"{now:%m-%d %H:%M}Z | {r.get('fileName', '')[:40]} | {r.get('status')} | score {r.get('publicScore')} | "
                        f"submitted {sub:%H:%M}Z | seen after {(now - sub).total_seconds() / 3600:.2f} h | {r.get('description', '')[:60]}\n")
        if rows and len(seen) == len(rows) and len(rows) >= 5:
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
