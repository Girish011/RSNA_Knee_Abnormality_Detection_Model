"""Append-only activity log so every autonomous action is auditable.

One JSON object per line in ``agent/runs/activity.jsonl``. The file is committed with the
code, so the GitHub history of this file is the campaign's audit trail, and
``scripts/agent_dashboard.py`` renders it live. Never log secrets; ``log`` refuses values
that look like API keys.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "agent" / "runs"
ACTIVITY = RUNS / "activity.jsonl"
JOURNAL = RUNS / "journal.json"
BUDGET_USD = 300.0

KINDS = {"plan", "code", "train", "dream", "submit", "data", "cost", "gate", "error", "note"}
_SECRET = re.compile(r"(KGAT_|rpa_|sk-)[A-Za-z0-9]{12,}")


def log(kind: str, msg: str, *, cost_usd: float = 0.0, path: Path = ACTIVITY, **details) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}; use one of {sorted(KINDS)}")
    blob = json.dumps({"msg": msg, **details}, default=str)
    if _SECRET.search(blob):
        raise ValueError("refusing to log a value that looks like an API key")
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "msg": msg,
             "cost_usd": round(float(cost_usd), 4), **details}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event


def read(path: Path = ACTIVITY) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def spent(events: list[dict]) -> float:
    return round(sum(e.get("cost_usd", 0.0) for e in events), 2)


def assert_budget(extra_usd: float, path: Path = ACTIVITY, budget: float = BUDGET_USD) -> None:
    """Hard stop before any paid action that would cross the campaign cap."""
    total = spent(read(path)) + extra_usd
    if total > budget:
        raise RuntimeError(f"budget cap: ${total:.2f} would exceed ${budget:.0f}")
