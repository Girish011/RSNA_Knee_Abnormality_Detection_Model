"""Live monitoring page for the autonomous campaign.

    python scripts/agent_dashboard.py            # write agent/runs/dashboard.html once
    python scripts/agent_dashboard.py --serve    # http://localhost:8765, refreshes every 20 s

Shows: spend vs the $300 cap, current best / gate, every journal node (online vs dream),
the latest activity events, and the latest commits with changed files. Standard library
only, so it runs from any Python on the machine.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsna_knee.agent.activity import ACTIVITY, BUDGET_USD, JOURNAL, RUNS, read, spent

OUT = RUNS / "dashboard.html"
GATE = ("2026-10-06", 0.950)

CSS = """
body{font:14px system-ui,sans-serif;margin:24px;background:#111;color:#ddd}
h1{font-size:20px}h2{font-size:15px;margin-top:28px;color:#9cf}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #333;padding:4px 8px;
text-align:left;vertical-align:top}th{color:#999;font-weight:500}
.card{display:inline-block;background:#1c1c1c;border:1px solid #333;border-radius:6px;
padding:10px 16px;margin:0 12px 12px 0}.big{font-size:22px;color:#fff}
.error{color:#f77}.train{color:#fc6}.dream{color:#8d8}.submit{color:#c9f}.gate{color:#6cf}
code{color:#bbb;font-size:12px}.bar{background:#333;height:8px;border-radius:4px;width:240px}
.fill{background:#fc6;height:8px;border-radius:4px}
"""


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=20, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _commits(n: int = 15) -> list[tuple[str, str, str, str]]:
    raw = _git("log", f"-{n}", "--date=format:%m-%d %H:%M", "--pretty=format:@@%h|%ad|%s",
               "--name-only")
    out = []
    for block in raw.split("@@")[1:]:
        head, *files = block.strip().splitlines()
        sha, date, subject = head.split("|", 2)
        out.append((sha, date, subject, ", ".join(f for f in files if f)[:300]))
    return out


def _journal_rows() -> list[dict]:
    if not JOURNAL.exists():
        return []
    nodes = [n for n in json.loads(JOURNAL.read_text()) if n["id"] != "root"]
    return sorted(nodes, key=lambda n: n["created"], reverse=True)


def _event_row(e: dict) -> str:
    esc = html.escape
    cost = f"${e['cost_usd']:.2f}" if e.get("cost_usd") else ""
    extra = {k: v for k, v in e.items() if k not in {"ts", "kind", "msg", "cost_usd"}}
    return (
        f"<tr><td>{esc(e['ts'])}</td><td class={esc(e['kind'])}>{esc(e['kind'])}</td>"
        f"<td>{esc(e['msg'])}</td><td>{cost}</td>"
        f"<td><code>{esc(json.dumps(extra))[:240]}</code></td></tr>"
    )


def render() -> str:
    events = read(ACTIVITY)
    nodes = _journal_rows()
    usd = spent(events)
    pct = min(100.0, 100 * usd / BUDGET_USD)
    lbs = [n["public_lb"] for n in nodes if n.get("public_lb") is not None]
    lbs += [e["public_lb"] for e in events if e.get("public_lb") is not None]
    best_lb = max(lbs) if lbs else None
    valid = [n for n in nodes if n.get("proxy_auc") is not None]
    best = max(valid, key=lambda n: n["proxy_auc"]) if valid else None
    best_proxy = f"{best['proxy_auc']:.4f}" if best else "-"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    esc = html.escape

    n_online = sum(n["kind"] == "online" for n in nodes)
    n_dream = sum(n["kind"] == "dream" for n in nodes)
    bar = f"<div class=bar><div class=fill style='width:{pct:.0f}%'></div></div>"
    cards = [
        f"<div class=card>Spend<div class=big>${usd:.2f} / ${BUDGET_USD:.0f}</div>{bar}</div>",
        f"<div class=card>Best public LB<div class=big>{best_lb if best_lb else '-'}</div></div>",
        f"<div class=card>Best proxy AUC<div class=big>{best_proxy}</div></div>",
        f"<div class=card>Gate {GATE[0]}<div class=big>LB &ge; {GATE[1]}</div></div>",
        f"<div class=card>Nodes<div class=big>{len(nodes)}</div>{n_online} online / {n_dream} dream</div>",
        f"<div class=card>Branch<div class=big>{esc(branch or '-')}</div></div>",
    ]
    ev_rows = "".join(_event_row(e) for e in reversed(events[-60:]))
    node_rows = "".join(
        f"<tr><td><code>{esc(n['id'])}</code></td><td class={esc(n['kind'])}>{esc(n['kind'])}</td>"
        f"<td>{esc(n['operator'])}</td><td>{esc(n['status'])}</td>"
        f"<td>{n['proxy_auc'] if n['proxy_auc'] is not None else ''}</td>"
        f"<td>{n['gold_auc'] if n['gold_auc'] is not None else ''}</td>"
        f"<td>{n['public_lb'] if n['public_lb'] is not None else ''}</td>"
        f"<td>{n['gpu_hours']:.1f}</td><td>{esc(n['plan'])[:200]}</td></tr>"
        for n in nodes
    )
    commit_rows = "".join(
        f"<tr><td><code>{esc(s)}</code></td><td>{esc(d)}</td><td>{esc(m)}</td>"
        f"<td><code>{esc(f)}</code></td></tr>"
        for s, d, m, f in _commits()
    )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=20><title>RSNA agent</title><style>{CSS}</style></head><body>
<h1>RSNA Knee autonomous campaign</h1><div>updated {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
<div style='margin-top:12px'>{''.join(cards)}</div>
<h2>Experiment graph (newest first)</h2><table><tr><th>id</th><th>kind</th><th>operator</th>
<th>status</th><th>proxy</th><th>gold-58</th><th>public LB</th><th>GPU h</th><th>plan</th></tr>
{node_rows or '<tr><td colspan=9>no nodes yet</td></tr>'}</table>
<h2>Activity (latest 60)</h2><table><tr><th>time</th><th>kind</th><th>what</th><th>cost</th>
<th>details</th></tr>{ev_rows or '<tr><td colspan=5>no events yet</td></tr>'}</table>
<h2>Code changes (latest commits)</h2><table><tr><th>commit</th><th>when</th><th>message</th>
<th>files</th></tr>{commit_rows}</table></body></html>"""


def write() -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    return OUT


def serve(port: int, every: float) -> None:
    def loop():
        while True:
            write()
            time.sleep(every)

    write()
    threading.Thread(target=loop, daemon=True).start()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(RUNS), **kw)

        def log_message(self, *a):
            pass

    print(f"dashboard: http://localhost:{port}/dashboard.html")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--every", type=float, default=20.0)
    args = ap.parse_args()
    if args.serve:
        serve(args.port, args.every)
    else:
        print(write())


if __name__ == "__main__":
    main()
