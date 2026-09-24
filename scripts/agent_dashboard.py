"""Live monitoring page for the autonomous campaign.

    python scripts/agent_dashboard.py            # write agent/runs/dashboard.html once
    python scripts/agent_dashboard.py --serve    # http://localhost:8765, refreshes every 20 s

Merges: spend, loop state, journal, activity.jsonl, RunPod pod API, and the active pod's
HTTP progress page (STATUS + train.log epochs). No DeepSeek / Kaggle calls — viewing is free.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsna_knee.agent.activity import ACTIVITY, BUDGET_USD, JOURNAL, RUNS, read, spent

OUT = RUNS / "dashboard.html"
STATE = RUNS / "loop_state.json"
HOLDOUT = RUNS / "holdout.json"
LOOP_LOG = RUNS / "loop.log"
RESULTS = REPO / "outputs" / "runpod_results"
GATE = ("2026-10-06", 0.950)

CSS = """
body{font:14px system-ui,sans-serif;margin:24px;background:#111;color:#ddd}
h1{font-size:20px}h2{font-size:15px;margin-top:28px;color:#9cf}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #333;padding:4px 8px;
text-align:left;vertical-align:top}th{color:#999;font-weight:500}
.card{display:inline-block;background:#1c1c1c;border:1px solid #333;border-radius:6px;
padding:10px 16px;margin:0 12px 12px 0;min-width:140px;vertical-align:top}
.big{font-size:22px;color:#fff}.muted{color:#888;font-size:12px}
.error{color:#f77}.train{color:#fc6}.dream{color:#8d8}.submit{color:#c9f}.gate{color:#6cf}
.note{color:#aaa}.cost{color:#fc6}.data{color:#8cf}.plan{color:#c9f}.code{color:#bbb}
pre{background:#0a0a0a;border:1px solid #333;padding:10px;overflow:auto;max-height:320px;
font:12px ui-monospace,Menlo,monospace;color:#cfc;white-space:pre-wrap}
a{color:#9cf}code{color:#bbb;font-size:12px}
.bar{background:#333;height:8px;border-radius:4px;width:240px}
.fill{background:#fc6;height:8px;border-radius:4px}
.live{border-color:#fc6}.warn{color:#fc6}
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


def _http_get(url: str, timeout: float = 8.0) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rsna-dashboard/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def _proxy_base(pod: str) -> str:
    return f"https://{pod}-8000.proxy.runpod.net"


def _epoch_lines(text: str, n: int = 16) -> list[str]:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^ep\d+\s", s) or s.startswith("[ckpt]") or s.startswith("DONE ") \
                or s.startswith("SWA ") or s.startswith("device ") or s.startswith("train "):
            lines.append(s)
    return lines[-n:]


def _live_runpod(act: dict) -> dict:
    """Pull STATUS + train epochs from the pod's free HTTP progress server (no API key)."""
    pod = act.get("pod") or ""
    nodes = act.get("nodes") or []
    if not pod:
        return {"ok": False, "msg": "no active pod"}
    base = _proxy_base(pod)
    code, status = _http_get(f"{base}/STATUS")
    if code != 200:
        return {"ok": False, "msg": f"STATUS HTTP {code}: {status[:120]}", "base": base, "pod": pod}
    tag = nodes[0] if nodes else ""
    epochs: list[str] = []
    job_tail = ""
    if tag:
        c2, train = _http_get(f"{base}/{tag}/train.log")
        if c2 == 200:
            epochs = _epoch_lines(train)
        else:
            c3, job = _http_get(f"{base}/job.log")
            if c3 == 200:
                epochs = _epoch_lines(job)
                job_tail = "\n".join(job.strip().splitlines()[-12:])
    elapsed_h = (time.time() - float(act.get("launched") or time.time())) / 3600
    price = float(act.get("price_hr") or 0)
    return {
        "ok": True, "pod": pod, "base": base, "status": status.strip(), "epochs": epochs,
        "job_tail": job_tail, "tag": tag, "elapsed_h": elapsed_h,
        "est_hours": act.get("est_hours"), "price_hr": price,
        "spend_so_far": round(elapsed_h * price, 2), "job": act.get("job"),
        "result": act.get("result"),
    }


def _runpod_api_pods() -> list[dict]:
    try:
        from rsna_knee.agent import runpod
        out = []
        for p in runpod.list_pods():
            out.append({
                "id": p.get("id"), "name": p.get("name"), "status": p.get("desiredStatus"),
                "costPerHr": p.get("costPerHr"), "gpu": (p.get("machine") or {}).get("gpuDisplayName")
                or p.get("gpu"), "uptime": p.get("uptimeSeconds"),
            })
        bal = runpod.balance()
        return [{"_balance": bal}, *out]
    except Exception as e:
        return [{"_error": f"{type(e).__name__}: {str(e)[:120]}"}]


def _result_summaries() -> list[str]:
    if not RESULTS.exists():
        return []
    rows = []
    for d in sorted(RESULTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
        if not d.is_dir():
            continue
        status = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else ""
        arms = [p.name for p in d.iterdir() if p.is_dir()]
        rows.append(f"{d.name}: {status or 'no STATUS'} | arms={arms}")
    return rows


def _event_row(e: dict) -> str:
    esc = html.escape
    cost = f"${e['cost_usd']:.2f}" if e.get("cost_usd") else ""
    extra = {k: v for k, v in e.items() if k not in {"ts", "kind", "msg", "cost_usd"}}
    return (
        f"<tr><td>{esc(e['ts'])}</td><td class={esc(e['kind'])}>{esc(e['kind'])}</td>"
        f"<td>{esc(e['msg'])}</td><td>{cost}</td>"
        f"<td><code>{esc(json.dumps(extra, default=str))[:320]}</code></td></tr>"
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

    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    act = st.get("active") or {}
    pend = st.get("pending")
    live = _live_runpod(act) if act else {"ok": False, "msg": "no active job"}
    api_pods = _runpod_api_pods()
    bal = next((p["_balance"] for p in api_pods if "_balance" in p), None)
    api_rows = [p for p in api_pods if "id" in p]

    n_online = sum(n["kind"] == "online" for n in nodes)
    n_dream = sum(n["kind"] == "dream" for n in nodes)
    bar = f"<div class=bar><div class=fill style='width:{pct:.0f}%'></div></div>"
    age = (time.time() - st.get("heartbeat", 0)) / 60 if st.get("heartbeat") else None
    loop_txt = ("PAUSED" if st.get("paused") else
                f"alive, {age:.0f} min ago" if age is not None and age < 30 else
                f"STALE {age:.0f} min" if age is not None else "no heartbeat")

    bal_txt = "-"
    spend_hr = "-"
    if isinstance(bal, dict):
        bal_txt = f"${float(bal.get('clientBalance', 0)):.2f}"
        spend_hr = str(bal.get("currentSpendPerHr", "-"))
    cards = [
        f"<div class=card>Spend<div class=big>${usd:.2f} / ${BUDGET_USD:.0f}</div>{bar}"
        f"<div class=muted>logged GPU+LLM</div></div>",
        f"<div class=card>RunPod balance<div class=big>{bal_txt}</div>"
        f"<div class=muted>spend/hr {spend_hr}</div></div>",
        f"<div class=card>Best public LB<div class=big>{best_lb if best_lb else '-'}</div></div>",
        f"<div class=card>Best proxy AUC<div class=big>{best_proxy}</div></div>",
        f"<div class=card>Gate {GATE[0]}<div class=big>LB &ge; {GATE[1]}</div></div>",
        f"<div class=card>Nodes<div class=big>{len(nodes)}</div>{n_online} online / {n_dream} dream</div>",
        f"<div class=card>Loop<div class=big>{esc(loop_txt)}</div>"
        f"{'pod ' + esc(act.get('pod', '')) if act else 'idle'}</div>",
        f"<div class=card>Branch<div class=big>{esc(branch or '-')}</div></div>",
    ]
    if pend:
        cards.append(
            f"<div class='card live'><b class=warn>APPROVAL NEEDED</b>"
            f"<div><code>{esc(json.dumps(pend, default=str)[:280])}</code></div>"
            f"<div class=muted>python -m rsna_knee.agent.loop approve|reject</div></div>")

    # Live RunPod panel
    if live.get("ok"):
        ep_pre = esc("\n".join(live["epochs"]) or "(no epoch lines yet — still installing/downloading)")
        live_html = f"""
<h2>Live RunPod job</h2>
<div class='card live' style='display:block;width:calc(100% - 34px)'>
  <div><b>STATUS:</b> <code>{esc(live['status'])}</code>
    · pod <code>{esc(live['pod'])}</code>
    · ${live['price_hr']:.2f}/hr · ~{live['elapsed_h']:.1f} h elapsed
    · ~${live['spend_so_far']:.2f} so far
    · est {live.get('est_hours')} h</div>
  <div class=muted>job {esc(str(live.get('job')))} → result {esc(str(live.get('result')))}
    · <a href="{esc(live['base'])}/STATUS" target=_blank>STATUS</a>
    · <a href="{esc(live['base'])}/{esc(live.get('tag') or '')}/train.log" target=_blank>train.log</a>
    · <a href="{esc(live['base'])}/job.log" target=_blank>job.log</a></div>
  <pre>{ep_pre}</pre>
</div>"""
    else:
        msg = esc(live.get("msg", "idle"))
        link = ""
        if live.get("base"):
            link = f' · <a href="{esc(live["base"])}/STATUS" target=_blank>open STATUS</a>'
        live_html = f"""
<h2>Live RunPod job</h2>
<div class=card style='display:block;width:calc(100% - 34px)'>
  <div class=muted>{msg}{link}</div>
</div>"""

    pod_rows = "".join(
        f"<tr><td><code>{esc(p.get('id') or '')}</code></td><td>{esc(str(p.get('name') or ''))}</td>"
        f"<td>{esc(str(p.get('status') or ''))}</td><td>{p.get('costPerHr')}</td>"
        f"<td>{esc(str(p.get('gpu') or ''))}</td>"
        f"<td><a href='{_proxy_base(p['id'])}/STATUS' target=_blank>progress</a></td></tr>"
        for p in api_rows
    ) or "<tr><td colspan=6>no running pods</td></tr>"
    if api_pods and "_error" in api_pods[0]:
        pod_rows = f"<tr><td colspan=6 class=error>{esc(api_pods[0]['_error'])}</td></tr>"

    holdout_html = ""
    if HOLDOUT.exists():
        try:
            h = json.loads(HOLDOUT.read_text())
            holdout_html = f"<h2>Holdout scores (human keep/kill; planner never sees)</h2><pre>{esc(json.dumps(h, indent=1)[:2000])}</pre>"
        except json.JSONDecodeError:
            holdout_html = ""

    results = _result_summaries()
    results_html = ("<h2>Local result pulls</h2><pre>" + esc("\n".join(results)) + "</pre>") if results else ""

    loop_tail = ""
    if LOOP_LOG.exists() and LOOP_LOG.stat().st_size:
        loop_tail = "<h2>Loop process log (tail)</h2><pre>" + esc(
            "\n".join(LOOP_LOG.read_text(errors="replace").splitlines()[-40:])) + "</pre>"

    ev_rows = "".join(_event_row(e) for e in reversed(events[-120:]))
    node_rows = "".join(
        f"<tr><td><code>{esc(n['id'])}</code></td><td class={esc(n['kind'])}>{esc(n['kind'])}</td>"
        f"<td>{esc(n.get('operator') or '')}</td><td>{esc(n['status'])}</td>"
        f"<td>{n['proxy_auc'] if n['proxy_auc'] is not None else ''}</td>"
        f"<td>{n['gold_auc'] if n['gold_auc'] is not None else ''}</td>"
        f"<td>{n['public_lb'] if n['public_lb'] is not None else ''}</td>"
        f"<td>{n['gpu_hours']:.1f}</td><td>{esc(n.get('plan') or '')[:200]}</td></tr>"
        for n in nodes
    )
    commit_rows = "".join(
        f"<tr><td><code>{esc(s)}</code></td><td>{esc(d)}</td><td>{esc(m)}</td>"
        f"<td><code>{esc(f)}</code></td></tr>"
        for s, d, m, f in _commits()
    )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=20><title>RSNA agent</title><style>{CSS}</style></head><body>
<h1>RSNA Knee autonomous campaign</h1>
<div class=muted>updated {time.strftime('%Y-%m-%d %H:%M:%S')} · refreshes every 20s ·
localhost only (no DeepSeek / Kaggle / billing APIs for this page; RunPod progress HTTP is free)</div>
<div style='margin-top:12px'>{''.join(cards)}</div>
{live_html}
<h2>RunPod pods (API)</h2>
<table><tr><th>id</th><th>name</th><th>status</th><th>$/hr</th><th>gpu</th><th>progress</th></tr>
{pod_rows}</table>
<h2>Experiment graph (newest first)</h2><table><tr><th>id</th><th>kind</th><th>operator</th>
<th>status</th><th>proxy</th><th>gold-dev</th><th>public LB</th><th>GPU h</th><th>plan</th></tr>
{node_rows or '<tr><td colspan=9>no nodes yet</td></tr>'}</table>
{holdout_html}
{results_html}
<h2>Activity (latest 120)</h2><table><tr><th>time</th><th>kind</th><th>what</th><th>cost</th>
<th>details</th></tr>{ev_rows or '<tr><td colspan=5>no events yet</td></tr>'}</table>
{loop_tail}
<h2>Code changes (latest commits)</h2><table><tr><th>commit</th><th>when</th><th>message</th>
<th>files</th></tr>{commit_rows}</table></body></html>"""


def write() -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    return OUT


def serve(port: int, every: float) -> None:
    def loop():
        while True:
            try:
                write()
            except Exception as e:
                # Keep serving the last good page; surface the error next refresh.
                (RUNS / "dashboard_error.txt").write_text(f"{time.strftime('%H:%M:%S')} {e}\n")
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
