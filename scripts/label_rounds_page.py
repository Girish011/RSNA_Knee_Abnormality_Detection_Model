#!/usr/bin/env python3
"""Monitoring page for the pseudo-label rounds (PLAN_TO_TOP "clean pseudo-label rounds").

    python scripts/label_rounds_page.py            # writes outputs/label_rounds.html

Reads the job registry agent/runs/label_rounds.json, polls each pod's monitor (STATUS + train.log),
checks which fold results are already under outputs/runpod_results/, and runs make_round_labels.py in
check mode for the label-quality numbers (holdout-29 vs experts). No secrets are read or written.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "agent" / "runs" / "label_rounds.json"
OUT = REPO / "outputs" / "label_rounds.html"
RES = REPO / "outputs" / "runpod_results"
SUBMITS = REPO / "agent" / "runs" / "submits.json"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEAK5 = {"Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis", "Contusion"}


def get(url: str, n: int = 200_000) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})   # the proxy 403s urllib's default UA
        with urllib.request.urlopen(req, timeout=12) as r:
            return r.read(n).decode("utf-8", "replace")
    except Exception:
        return ""


def poll(job: dict, now: dt.datetime) -> dict:
    j = dict(job)
    local = list((RES / j["job"]).glob(f"*/raptor_oof_*_fold{j['fold']}.npz")) if j.get("job") else []
    if j.get("done_local") or local:
        j.update(state="collected", ep=j["epochs"], status="results on disk")
        return j
    if j.get("terminated"):
        j.update(state="gone", ep=j.get("ep_at_stop", 0), status="terminated: " + j["terminated"])
        return j
    base = f"https://{j['pod']}-8000.proxy.runpod.net"
    status = get(f"{base}/STATUS", 300).strip()
    log = get(f"{base}/{j['tag']}/train.log") if status else ""
    eps = [int(m) for m in re.findall(r"^ep(\d+) ", log, flags=re.M)]
    j["ep"] = (max(eps) + 1) if eps else 0
    j["status"] = status or "pod not reachable"
    if not status:
        j["state"] = "gone"
    elif re.search(r"fail|FAIL|FAULT|UNAVAIL", status):
        j["state"] = "failed"
    elif status.startswith("done"):
        j["state"] = "uploaded"
    elif status.startswith("training") or status.startswith("uploading"):
        j["state"] = "training"
    else:
        j["state"] = "starting"
    hours = (now - dt.datetime.fromisoformat(j["launched"])).total_seconds() / 3600
    j["hours"] = hours
    j["cost"] = hours * j["price"]
    secs = [int(m) for m in re.findall(r'"sec": (\d+)', log) + re.findall(r"\| (\d+)s\s*$", log, flags=re.M)]
    if j["ep"] and secs:
        j["eta_h"] = max(secs) / j["ep"] * (j["epochs"] - j["ep"]) / 3600
    return j


def label_quality(reg: dict) -> dict:
    cmd = [sys.executable, str(REPO / "scripts" / "make_round_labels.py"), "--out", "_check.parquet", "--min-cover", "2"]
    src = reg.get("label_source")   # the label set the current round trains on: {"families": [[glob, ..], ..], "args": [..]}
    if src:
        cmd += src.get("args", [])
        for globs in src["families"]:
            cmd += ["--family", ",".join(str(RES / g) for g in globs)]
    else:
        for fam in reg["families"]:
            cmd += ["--family", ",".join(str(RES / g) for g in fam["globs"])]
    txt = subprocess.run(cmd, capture_output=True, text=True).stdout
    q = {"rows": {}, "per": {}, "cover": None}
    m = re.search(r"OOF coverage (\d+)/(\d+)", txt)
    if m:
        q["cover"] = (int(m[1]), int(m[2]))
    for name in ("report (pilk)", "clean OOF ensemble", "round target"):
        m = re.search(re.escape(name) + r"\s+([\d.]+)\s+(.*)", txt)
        if m:
            q["rows"][name] = float(m[1])
            q["per"][name] = [float(x) for x in m[2].split()]
    q["silent"] = {f: float(m[1]) for f in LAB
                   if (m := re.search(r"^" + re.escape(f) + r"\s+([\d.]+)", txt, flags=re.M))}
    m = re.search(r"holdout-(\d+)", txt)
    q["n"] = int(m[1]) if m else 0
    return q


def esc(x) -> str:
    return html.escape(str(x))


def render(reg: dict, jobs: list[dict], q: dict, now: dt.datetime) -> str:
    lo, hi = 0.80, 1.00
    pct = lambda v: max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))
    bars = "".join(
        f'<div class="bar"><span class="bl">{esc(lbl)}</span><span class="bt"><i class="{cls}" style="width:{pct(v):.1f}%"></i></span>'
        f'<b>{v:.3f}</b></div>'
        for lbl, key, cls in (("Report labels", "report (pilk)", "b0"), ("Clean model ensemble", "clean OOF ensemble", "b1"),
                              ("Round target (blend)", "round target", "b2"))
        if (v := q["rows"].get(key)) is not None)
    fams = [f for f in reg["families"]   # only the families this round trains
            if any(x["family"] == f["key"] and x["round"] == reg["round"] for x in jobs)] or reg["families"]
    grid = ""
    for fam in fams:
        cells = ""
        for k in range(5):
            j = next((x for x in jobs if x["family"] == fam["key"] and x["fold"] == k and x["round"] == reg["round"]), None)
            if j is None:
                cells += '<td><span class="chip none">not started</span></td>'
                continue
            p = 100 * j["ep"] / j["epochs"]
            eta = f' · {j["eta_h"]:.1f} h left' if j.get("eta_h") and j["state"] == "training" else ""
            cells += (f'<td><span class="chip {j["state"]}">{esc(j["state"])}</span>'
                      f'<span class="prog"><i style="width:{p:.0f}%"></i></span>'
                      f'<small>{j["ep"]}/{j["epochs"]} epochs{eta}</small></td>')
        grid += f'<tr><th>{esc(fam["name"])}<small>{esc(fam["note"])}</small></th>{cells}</tr>'
    rows = ""
    for j in jobs:
        if j.get("done_local") or (j["state"] == "collected" and j["round"] != reg["round"]):
            continue   # earlier rounds are on disk; the table is for what is running or belongs to this round
        rows += (f'<tr><td class="mono">{esc(j["job"])}</td><td><span class="chip {j["state"]}">{esc(j["state"])}</span></td>'
                 f'<td>{esc(j["status"][:60])}</td><td class="num">{j["ep"]}/{j["epochs"]}</td>'
                 f'<td class="num">{j.get("hours", 0):.1f}</td><td class="num">${j["price"]:.2f}</td>'
                 f'<td class="num">${j.get("cost", 0):.2f}</td><td class="mono">{esc(j["pod"])}</td></tr>')
    per = ""
    if q["per"]:
        for i, f in enumerate(LAB):
            r, e, t = (q["per"][k][i] for k in ("report (pilk)", "clean OOF ensemble", "round target"))
            d = t - r
            per += (f'<tr><td>{esc(f)}{" <span class=weak>weak five</span>" if f in WEAK5 else ""}</td>'
                    f'<td class="num">{100 * q["silent"].get(f, 0):.0f}%</td><td class="num">{r:.2f}</td><td class="num">{e:.2f}</td>'
                    f'<td class="num"><b>{t:.2f}</b></td><td class="num {"up" if d > 0.004 else "dn" if d < -0.004 else ""}">{d:+.2f}</td></tr>')
    live = sum(j.get("cost", 0) for j in jobs)
    n_train = sum(j["state"] == "training" for j in jobs)
    n_done = sum(j["state"] in ("collected", "uploaded") for j in jobs if j["round"] == reg["round"])
    n_all = sum(j["round"] == reg["round"] for j in jobs)   # launched folds; a fold-0 A/B counts one, not five
    cov = f'{q["cover"][0]:,} of {q["cover"][1]:,}' if q["cover"] else "–"
    lb = reg["lb"]
    subs = ""
    for r in reversed(json.loads(SUBMITS.read_text())[-8:] if SUBMITS.exists() else []):
        t = dt.datetime.fromisoformat(r["date"]).astimezone(IST)
        sc = r.get("public_lb")
        subs += (f'<tr><td class="mono">{t:%m-%d %H:%M}</td><td class="mono">{esc(r["kernel"].split("/")[-1])}</td>'
                 f'<td>{esc(r.get("why", ""))}</td><td class="num">'
                 + (f"<b>{sc:.3f}</b>" if sc is not None else '<span class="chip starting">scoring</span>') + "</td></tr>")
    src_name = (reg.get("label_source") or {}).get("name", "next round")
    return f"""<title>Label Rounds Monitor</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one column, summary strip -> label quality -> fold grid -> pods -> per-finding table */
:root {{ --bg:#f3f5f6; --panel:#ffffff; --fg:#17232b; --mute:#5c6b75; --line:#d5dce0; --accent:#0d6e7a;
  --good:#1f7a4d; --warn:#a86a00; --bad:#b3372f; --track:#e4e9ec;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#10171b; --panel:#182227; --fg:#e3eaed; --mute:#93a3ac;
  --line:#2b3940; --accent:#4fb8c4; --good:#5cc08d; --warn:#e0a94a; --bad:#ec7b72; --track:#25323a; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#10171b; --panel:#182227; --fg:#e3eaed; --mute:#93a3ac; --line:#2b3940; --accent:#4fb8c4;
  --good:#5cc08d; --warn:#e0a94a; --bad:#ec7b72; --track:#25323a; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.45; padding-inline:16px; padding-block:24px 48px }}
main {{ max-width:1040px; margin:0 auto; display:flex; flex-direction:column; gap:28px }}
h1 {{ font-size:24px; font-weight:600; margin:0; text-wrap:balance }}
h2 {{ font-size:13px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--mute); margin:0 0 10px }}
p {{ margin:0; max-width:68ch }} small {{ color:var(--mute); font-size:12px; display:block }}
.asof {{ font-family:var(--mono); font-size:12px; color:var(--mute) }}
.strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line) }}
.strip div {{ background:var(--panel); padding:12px 14px }} .strip b {{ display:block; font-family:var(--mono); font-size:22px; font-weight:500; font-variant-numeric:tabular-nums }}
.strip span {{ font-size:12px; color:var(--mute) }}
.panel {{ background:var(--panel); border:1px solid var(--line); padding:16px }}
.bar {{ display:grid; grid-template-columns:minmax(120px,190px) 1fr 56px; gap:12px; align-items:center; margin:8px 0 }}
.bl {{ font-size:14px }} .bt {{ background:var(--track); height:14px; display:block }} .bt i {{ display:block; height:100% }}
.b0 {{ background:var(--mute) }} .b1 {{ background:var(--accent); opacity:.6 }} .b2 {{ background:var(--accent) }}
.bar b {{ font-family:var(--mono); font-weight:500; text-align:right; font-variant-numeric:tabular-nums }}
.axis {{ display:grid; grid-template-columns:minmax(120px,190px) 1fr 56px; gap:12px; font-family:var(--mono); font-size:11px; color:var(--mute) }}
.axis span {{ display:flex; justify-content:space-between }}
.scroll {{ overflow-x:auto }} table {{ border-collapse:collapse; width:100%; font-size:14px }}
th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top }}
thead th {{ font-size:12px; font-weight:600; color:var(--mute); white-space:nowrap }}
tbody th {{ font-weight:600; min-width:150px }} .num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap }}
.mono {{ font-family:var(--mono); font-size:13px; white-space:nowrap }}
.chip {{ display:inline-block; font-size:12px; font-weight:500; padding:1px 8px; border:1px solid currentColor; border-radius:2px; white-space:nowrap }}
.chip.training {{ color:var(--accent) }} .chip.collected,.chip.uploaded {{ color:var(--good) }} .chip.starting {{ color:var(--warn) }}
.chip.failed,.chip.gone {{ color:var(--bad) }} .chip.none {{ color:var(--mute) }}
.prog {{ display:block; background:var(--track); height:4px; margin:7px 0 4px; min-width:90px }} .prog i {{ display:block; height:100%; background:var(--accent) }}
.weak {{ font-size:11px; color:var(--warn); border:1px solid currentColor; padding:0 5px; margin-left:6px; white-space:nowrap }}
.up {{ color:var(--good) }} .dn {{ color:var(--bad) }}
</style>
<main>
<header><h1>Label rounds — round {reg["round"]} models</h1>
<div class="asof">as of {now:%Y-%m-%d %H:%M} IST · regenerate with scripts/label_rounds_page.py</div></header>
<div class="strip">
<div><b>{lb["best"]:.3f}</b><span>best LB (stack)</span></div>
<div><b>{lb["own"]:.3f}</b><span>own model alone, LB</span></div>
<div><b>{lb["top"]:.3f}</b><span>LB #1</span></div>
<div><b>{n_done}/{n_all}</b><span>folds finished this round</span></div>
<div><b>{n_train}</b><span>pods training</span></div>
<div><b>${reg["spent_before"] + live:.0f}</b><span>RunPod spend of ${reg["budget"]}</span></div>
</div>
<section><h2>Label quality against the experts (holdout-{q["n"]})</h2><div class="panel">
<p>Macro AUC of each label source against expert labels on the held-out expert studies. The round target is what this round's models train on ({esc(src_name)}); it covers {cov} report-only studies.</p>
{bars or "<p>No clean teacher predictions on disk yet.</p>"}
<div class="axis"><i></i><span><i>{lo:.2f}</i><i>{(lo + hi) / 2:.2f}</i><i>{hi:.2f}</i></span><i></i></div></div></section>
<section><h2>Leaderboard submissions (latest 8)</h2><div class="panel scroll"><table><thead><tr><th>Submitted (IST)</th><th>Kernel</th><th>What it tests</th><th class="num">Public LB</th></tr></thead><tbody>{subs}</tbody></table></div></section>
<section><h2>Round {reg["round"]} folds</h2><div class="panel scroll"><table><thead><tr><th>Family</th>{"".join(f"<th>Fold {k}</th>" for k in range(5))}</tr></thead><tbody>{grid}</tbody></table></div></section>
<section><h2>Pods</h2><div class="panel scroll"><table><thead><tr><th>Job</th><th>State</th><th>Pod status</th><th class="num">Epochs</th><th class="num">Hours</th><th class="num">$/h</th><th class="num">Cost</th><th>Pod id</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>Per finding (holdout-{q["n"]} AUC)</h2><div class="panel scroll"><table><thead><tr><th>Finding</th><th class="num">Report silent</th><th class="num">Report</th><th class="num">Ensemble</th><th class="num">Target</th><th class="num">Target − report</th></tr></thead><tbody>{per}</tbody></table>
<small>Report silent = share of training reports that do not address the finding. {q["n"]} studies only, so single findings move in steps of several points.</small></div></section>
</main>"""


def main() -> None:
    reg = json.loads(REG.read_text())
    now = dt.datetime.now()
    jobs = [poll(j, now) for j in reg["jobs"]]
    OUT.write_text(render(reg, jobs, label_quality(reg), now))
    print(f"wrote {OUT}")
    for j in jobs:
        print(f"  {j['job']:14s} {j['state']:10s} ep {j['ep']}/{j['epochs']}  {j['status'][:50]}")


if __name__ == "__main__":
    main()
