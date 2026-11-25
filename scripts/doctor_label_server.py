#!/usr/bin/env python3
"""Local labelling server for this Mac: serves the labelling page and keeps the progress file on disk.

    python scripts/doctor_label_server.py [--port 8791]

Serves outputs/doctor_label/KneeLabelling on 127.0.0.1 only. The page posts its answers after every click:
POST /progress writes outputs/doctor_label/returns/knee_labels_<reader>.json (atomic replace);
GET /progress?reader=<name> returns that reader's own file, so the work survives cleared browser data and
two readers on this Mac stay separate. On other computers the page is opened as a plain file and downloads its files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "outputs" / "doctor_label" / "KneeLabelling"
RET = REPO / "outputs" / "doctor_label" / "returns"


def slug(reader: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(reader or "")).strip("_")


def reader_progress(reader: str) -> dict:
    """This reader's own file only: two people labelling on the same Mac must never see each other's answers."""
    f = RET / f"knee_labels_{slug(reader)}.json"
    if not slug(reader) or not f.exists():
        return {"version": 1, "reader": reader, "cases": {}}
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {"version": 1, "reader": reader, "cases": {}}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path.rstrip("/").endswith("progress"):
            self._json(200, reader_progress((parse_qs(url.query).get("reader") or [""])[0]))
            return
        super().do_GET()

    def do_POST(self) -> None:
        if not self.path.split("?")[0].rstrip("/").endswith("progress"):
            self._json(404, {"error": "unknown path"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n < 20_000_000:
            self._json(413, {"error": "bad size"})
            return
        try:
            d = json.loads(self.rfile.read(n))
            assert isinstance(d.get("cases"), dict)
        except (ValueError, AssertionError):
            self._json(400, {"error": "not a progress file"})
            return
        reader = slug(d.get("reader")) or "reader"
        RET.mkdir(parents=True, exist_ok=True)
        out = RET / f"knee_labels_{reader}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, out)
        self._json(200, {"saved": out.name, "cases": len(d["cases"])})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), partial(Handler, directory=str(PKG)))
    print(f"labelling page: http://localhost:{a.port}/  (progress file folder: {RET})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
