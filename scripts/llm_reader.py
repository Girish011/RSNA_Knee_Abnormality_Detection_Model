#!/usr/bin/env python3
"""Viewer for an LLM that reads one knee-MRI study slice by slice (the labelling tool's slice folders; 9 Oct).

    python scripts/llm_reader.py <packet> info                       # the series: plane, sequence type, slice count
    python scripts/llm_reader.py <packet> sheet s0                   # numbered contact sheet of the whole series
    python scripts/llm_reader.py <packet> slices s0 14-15            # one or two slices at full resolution
    python scripts/llm_reader.py <packet> crop s0 14 0.3 0.3 0.7 0.7 [--span 1]   # zoom; --span 1 = also slices 13 and 15
    ... slices / crop accept --win LO HI (grey levels 0-255 stretched to black..white); slices carry .1-.9 edge marks

Coordinates of `crop` are fractions of the image (x0 y0 x1 y1, origin top-left). Each call writes one JPG under
<packet>/_views/ and prints its path (open it with the Read tool); every call is appended to _views/log.jsonl.
Images are kept near one megapixel so that nothing is lost when the model's vision input resizes them.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def series(packet: Path) -> dict[str, dict]:
    return {s["dir"]: s for s in json.loads((packet / "meta.json").read_text())["series"]}


WIN = None   # (lo, hi) grey levels 0-255 stretched to black..white (--win), like a window / level setting


def load(packet: Path, s: str, i: int) -> Image.Image:
    im = Image.open(packet / s / f"{i:03d}.jpg").convert("L")
    if WIN:
        lo, hi = WIN
        im = im.point(lambda v: 0 if v <= lo else 255 if v >= hi else int((v - lo) * 255 / (hi - lo)))
    return im


def ticks(im: Image.Image) -> None:
    """Fraction marks (.1 ... .9) along the top and left edges, so a crop box can be read off the image."""
    d = ImageDraw.Draw(im)
    f = font(13)
    w, h = im.size
    for k in range(1, 10):
        x, y = round(w * k / 10), round(h * k / 10)
        d.line([x, 0, x, 10], fill=255, width=2); d.line([0, y, 10, y], fill=255, width=2)
        d.text((x + 3, 12), f".{k}", fill=255, font=f); d.text((13, y - 7), f".{k}", fill=255, font=f)


def tag(im: Image.Image, text: str, size: int = 18) -> None:
    d = ImageDraw.Draw(im)
    f = font(size)
    w = int(d.textlength(text, font=f)) + 8
    d.rectangle([0, 0, w, size + 8], fill=0)
    d.text((4, 3), text, fill=255, font=f)


def save(packet: Path, im: Image.Image, name: str, call: dict) -> None:
    out = packet / "_views"
    out.mkdir(exist_ok=True)
    n = len(list(out.glob("*.jpg")))
    p = out / f"{n:03d}_{name}.jpg"
    im.save(p, quality=92)
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(dict(call, t=round(time.time()), out=p.name)) + "\n")
    print(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packet", type=Path)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    a1 = sub.add_parser("sheet"); a1.add_argument("series")
    a2 = sub.add_parser("slices"); a2.add_argument("series"); a2.add_argument("range")
    a3 = sub.add_parser("crop"); a3.add_argument("series"); a3.add_argument("i", type=int)
    a3.add_argument("box", type=float, nargs=4); a3.add_argument("--span", type=int, default=0)
    for sp in (a2, a3):
        sp.add_argument("--win", type=int, nargs=2, metavar=("LO", "HI"), help="contrast window in grey levels 0-255")
    a = ap.parse_args()
    global WIN
    WIN = tuple(a.win) if getattr(a, "win", None) else None
    S = series(a.packet)
    if a.cmd == "info":
        for k, s in S.items():
            kind = ("fluid-sensitive" if s["fluid"] else "not fluid-sensitive (T1 / PD type)") + (", fat-suppressed" if s["fatsat"] else "")
            print(f"{k}: {s['plane']:9s} | {kind} | slices 0..{s['n'] - 1} | {s['ps']:.2f} mm/px")
        return
    if a.series not in S:
        raise SystemExit(f"unknown series {a.series}; have {list(S)}")
    n = S[a.series]["n"]
    head = f"{a.series} {S[a.series]['plane']} {'fluid' if S[a.series]['fluid'] else 'non-fluid'}"
    if a.cmd == "sheet":
        cols = max(4, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        t = 1080 // cols
        im = Image.new("L", (cols * t, rows * t), 0)
        for i in range(n):
            tile = load(a.packet, a.series, i).resize((t, t), Image.LANCZOS)
            tag(tile, str(i), 14)
            im.paste(tile, ((i % cols) * t, (i // cols) * t))
        save(a.packet, im, f"sheet_{a.series}", {"cmd": "sheet", "series": a.series})
    elif a.cmd == "slices":
        lo, _, hi = a.range.partition("-")
        idx = list(range(int(lo), int(hi or lo) + 1))
        if len(idx) > 2 or not all(0 <= i < n for i in idx):
            raise SystemExit(f"give one or two slices inside 0..{n - 1} (e.g. 14 or 14-15)")
        t = 1024 if len(idx) == 1 else 768
        im = Image.new("L", (t * len(idx), t), 0)
        for j, i in enumerate(idx):
            tile = load(a.packet, a.series, i).resize((t, t), Image.LANCZOS)
            ticks(tile)
            tag(tile, f"{head} slice {i}/{n - 1}")
            im.paste(tile, (j * t, 0))
        save(a.packet, im, f"slices_{a.series}_{a.range}", {"cmd": "slices", "series": a.series, "range": a.range})
    else:
        x0, y0, x1, y1 = a.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise SystemExit("crop box = x0 y0 x1 y1 as fractions, 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        idx = [i for i in range(a.i - a.span, a.i + a.span + 1) if 0 <= i < n]
        t = 1024 if len(idx) == 1 else (768 if len(idx) == 2 else 512)
        tiles = []
        for i in idx:
            src = load(a.packet, a.series, i)
            w, h = src.size
            c = src.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            c = c.resize((t, max(1, round(t * c.size[1] / c.size[0]))), Image.LANCZOS)
            tag(c, f"{head} slice {i} zoom")
            tiles.append(c)
        im = Image.new("L", (sum(c.size[0] for c in tiles), max(c.size[1] for c in tiles)), 0)
        x = 0
        for c in tiles:
            im.paste(c, (x, 0)); x += c.size[0]
        save(a.packet, im, f"crop_{a.series}_{a.i}", {"cmd": "crop", "series": a.series, "i": a.i, "box": a.box, "span": a.span})


if __name__ == "__main__":
    main()
