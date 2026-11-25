#!/usr/bin/env python3
"""VRAM guard for on-pod training args.

Measured fact (2026-09-27): on a 24 GB card, CoAtNet-384 with k>8 windows CUDA-OOMs
during the *first* training batch, i.e. before the trainer has written any checkpoint.
The host then sees an empty results dir ("no results in .../n-<id>") and the whole pod
is wasted. k_eval>24 has the same effect at evaluation time.

Genomes carried k=12 for a while (inherited from a parent node), so passes like

    --k 12 --k_eval 24 --epochs 20 ...

kept dying instantly. This module rewrites such flag pairs down to the measured-safe
menu values BEFORE the trainer starts, so a bad inherited genome degrades into a real
train instead of a vanished pod. It never raises; unknown args pass through untouched.
"""
from __future__ import annotations

import shlex
import sys

# Measured-safe ceiling on 24 GB: k=8 / k_eval=24. Do not raise without a new measurement.
LIMITS = {"--k": 8, "--k_eval": 24}


def clamp_args(args, limits=None):
    """Return (new_args, notes) with ``--k`` / ``--k_eval`` capped at the VRAM-safe menu.

    Only the long, two-token form (``--k 12``) is understood; ``--k=12`` is also handled.
    Values are compared as ints when possible and left alone when unparseable.
    """
    limits = dict(LIMITS if limits is None else limits)
    out, notes, i = [], [], 0
    while i < len(args):
        tok = args[i]
        name, sep, inline = tok.partition("=")
        cap = limits.get(name)
        if cap is None:
            out.append(tok)
            i += 1
            continue
        if sep:  # --k=12
            raw = inline
            nxt = None
            i += 1
        else:  # --k 12
            raw = args[i + 1] if i + 1 < len(args) else None
            nxt = raw
            i += 2 if raw is not None else 1
        try:
            val = int(raw)
        except (TypeError, ValueError):
            out.append(tok)
            if nxt is not None:
                out.append(nxt)
            continue
        if val > cap:
            notes.append(f"{name} {val} -> {cap}")
            val = cap
        out.append(f"{name} {val}" if nxt is None else name)
        if nxt is not None:
            out.append(str(val))
    return out, notes


def clamp_line(line: str):
    """Clamp a whitespace-separated extra-args string; returns (string, notes)."""
    toks = shlex.split(line or "")
    out, notes = clamp_args(toks)
    return " ".join(shlex.quote(t) for t in out), notes


def main(argv):
    toks = []
    for a in argv:
        toks.extend(shlex.split(a))  # accept one quoted string or many tokens
    out, notes = clamp_args(toks)
    for n in notes:
        print(f"[vram-guard] clamping {n} (24GB ceiling; k>8 OOMs on the first batch)",
              file=sys.stderr)
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
