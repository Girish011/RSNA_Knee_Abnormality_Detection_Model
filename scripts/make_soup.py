#!/usr/bin/env python3
"""Uniform weight soup of full-data members → one checkpoint for the D4 maxspan swap.

The public stack runs its forward + reverse maxspan arms through ONE loaded model, so several
members can only enter that slot as a single checkpoint. Members fine-tuned from the same
init (public v5) for the same short schedule are soup-compatible (Wortsman et al. 2022,
"model soups"); we average every floating tensor, including BatchNorm running stats.

    python scripts/make_soup.py --name bestof4 --members m0b16e9a09f1,m470c713b9fe,...
    python scripts/make_soup.py --name bestof4 --members ... --upload

Writes ``outputs/soups/<name>/soup<name>/raptor_ft_soup<name>.pt`` and, with --upload,
creates the private dataset ``girishbose/rsna-knee-soup-<name>`` and registers the soup in
``agent/runs/members.json`` done (tag ``soup<name>``) so the kernel builder resolves it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
MEMBERS = REPO / "agent" / "runs" / "members.json"
USER = "girishbose"
KAGGLE = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))


def fetch(tag: str, info: dict, cache: Path) -> Path:
    dest = cache / tag
    f = dest / info["weights"]
    if not f.exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run([KAGGLE, "datasets", "download", info["dataset"], "-f", f"{tag}/{info['weights']}",
                        "-p", str(dest), "--unzip", "-q"], check=True)
        for z in dest.glob("*.zip"):
            shutil.unpack_archive(z, dest)
            z.unlink()
    assert f.exists(), f
    return f


def soup(paths: list[Path]) -> dict:
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    arch = {c.get("arch") for c in cks}
    if len(arch) != 1:
        raise SystemExit(f"members differ in arch: {arch}")
    keys = set(cks[0]["model"])
    if any(set(c["model"]) != keys for c in cks):
        raise SystemExit("members differ in state-dict keys")
    avg = {}
    for k in cks[0]["model"]:
        ts = [c["model"][k] for c in cks]
        avg[k] = (torch.stack([t.float() for t in ts]).mean(0).to(ts[0].dtype)
                  if ts[0].is_floating_point() else ts[0].clone())
    out = {k: v for k, v in cks[0].items() if k != "model"}
    out.update(model=avg, src="soup:" + ",".join(p.stem for p in paths), soup_of=[p.stem for p in paths],
               gold_auc=None, aucs={})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", required=True)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    done = json.loads(MEMBERS.read_text())["done"]
    tags = a.members.split(",")
    missing = [t for t in tags if t not in done]
    if missing or len(tags) < 2:
        raise SystemExit(f"need >=2 finished members; missing {missing}")
    inits = {done[t].get("init") for t in tags}
    if len(inits) != 1 or None in inits:
        raise SystemExit(f"soup needs a shared warm start; got inits {inits}")
    cache = REPO / "outputs" / "soups" / "_cache"
    paths = [fetch(t, done[t], cache) for t in tags]
    tag = f"soup{a.name}"
    d = REPO / "outputs" / "soups" / a.name
    (d / tag).mkdir(parents=True, exist_ok=True)
    wname = f"raptor_ft_{tag}.pt"
    torch.save(soup(paths), d / tag / wname)
    print(f"wrote {d / tag / wname}")
    if not a.upload:
        return
    slug = f"{USER}/rsna-knee-soup-{a.name}"
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"rsna-knee-soup-{a.name}", "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
    subprocess.run([KAGGLE, "datasets", "create", "-p", str(d), "--dir-mode", "zip", "-q"], check=True)
    # Kaggle flattens a single-file subfolder: the soup sits at the dataset root.
    # Registered through the loop's ops file (a direct members.json write races a running tick).
    sys.path.insert(0, str(REPO / "src"))
    from rsna_knee.agent import members as M
    M.request_op({"op": "add_done", "record": {
        "tag": tag, "dataset": slug, "weights": wname, "path": wname, "soup_of": tags,
        "genome": done[tags[0]]["genome"], "init": done[tags[0]].get("init"), "seed": None,
        "gold_dev": None, "holdout_auc": None}})
    print(f"uploaded {slug}; member {tag} registers at the loop's next tick")


if __name__ == "__main__":
    main()
