#!/usr/bin/env python3
"""Fork a public stack-component training notebook onto OUR label set (label transplant).

Every public stack component is trained on public report-derived labels; our image-informed
label set (bestof_v1) lifted public v5 from 0.926 to 0.937 on the 58 experts. This script
pulls a public trainer, swaps its target builder for one that reads our labels, and writes a
private kernel ready to push (Kaggle GPU). The expert gold still overrides on the 58 studies,
exactly as upstream does, so the trainer's own gold OOF AUC stays comparable to the public run.

    python scripts/fork_public_trainer.py e13 --labels labels_bestof_v1.csv
    kaggle kernels push -p outputs/kernels/fork-e13-bestof_v1

Supported:
  e13     sofiaanjenje/rsna-knee-e13-train  (RadImageNet ResNet-50 E11 heads -> v52_e11_heads.pt)
  e9main  same notebook, V52 main mode     (main RadImageNet heads -> v52_radimagenet_heads.pt)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "girishbose"
LABEL_DS = f"{USER}/rsna-knee-labels-ours"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))

TARGETS = {
    "e13": {"src": "sofiaanjenje/rsna-knee-e13-train", "anchor": "def make_targets(train):\n",
            "end": "\n\ndef report_groups(train):"},
    # Same notebook in V52 main mode: trains the stack's MAIN heads (v52_radimagenet_heads.pt, baseline
    # gold OOF 0.854). main_v52 only runs when no v52_radimagenet_heads.pt is mounted, so drop that dataset.
    "e9main": {"src": "sofiaanjenje/rsna-knee-e13-train", "anchor": "def make_targets(train):\n",
               "end": "\n\ndef report_groups(train):",
               "replace": [('ARM_MODE = "e11"', 'ARM_MODE = "e9main"'),
                           ("if ARM_MODE == 'e11':\n    print(", "if ARM_MODE in ('e11', 'e9main'):\n    print(")],
               "drop_sources": ["antoinegg1/rsna-knee-e9-radimagenet-heads-v15"]},
}

# Single-source targets: our soft labels, cell weight from certainty (upstream also used
# three-teacher agreement, which a single source does not have). Gold wins with weight 3.
OUR_TARGETS = '''def make_targets(train):
    """OURS: image-informed label set __LABELS__ (label transplant); image-read gold always wins."""
    uid = "StudyInstanceUID"
    ours = pd.read_csv(find_input_file("__LABELS__"), dtype={uid: str})
    if ours[uid].duplicated().any():
        raise ValueError("duplicate study in our label set")
    aligned = train[[uid]].merge(ours[[uid] + TARGETS], on=uid, how="left")
    y = aligned[TARGETS].to_numpy(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if np.isnan(y[~gold]).any():
        raise ValueError(f"our labels miss {int(np.isnan(y[~gold]).any(1).sum())} non-gold studies")
    y = np.nan_to_num(y, nan=0.5)
    certainty = np.clip(2.0 * np.abs(y - .5), 0, 1)
    w = (.15 + .85 * certainty).astype(np.float32)
    y[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    w[gold] = 3.0
    print(f"[ours] targets from __LABELS__: {len(y)} studies, {int(gold.sum())} gold", flush=True)
    return y, w, gold'''


def fork(kind: str, labels: str) -> Path:
    spec = TARGETS[kind]
    stem = Path(labels).stem.removeprefix("labels_")
    name = f"fork-{kind}-{stem}".replace("_", "-")
    d = REPO / "outputs" / "kernels" / name
    d.mkdir(parents=True, exist_ok=True)
    src_dir = d / "_upstream"
    subprocess.run([KAGGLE, "kernels", "pull", spec["src"], "-m", "-p", str(src_dir)], check=True)
    meta = json.loads((src_dir / "kernel-metadata.json").read_text())
    nb = json.loads((src_dir / meta["code_file"]).read_text())
    n = 0
    for c in nb["cells"]:
        t = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if t.count(spec["anchor"]) == 1:
            i = t.index(spec["anchor"])
            j = t.index(spec["end"], i)
            t = t[:i] + OUR_TARGETS.replace("__LABELS__", labels) + t[j:]
            c["source"] = t
            n += 1
    if n != 1:
        raise SystemExit(f"make_targets anchor found in {n} cells (want 1)")
    for a, b in spec.get("replace", []):
        hits = 0
        for c in nb["cells"]:
            t = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if a in t:
                c["source"] = t.replace(a, b)
                hits += t.count(a)
        if hits != 1:
            raise SystemExit(f"replace anchor {a!r} found {hits}x (want 1)")
    code_file = f"{name}.ipynb"
    (d / code_file).write_text(json.dumps(nb, indent=1))
    sources = [s for s in meta.get("dataset_sources", []) if s and s not in spec.get("drop_sources", [])]
    sources += [LABEL_DS]
    meta2 = {k: v for k, v in meta.items() if k not in ("id_no",)}
    meta2.update(id=f"{USER}/{name}", title=name, code_file=code_file, is_private=True,
                 enable_internet=False, dataset_sources=list(dict.fromkeys(sources)))
    (d / "kernel-metadata.json").write_text(json.dumps(meta2, indent=1))
    (d / "FORK.json").write_text(json.dumps({"upstream": spec["src"], "labels": labels,
                                             "label_dataset": LABEL_DS}, indent=1))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(TARGETS))
    ap.add_argument("--labels", default="labels_bestof_v1.csv")
    a = ap.parse_args()
    d = fork(a.kind, a.labels)
    print(f"built {d}\npush: kaggle kernels push -p {d}")


if __name__ == "__main__":
    main()
