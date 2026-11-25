#!/usr/bin/env python3
"""Build a private, offline Kaggle notebook for one frozen definition-reader EMA.

This only creates local source/metadata. Push/run/submit is a separate decision
after complete pilot review and a successful full inference rehearsal.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def build(out, slug, dataset, checkpoint):
    if not checkpoint.startswith("definition_") or not checkpoint.endswith("_ema.pt") or Path(checkpoint).name != checkpoint:
        raise ValueError("select exactly one fixed EMA filename")
    if "/" not in slug or "/" not in dataset:
        raise ValueError("kernel and private dataset need owner/slug")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    names = ["definition_geometry.py", "definition_reader.py", "definition_cache.py", "definition_predict.py"]
    files = {name: (REPO / "runpod" / name).read_bytes() for name in names}
    encoded = {name: base64.b64encode(blob).decode() for name, blob in files.items()}
    source = '''# Private, offline fixed-EMA inference. Generated from checked repository files.
import base64, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
CODE = Path('/kaggle/working/definition_code'); CODE.mkdir(exist_ok=True)
FILES = __FILES__
HASHES = __HASHES__
for name, encoded in FILES.items():
    blob = base64.b64decode(encoded)
    assert hashlib.sha256(blob).hexdigest() == HASHES[name]
    (CODE / name).write_bytes(blob)
NAME = __CHECKPOINT__
INPUT = Path('/kaggle/input')
matches = list(INPUT.rglob(NAME))
if not matches:
    # Campaign results preserve arm directories as zip files in Kaggle datasets.
    for archive in INPUT.rglob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            members = [i for i in z.infolist() if Path(i.filename).name == NAME and not i.is_dir()]
            for member in members:
                dest = Path('/kaggle/working') / NAME
                if dest in matches:
                    raise RuntimeError('ambiguous checkpoint archives')
                with z.open(member) as src, dest.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
                matches.append(dest)
if len(matches) != 1:
    raise RuntimeError('missing or ambiguous fixed EMA checkpoint')
roots = [p.parent for p in INPUT.rglob('sample_submission.csv') if (p.parent / 'test_series.csv').exists()]
if len(roots) != 1:
    raise RuntimeError('missing or ambiguous competition data')
cmd = [sys.executable, str(CODE / 'definition_predict.py'), '--competition-root', str(roots[0]),
       '--checkpoint', str(matches[0]), '--out', '/kaggle/working/submission.csv',
       '--receipt', '/kaggle/working/definition_receipt.json', '--max-seconds', '29400']
subprocess.run(cmd, check=True, timeout=30000)
print(Path('/kaggle/working/definition_receipt.json').read_text())
'''
    source = source.replace("__FILES__", repr(encoded)).replace("__HASHES__", repr({n: hashlib.sha256(b).hexdigest() for n,b in files.items()})).replace("__CHECKPOINT__", repr(checkpoint))
    compile(source, "definition_infer.py", "exec")
    (out / "definition_infer.py").write_text(source)
    metadata = {"id": slug, "title": "RSNA definition reader fixed EMA", "code_file": "definition_infer.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": False, "dataset_sources": [dataset],
                "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "build_receipt.json").write_text(json.dumps({"checkpoint": checkpoint, "dataset": dataset,
        "source_sha256": {n: hashlib.sha256(b).hexdigest() for n,b in files.items()},
        "entrypoint_sha256": hashlib.sha256(source.encode()).hexdigest(), "status": "built_not_executed"}, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    print(build(a.out, a.slug, a.dataset, a.checkpoint))
