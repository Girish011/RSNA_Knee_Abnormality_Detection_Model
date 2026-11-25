#!/usr/bin/env python3
"""Kaggle T4 x 2 kernel that trains the public 2.5D ConvNeXt reader with ITS train.py / knee.py (9 Oct).

    python scripts/build_cx_train.py --name cx-train-a --runs "3:0,4:0" [--args "--backbone convnext_tiny"] [--push]

--runs = "fold:seed" (one run: their batch of 8 split over both T4s with DataParallel - their defaults need more than
one 16 GB card) or "fold:seed,fold:seed" (one GPU each, only for settings that fit); fold -1 = all weak-labelled studies. Changes to their
code, all mechanical: paths; the cache is read from our per-study npz shards through StudyDataset's `loader=` hook;
ImageNet weights from a file (no internet); float16 autocast + GradScaler instead of bfloat16 (T4 has no bf16);
gradient checkpointing in the encoder (identical gradients; their batch does not fit a 16 GB card otherwise).
Labels = data/processed/cx/labels_v0.csv built with their make_labels.py, embedded in the kernel.
"""
import argparse, base64, gzip, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = r'''
import base64, glob, gzip, os, subprocess, sys, time
import pandas as pd
RUNS, ARGS, TAG, NSH = __RUNS__, __ARGS__, __TAG__, __NSH__
W = "/kaggle/working"
os.makedirs(W + "/src", exist_ok=True); os.makedirs(W + "/work", exist_ok=True)
rd = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
root = next(d for d in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection") if os.path.exists(d + "/train_series.csv"))
bb = glob.glob("/kaggle/input/**/convnext_tiny.in12k_ft_in1k.pth", recursive=True)[0] if "convnext_small" not in ARGS else glob.glob("/kaggle/input/**/convnext_small.in12k_ft_in1k.pth", recursive=True)[0]
open(W + "/work/labels_v0.csv", "wb").write(gzip.decompress(base64.b64decode(__LABELS__)))
metas = sorted(glob.glob("/kaggle/input/**/meta_s*.csv", recursive=True))
assert len(metas) == NSH, f"{len(metas)} cache shards mounted, expected {NSH}"
meta = pd.concat([pd.read_csv(m) for m in metas]); meta.to_csv(W + "/work/cache384_train_meta.csv", index=False)
npz = {os.path.basename(p)[:-4]: p for p in glob.glob("/kaggle/input/**/*.npz", recursive=True)}
print("recipe", rd, "| studies with cache", len(npz), "| series", len(meta), "| errors", int(meta.get("error", pd.Series(dtype=str)).notna().sum()), "| bb", bb, flush=True)
pd.Series(npz).to_json(W + "/work/npz_index.json")
pd.Series(dict(zip(meta.series, meta.study))).to_json(W + "/work/series_study.json")

def patch(text, old, new, n=1):
    assert text.count(old) == n, (old, text.count(old))
    return text.replace(old, new)

k = open(rd + "/knee.py").read()
k = patch(k, "timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)",
          "timm.create_model(backbone, pretrained=pretrained, num_classes=0, pretrained_cfg_overlay=dict(file=os.environ['CX_BB']), **kw)\n"
          "        self.enc.set_grad_checkpointing(True)   # same gradients (checked: max diff 0.0), a fraction of the memory")
open(W + "/src/knee.py", "w").write("import os\n" + k)
t = open(rd + "/train.py").read()
t = patch(t, 'DATA = "/mnt/ssd/rsna_knee/data"', f'DATA = "{root}"')
t = patch(t, 'WORK = "/mnt/ssd/rsna_knee/work"', f'WORK = "{W}/work"')
t = patch(t, 'RUNS = "/mnt/ssd/rsna_knee/runs"', f'RUNS = "{W}/runs"')
t = patch(t, "args.k, train=True)", "args.k, train=True, loader=_load)")
t = patch(t, "None, args.k_eval)", "None, args.k_eval, loader=_load)")
t = patch(t, "def macro_auc(y, p):", """_NPZ = json.load(open(WORK + "/npz_index.json")); _S2S = json.load(open(WORK + "/series_study.json"))


def _load(sr):   # our cache: one compressed file per study, key = series uid
    return np.load(_NPZ[_S2S[sr]])[sr]


def macro_auc(y, p):""")
if not __BF16__:   # T4: no bfloat16 -> float16 autocast with a gradient scaler
    t = patch(t, 'with torch.autocast("cuda", dtype=torch.bfloat16):', 'with torch.autocast("cuda", dtype=torch.float16):', 2)
    t = patch(t, "            loss.backward()\n            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)\n            opt.step()\n",
              "            _sc.scale(loss).backward()\n            _sc.unscale_(opt)\n            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)\n            _sc.step(opt)\n            _sc.update()\n")
    t = patch(t, "    steps = args.epochs * len(train_dl)\n", "    steps = args.epochs * len(train_dl)\n    _sc = torch.amp.GradScaler('cuda')\n")
if len(RUNS) == 1:   # bs 8 x 6 slots x 8 windows does not fit one 16 GB T4 with gradients: split the batch over both
    t = patch(t, "    model = KneeNet(args.backbone).cuda().to(memory_format=torch.channels_last)\n",
              "    model = KneeNet(args.backbone).cuda().to(memory_format=torch.channels_last)\n    _dp = torch.nn.DataParallel(model)\n")
    t = patch(t, "logit = model(x, mask, pos, res=args.res, train=True)", "logit = _dp(x, mask, pos, res=args.res, train=True)")
if "import json" not in t:
    t = "import json\n" + t
open(W + "/src/train.py", "w").write(t)
procs = []
for g, (fold, seed) in enumerate(RUNS):
    env = dict(os.environ, CX_BB=bb, PYTHONPATH=W + "/src")
    if len(RUNS) > 1:
        env["CUDA_VISIBLE_DEVICES"] = str(g)
    cmd = [sys.executable, W + "/src/train.py", "--name", TAG, "--fold", str(fold), "--seed", str(seed), "--workers", "4" if len(RUNS) == 1 else "2", *ARGS.split()]
    lg = open(f"{W}/train_f{fold}_s{seed}.log", "w")
    procs.append((fold, seed, subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)))
    print("started", " ".join(cmd[1:]), "on GPU", g, flush=True)
t0 = time.time()
while any(p.poll() is None for _, _, p in procs):
    time.sleep(600)
    for fold, seed, p in procs:
        tail = [l for l in open(f"{W}/train_f{fold}_s{seed}.log").read().splitlines() if l.strip().startswith("ep ") or "Error" in l or "error" in l][-1:]
        print(f"[{(time.time() - t0) / 60:.0f} min] fold {fold} seed {seed}: {tail}", flush=True)
for fold, seed, p in procs:
    print(f"fold {fold} seed {seed} exit {p.returncode}", flush=True)
    print("\n".join(open(f"{W}/train_f{fold}_s{seed}.log").read().splitlines()[-6:]), flush=True)
for f in ("src", "work"):
    subprocess.run(["rm", "-rf", f"{W}/{f}"])
print("outputs:", sorted(glob.glob(W + "/runs/*/*")), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--tag", default="cx_v0")
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    runs = [tuple(int(v) for v in r.split(":")) for r in a.runs.split(",")]
    assert 1 <= len(runs) <= 2
    lab = base64.b64encode(gzip.compress((REPO / "data/processed/cx/labels_v0.csv").read_bytes())).decode()
    src = (SRC.replace("__RUNS__", repr(runs)).replace("__ARGS__", repr(a.args)).replace("__TAG__", repr(a.tag))
           .replace("__NSH__", str(a.shards)).replace("__LABELS__", repr(lab)).replace("__BF16__", "False"))
    compile(src, a.name, "exec")
    d = REPO / "outputs" / "kernels" / a.name
    d.mkdir(parents=True, exist_ok=True)
    code = a.name.replace("-", "_") + ".py"
    (d / code).write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"girishbose/{a.name}", "title": a.name, "code_file": code, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "girishbose/rsna-knee-timm-pretrained"],
        "kernel_sources": [f"girishbose/cx-cache-s{k}" for k in range(a.shards)], "model_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=1))
    print(f"wrote {d} | runs {runs} | args '{a.args}' | script {len(src) // 1024} kB")
    if a.push:
        r = subprocess.run([str(Path(sys.executable).with_name("kaggle")), "kernels", "push", "-p", str(d)], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip().splitlines()[-1][:120])


if __name__ == "__main__":
    main()
