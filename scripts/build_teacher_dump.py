#!/usr/bin/env python3
"""Kaggle GPU kernel: score TRAIN studies with the two expert-trained public checkpoints of
`nartaa/rsna-knee-publication-swa-weights-20261007` (384 px "accuracy", 224 px "efficiency").

    python scripts/build_teacher_dump.py --name teacher-dump --n 600      # 58 expert studies + 542 others
    python scripts/build_teacher_dump.py --name teacher-dump-all --n 0    # every train study
    kaggle kernels push -p outputs/kernels/<name>
    kaggle kernels output girishbose/<name> -p outputs/kernels/<name>/_out   # teacher_dump.npz + log

The preprocessing, model and window code are the forked 0.950 notebook's own cell (`outputs/kernels/pub/rsna-versia-5`),
unchanged; only the driver is ours. Arms written to teacher_dump.npz (raw sigmoid means over K windows, (N, 12)):
  t384   384 checkpoint, centre 320 px of the 384 px corpus (the notebook's protocol), plain pass
  t384m  same, mean of the plain and the left-right mirrored pass (how the notebook scores)
  t224A / t224B / t224C  224 checkpoint on the full field resized / the central 0.8 resized / the central 224 px
The 224 checkpoint ships without its inference code; the three inputs are tried so the data shows which one it was
trained on (agreement with the 384 model, and the 58 expert studies).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
USER = "girishbose"
BASE = REPO / "outputs" / "kernels" / "pub" / "rsna-versia-5"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
DRIVER = r'''

# ============================ driver (ours) ============================
IDS = __IDS__            # [] = every study in train.csv
K_DUMP = __K__
ARM384 = ("raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef")
ARM224 = ("raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt", "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6")
MODES = __MODES__      # inputs tried for the 224 checkpoint
NAMES = ["t384", "t384m"] + ["t224" + m for m in MODES]


def load_arm(fname, sha, img_size, device, say):
    import hashlib
    p = find_weight_file(fname)
    with open(p, "rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    assert got == sha, f"{fname}: checkpoint bytes differ from the manifest"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if say:
        print(f"[ck] {fname} | " + " | ".join(f"{k}={v!r}"[:160] for k, v in ck.items() if k != "model" and not torch.is_tensor(v)), flush=True)
    bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, in_chans=3, img_size=img_size, global_pool="avg")
    m = RaptorClassifier(bb, F_dim=bb.num_features)
    m.load_state_dict(ck["model"], strict=True)
    del ck; gc.collect()
    return m.eval().to(device)


def win224(tri, mode, dev):
    """tri: (k, 3, 384, 384) uint8 on dev -> (k, 3, 224, 224) normalised."""
    x = tri.float() / 255.0
    if mode == "B":
        lo = (IMG - 307) // 2; x = x[..., lo:lo + 307, lo:lo + 307]
    elif mode == "C":
        lo = (IMG - 224) // 2; x = x[..., lo:lo + 224, lo:lo + 224]
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    return (x - _MEAN.to(dev)) / _STD.to(dev)


def dump_main():
    import pandas as pd, multiprocessing as mp
    t0 = time.time()
    ROOT = next(b for b in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
                            "/kaggle/input/rsna-knee-abnormality-detection") if os.path.exists(b + "/train.csv"))
    tsdir = ROOT + "/train_series"
    assert os.path.isdir(tsdir), os.listdir(ROOT)
    tr = pd.read_csv(ROOT + "/train.csv"); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    ids = IDS or tr["StudyInstanceUID"].tolist()
    assert set(ids) <= set(tr["StudyInstanceUID"])
    ser = pd.read_csv(ROOT + "/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str); ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    N = len(ids); pos = {s: i for i, s in enumerate(ids)}
    out = {n: np.full((N, len(LAB)), np.nan, np.float32) for n in NAMES}
    devs = [f"cuda:{g}" for g in range(torch.cuda.device_count())]
    print(f"studies {N} | K {K_DUMP} | devices {devs}", flush=True)
    models = {d: (load_arm(*ARM384, INPUT_RES, d, d == devs[0]), load_arm(*ARM224, 224, d, d == devs[0])) for d in devs}
    print(f"models loaded | {time.time() - t0:.0f}s", flush=True)
    q = queue.Queue(maxsize=QUEUE_MAX); done = [0]; lock = threading.Lock()
    # at most 64 studies between the prep pool and the GPUs: imap_unordered buffers finished volumes (14 MB each)
    # without limit when the GPUs are the slower side - the 4,407-study run was killed for memory at ~700 (11 Oct)
    gate = threading.BoundedSemaphore(64)

    def save():
        np.savez_compressed("/kaggle/working/teacher_dump.npz", ids=np.array(ids), lab=np.array(LAB), **out)

    def feed():
        for s in ids:
            gate.acquire()
            yield s

    def consumer(d):
        m384, m224 = models[d]
        while True:
            item = q.get()
            if item is None:
                q.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                if mask.sum() >= 3:
                    p = infer_probs(m384, gpu_windows(vol, mask, K_DUMP, INPUT_RES, d), d)
                    pm = infer_probs(m384, gpu_windows(vol, mask, K_DUMP, INPUT_RES, d, mirror=True), d)
                    out["t384"][i] = p; out["t384m"][i] = 0.5 * (p + pm)
                    D = vol.shape[0]
                    cs = np.array([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, K_DUMP)])
                    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(d)
                    for mode in MODES:
                        out["t224" + mode][i] = infer_probs(m224, win224(tri, mode, d), d)
            except Exception as e:
                print(f"  [gpu {d}] {sid[-12:]} FAILED ({type(e).__name__}: {e})", flush=True)
            gate.release()
            with lock:
                done[0] += 1
                if done[0] % 500 == 0:
                    save()          # a partial file survives a killed run
                if done[0] % 100 == 0 or done[0] == N:
                    print(f"  [gpu] {done[0]}/{N} | {time.time() - t0:.0f}s", flush=True)
            q.task_done()

    threads = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]
    for th in threads: th.start()
    global _SER, _TSDIR
    _SER, _TSDIR = SER, tsdir
    with mp.get_context("fork").Pool(4, initializer=_pre_init) as pool:
        for sid, vol, mask in pool.imap_unordered(_pre_one, feed(), chunksize=1):
            q.put((sid, vol, mask))
    for _ in devs: q.put(None)
    q.join()
    save()
    for f in glob.glob("/kaggle/working/_fastread.py") + glob.glob("/kaggle/working/__pycache__/*"):
        os.remove(f)
    print(f"wrote teacher_dump.npz | {N} studies | missing {int(np.isnan(out['t384']).any(1).sum())} | {time.time() - t0:.0f}s", flush=True)

    # quick reading in the log: the 58 expert studies and agreement between the arms
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    g = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")
    gi = [i for i, s in enumerate(ids) if s in g.index]
    ok = ~np.isnan(out["t384"]).any(1)
    print(f"expert studies in this run: {len(gi)}", flush=True)
    for n in NAMES:
        P = out[n]
        rho = np.mean([spearmanr(P[ok, j], out["t384"][ok, j]).correlation for j in range(len(LAB))])
        line = f"{n:6s} | Spearman with t384 {rho:.3f}"
        if len(gi) >= 40:
            Y = g.loc[[ids[i] for i in gi], LAB].values
            a = [roc_auc_score(Y[:, j], P[gi, j]) for j in range(len(LAB))]
            line += f" | 58 experts macro {np.mean(a):.4f} | " + " ".join(f"{x:.3f}" for x in a)
        print(line, flush=True)


dump_main()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=600, help="studies: the 58 expert ones + random others (0 = all)")
    ap.add_argument("--k", type=int, default=32, help="windows per study")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--modes", default="ABC", help="224 inputs to score: A full field, B central 0.8 (the right one, 11 Oct), C central 224 px")
    a = ap.parse_args()
    nb = json.loads(next(BASE.glob("*.ipynb")).read_text())
    cell = max(("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"), key=len)
    tail = '\nif __name__ == "__main__":\n    main()'
    assert cell.rstrip().endswith(tail.strip()) and cell.count("def main():") == 1
    cell = cell.rstrip()[: -len(tail.strip())]
    ids: list[str] = []
    if a.n:
        tr = pd.read_csv(REPO / "data" / "raw" / "train.csv")
        tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
        gold = tr[tr[LAB].notna().all(axis=1)]["StudyInstanceUID"].tolist()
        rest = sorted(set(tr["StudyInstanceUID"]) - set(gold))
        random.Random(a.seed).shuffle(rest)
        ids = gold + rest[: max(0, a.n - len(gold))]
    src = cell + DRIVER.replace("__IDS__", repr(ids)).replace("__K__", str(a.k)).replace("__MODES__", repr(a.modes))
    code = a.name.replace("-", "_") + ".py"
    compile(src, code, "exec")
    d = REPO / "outputs" / "kernels" / a.name
    d.mkdir(parents=True, exist_ok=True)
    (d / code).write_text(src)
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    meta.update(id=f"{USER}/{a.name}", title=a.name, code_file=code, kernel_type="script", is_private=True,
                enable_internet=False, dataset_sources=["nartaa/rsna-knee-publication-swa-weights-20261007"])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {d} | {len(ids) or 'all'} studies | k {a.k} | push: kaggle kernels push -p {d}")


if __name__ == "__main__":
    main()
