#!/usr/bin/env python3
"""fastMRI knee DICOM exams -> dense80 volumes on the pod's local volume. Nothing image-derived is uploaded.

    python fastmri_dicom_cache.py download    # both DICOM batches: resumable download, extract, delete the archive
    python fastmri_dicom_cache.py inventory   # one header per series -> /workspace/fastmri/inventory.json
                                              # + a counts-only summary in $MEGA_JOB_DIR (no images, no patient fields)
    python fastmri_dicom_cache.py build       # dense80 shards -> /workspace/fastmri/dense80/{vols,masks,ids}_e<k>.npy

Links come from env FASTMRI_LINKS (whitespace separated) and are never printed or written to a log; only links whose
file name contains "DICOM" are used. Volumes follow outputs/kernels/dense80_cache (same slots, ordering, 2-98 % window,
140 mm crop, 336 px) so a model sees fastMRI studies in the geometry it was trained on. Study ids get the prefix
"fm_". The fastMRI data-sharing agreement forbids redistribution: keep everything under /workspace/fastmri.
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

FM = Path(os.environ.get("FASTMRI_DIR", "/workspace/fastmri"))
DL, DCM, CACHE = FM / "dl", FM / "dicom", FM / "dense80"
JOB = Path(os.environ.get("MEGA_JOB_DIR", str(FM)))
IMG, CROP_MM, SPAN_LO, SPAN_HI = 336, 140.0, 0.02, 0.98
SLOTS = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15)]
MAXS = sum(s[2] for s in SLOTS)
SHARD = 1000


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ download
def _links() -> list[tuple[str, str]]:
    out = []
    for u in os.environ.get("FASTMRI_LINKS", "").split():
        name = Path(urlparse(u).path).name
        if u.startswith("http") and "DICOM" in name:
            out.append((name, u))
    if not out:
        raise SystemExit("FASTMRI_LINKS holds no DICOM link")
    return out


def _fetch(name: str, url: str) -> None:
    import requests
    dest, mark = DL / name, DCM / (name + ".EXTRACTED")
    if mark.exists():
        say(f"{name}: already extracted")
        return
    DL.mkdir(parents=True, exist_ok=True)
    total = None
    for t in range(1, 41):
        pos = dest.stat().st_size if dest.exists() else 0
        if total is not None and pos >= total:
            break
        try:
            with requests.get(url, headers={"Range": f"bytes={pos}-"}, stream=True, timeout=300) as r:
                if r.status_code == 416:
                    break
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                cr = r.headers.get("Content-Range", "")
                total = int(cr.split("/")[-1]) if "/" in cr else pos + int(r.headers.get("Content-Length", 0))
                t0, got = time.time(), 0
                with open(dest, "ab" if r.status_code == 206 and pos else "wb") as f:
                    for chunk in r.iter_content(16 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - t0 > 120:
                            say(f"{name}: {dest.stat().st_size / 1e9:.1f} / {total / 1e9:.1f} GB ({got / (time.time() - t0) / 1e6:.0f} MB/s)")
                            t0, got = time.time(), 0
        except Exception as e:  # noqa: BLE001  (never print the URL: report the error type only)
            say(f"{name}: try {t} interrupted at {pos / 1e9:.1f} GB ({type(e).__name__})")
            time.sleep(min(120, 10 * t))
    if total is None or dest.stat().st_size < total:
        raise SystemExit(f"{name}: download incomplete")
    out = DCM / name.split(".")[0]
    out.mkdir(parents=True, exist_ok=True)
    say(f"{name}: downloaded {total / 1e9:.1f} GB, extracting")
    rc = subprocess.run(f"xz -T0 -dc {dest} | tar -x -C {out}", shell=True).returncode
    if rc != 0:
        raise SystemExit(f"{name}: extraction failed (rc {rc})")
    mark.write_text(time.strftime("%FT%T"))
    dest.unlink()
    say(f"{name}: extracted")


def download() -> None:
    DCM.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda nu: _fetch(*nu), _links()))
    (FM / "DOWNLOADED").write_text(time.strftime("%FT%T"))
    say(subprocess.run(["du", "-sh", str(DCM)], capture_output=True, text=True).stdout.strip())


# ------------------------------------------------------------------ inventory
def _plane(iop) -> str:
    try:
        n = np.abs(np.cross(np.array(iop[:3], float), np.array(iop[3:], float)))
        return ("Sagittal", "Coronal", "Axial")[int(n.argmax())]
    except Exception:  # noqa: BLE001
        return "Unknown"


def _series(d: str) -> dict | None:
    import pydicom
    fs = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    for f in fs[:3]:
        try:
            h = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True, force=True)
            if getattr(h, "Rows", None) is None:
                continue
            g = lambda k, default=None: getattr(h, k, default)  # noqa: E731
            return {"dir": d, "n": len(fs), "study": str(g("StudyInstanceUID", "")), "series": str(g("SeriesInstanceUID", "")),
                    "desc": str(g("SeriesDescription", "")), "protocol": str(g("ProtocolName", "")),
                    "plane": _plane(g("ImageOrientationPatient")), "rows": int(g("Rows", 0)), "cols": int(g("Columns", 0)),
                    "ps": float((g("PixelSpacing") or [0])[0]), "thick": float(g("SliceThickness", 0) or 0),
                    "tr": float(g("RepetitionTime", 0) or 0), "te": float(g("EchoTime", 0) or 0),
                    "ti": float(g("InversionTime", 0) or 0), "scanopt": str(g("ScanOptions", "")),
                    "seqname": str(g("SequenceName", "")), "seqvar": str(g("SequenceVariant", "")),
                    "acq": str(g("MRAcquisitionType", "")), "body": str(g("BodyPartExamined", "")),
                    "maker": str(g("Manufacturer", "")), "field": float(g("MagneticFieldStrength", 0) or 0),
                    "tsuid": str(getattr(getattr(h, "file_meta", None), "TransferSyntaxUID", ""))}
        except Exception:  # noqa: BLE001
            continue
    return None


def classify(s: dict) -> tuple[int, int]:
    """(Fluid_Sensitive, Fat_Suppression) in the competition's sense, from the header.
    Fat-suppressed PD / T2 / STIR series fill the fluid slots; non-fat-suppressed PD and T1 fill the others,
    which is how the two slot types look in the competition data."""
    text = f"{s['desc']} {s['protocol']} {s['scanopt']} {s['seqname']}".upper()
    fatsat = int(bool(re.search(r"\bFS\b|FAT.?SAT|FATSAT|FTST|SPAIR|SPIR|STIR|\bFSAT\b|_FS|FS_", text)) or s["ti"] > 0 and "STIR" in text
                 or "FS" in s["scanopt"].upper().split("\\"))
    if re.search(r"NON.?FAT.?SAT|NON.?FS\b|NO.?FAT.?SAT", f"{s['desc']} {s['protocol']}".upper()):   # "SAG PD FSE NON FAT SAT"
        fatsat = 0
    t1 = bool(re.search(r"\bT1\b|T1_|_T1", text)) or (0 < s["te"] < 25 and 0 < s["tr"] < 900)
    fluid = int(fatsat or (bool(re.search(r"\bT2\b|T2_|_T2", text)) and not t1))
    return fluid, fatsat


def inventory() -> None:
    # FASTMRI_WALK / FASTMRI_INV_NAME: inventory of one finished batch while the other is still extracting
    root = os.environ.get("FASTMRI_WALK", str(DCM))
    name = os.environ.get("FASTMRI_INV_NAME", "inventory")
    dirs = [d for d, _, fs in os.walk(root) if fs]
    say(f"{len(dirs)} directories with files")
    with Pool(os.cpu_count()) as pool:
        rows = [r for r in pool.imap_unordered(_series, dirs, chunksize=64) if r]
    for r in rows:
        r["fluid"], r["fatsat"] = classify(r)
    (FM / f"{name}.json").write_text(json.dumps(rows))
    studies = collections.defaultdict(list)
    for r in rows:
        studies[r["study"]].append(r)
    c = collections.Counter
    norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[0-9]+(\.[0-9]+)?MM|\b[LR]T?\b|LEFT|RIGHT", "", t.upper())).strip()  # noqa: E731
    slot = c()
    for rs in studies.values():
        have = {(r["plane"], r["fluid"]) for r in rs}
        slot.update(f"{p}/{'fluid' if f else 'other'}" for p, f in have)
    summary = {
        "series": len(rows), "studies": len(studies),
        "series_per_study": dict(sorted(c(len(v) for v in studies.values()).items())),
        "plane": dict(c(r["plane"] for r in rows)),
        "plane_fluid_fatsat": dict(c(f"{r['plane']}|fluid{r['fluid']}|fs{r['fatsat']}" for r in rows).most_common()),
        "studies_with_slot": dict(slot),
        "desc_top": c(f"{norm(r['desc'])} || {r['plane']} | fluid{r['fluid']} fs{r['fatsat']}" for r in rows).most_common(120),
        "scanopt_top": c(r["scanopt"] for r in rows).most_common(25),
        "seqname_top": c(r["seqname"] for r in rows).most_common(25),
        "acq": dict(c(r["acq"] for r in rows)), "body": dict(c(r["body"] for r in rows).most_common(10)),
        "maker": dict(c(r["maker"] for r in rows).most_common(8)), "field": dict(c(r["field"] for r in rows).most_common(6)),
        "tsuid": dict(c(r["tsuid"] for r in rows).most_common(8)),
        "slices_per_series_q": [int(x) for x in np.percentile([r["n"] for r in rows], [1, 25, 50, 75, 99])],
        "ps_q": [round(float(x), 3) for x in np.percentile([r["ps"] for r in rows], [1, 25, 50, 75, 99])],
        "te_by_group": {k: [round(float(x), 1) for x in np.percentile(v, [10, 50, 90])] for k, v in
                        ((k, [r["te"] for r in rows if f"fluid{r['fluid']}|fs{r['fatsat']}" == k]) for k in
                         ("fluid1|fs1", "fluid1|fs0", "fluid0|fs0")) if v},
    }
    JOB.mkdir(parents=True, exist_ok=True)
    (JOB / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))
    say(f"inventory: {len(rows)} series, {len(studies)} studies -> {JOB / (name + '_summary.json')}")


# ------------------------------------------------------------------ build (mirrors dense80_cache_template.py)
def _order(sdir: str):
    import pydicom
    recs, ps_list = [], []
    for f in os.listdir(sdir):
        p = os.path.join(sdir, f)
        try:
            h = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                pos = float(np.dot(np.array(ipp, float), np.cross(np.array(iop[:3], float), np.array(iop[3:], float))))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps)
            recs.append((pos, p, ps))
        except Exception:  # noqa: BLE001
            recs.append((0.0, p, 0.5))
    recs.sort(key=lambda x: x[0])
    return [(f, ps) for _, f, ps in recs], (float(np.median(ps_list)) if ps_list else 0.5)


def _px(f: str):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(f, force=True)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def _crop(a, ps):
    import cv2
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def _pick(rows, plane, fluid, used):
    cands = [r for r in rows if r["plane"] == plane and r["dir"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if r["fluid"] == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _study(rows):
    import cv2
    cv2.setNumThreads(1)
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    used, offset, err = set(), 0, ""
    try:
        for plane, fluid, count in SLOTS:
            rec = _pick(rows, plane, fluid, used)
            if rec is None:
                offset += count
                continue
            used.add(rec["dir"])
            files, med_ps = _order(rec["dir"])
            n = len(files)
            if not n:
                offset += count
                continue
            lo = int(n * SPAN_LO)
            hi = max(int(n * SPAN_HI) - 1, lo)
            picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else np.zeros(count, dtype=int)
            cache = {}
            for p in sorted(set(picks.tolist())):
                p = min(int(p), n - 1)
                try:
                    cache[p] = _px(files[p][0])
                except Exception:  # noqa: BLE001
                    cache[p] = None
            arrays = [cache.get(min(int(p), n - 1)) for p in picks]
            valid = [a for a in arrays if a is not None]
            lo_v, hi_v = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0]) if valid else (0.0, 1.0)
            for j, (a, p) in enumerate(zip(arrays, picks)):
                if a is None or offset + j >= MAXS:
                    continue
                ps = files[min(int(p), n - 1)][1]
                vol[offset + j] = (_crop(np.clip((a - lo_v) / (hi_v - lo_v + 1e-6), 0, 1), ps if ps > 0 else med_ps) * 255).astype(np.uint8)
            offset += count
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return vol, (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8), err


def build() -> None:
    # FASTMRI_INV_NAME / FASTMRI_SHARD0: build one batch's shards while the other batch is still extracting
    name = os.environ.get("FASTMRI_INV_NAME", "inventory")
    shard0 = int(os.environ.get("FASTMRI_SHARD0", "0"))
    rows = json.loads((FM / f"{name}.json").read_text())
    for r in rows:      # the rule may have been updated since the inventory ran
        r["fluid"], r["fatsat"] = classify(r)
    studies = collections.defaultdict(list)
    for r in rows:
        if r["plane"] != "Unknown" and r["n"] >= 5 and r["acq"].upper() != "3D":
            studies[r["study"]].append(r)
    order = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
    ids = sorted(s for s, rs in studies.items() if len({r["plane"] for r in rs}) >= 2)
    for s in ids:
        studies[s].sort(key=lambda r: (order[r["plane"]], -r["fluid"], -r["fatsat"], -r["n"], r["dir"]))
    CACHE.mkdir(parents=True, exist_ok=True)
    say(f"{len(ids)} studies with >= 2 planes of {len(studies)} | cpus {os.cpu_count()}")
    t0, errors, slots = time.time(), 0, []
    with Pool(os.cpu_count()) as pool:
        for k in range(0, len(ids), SHARD):
            part = ids[k:k + SHARD]
            e = shard0 + k // SHARD
            vols = np.lib.format.open_memmap(CACHE / f"vols_e{e}.npy", mode="w+", dtype=np.uint8, shape=(len(part), MAXS, IMG, IMG))
            masks = np.zeros((len(part), MAXS), np.uint8)
            for i, (vol, mask, err) in enumerate(pool.imap(_study, [studies[s] for s in part], chunksize=2)):
                vols[i], masks[i] = vol, mask
                errors += bool(err)
            vols.flush()
            del vols
            np.save(CACHE / f"masks_e{e}.npy", masks)
            np.save(CACHE / f"ids_e{e}.npy", np.array(["fm_" + s for s in part]))
            slots.append(float(masks.sum(1).mean()))
            say(f"shard {e}: {k + len(part)}/{len(ids)} | {(time.time() - t0) / 60:.1f} min | mean filled slots {slots[-1]:.1f} | errors {errors}")
    info = {"studies": len(ids), "shards": len(slots), "first_shard": shard0, "mean_slots": slots, "errors": errors,
            "built": time.strftime("%FT%T")}
    (CACHE / ("READY" if name == "inventory" else f"READY_{name}")).write_text(json.dumps(info))
    JOB.mkdir(parents=True, exist_ok=True)
    (JOB / f"cache_summary_{name}.json").write_text(json.dumps(info, indent=1))
    say(f"cache ready: {info}")


if __name__ == "__main__":
    {"download": download, "inventory": inventory, "build": build}[sys.argv[1]]()
