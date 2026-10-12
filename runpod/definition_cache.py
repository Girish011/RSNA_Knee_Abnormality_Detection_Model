"""Frozen dense512 pixel preprocessing shared with the original training cache.

Functions copied from the project's dense512_cache_template.py without pixel
changes. Keep this module stable: training cache compatibility requires matching
series selection, slice selection, intensity scaling and physical crop.
"""
import glob
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
IMG = 512
GEOM = 20
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15),
         ("Coronal", 0, 10), ("Axial", -1, 15)]
MAXS = 80
SOURCE_SHA256 = 'bdd033a6d79fddf03e9a854c2729bfc389baeac8ceb00829109d32de68ddf8eb'

def order_and_meta(sdir):
    fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []; info = {}
    for f in fs:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            g = np.zeros(GEOM, np.float32)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                g[2:5] = np.array(ipp, float); g[5:8] = r; g[8:11] = c; g[19] = 1.0
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            psv = getattr(h, "PixelSpacing", None); ps = float(psv[0]) if psv is not None else 0.5
            g[0] = ps; g[1] = float(psv[1]) if psv is not None and len(psv) > 1 else ps
            try:
                g[16] = float(getattr(h, "SliceThickness", 0) or 0)
            except Exception:
                pass
            if not info:
                info = {k: str(getattr(h, k, "") or "") for k in
                        ("Laterality", "ImageLaterality", "SeriesDescription", "BodyPartExamined",
                         "PatientPosition", "Manufacturer")}
            ps_list.append(ps); recs.append((pos, f, ps, g))
        except Exception:
            recs.append((0.0, f, 0.5, np.zeros(GEOM, np.float32)))
    recs.sort(key=lambda x: x[0])
    med_ps = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps, g) for _, f, ps, g in recs], med_ps, info


def read_px(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps):
    h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    interp = cv2.INTER_AREA if cpx >= IMG else cv2.INTER_CUBIC
    return np.clip(cv2.resize(a, (IMG, IMG), interpolation=interp), 0, 1), (y0, x0, cpx, h, w)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def fill(vol, geom, offset, picks, cache, files, med_ps):
    arrays, spacings = [], []
    for p in picks:
        p = min(int(p), len(files) - 1)
        arrays.append(cache.get(p)); ps = files[p][1]; spacings.append(ps if ps > 0 else med_ps)
    valid = [a for a in arrays if a is not None]
    lo, hi = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
              if valid else (0.0, 1.0))
    for j, (a, ps) in enumerate(zip(arrays, spacings)):
        if offset + j >= MAXS:
            break
        if a is None:
            continue
        x, (y0, x0, cpx, h, w) = mm_crop_resize(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1), ps)
        vol[offset + j] = (x * 255).astype(np.uint8)
        p = min(int(picks[j]), len(files) - 1)
        g = files[p][2].copy()
        g[11:16] = (y0, x0, cpx, h, w); g[17] = p; g[18] = len(files)
        geom[offset + j] = g


def build_study(args):
    sid, rows, sdir = args
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    geom = np.zeros((MAXS, GEOM), np.float32)
    used, offset, infos = set(), 0, []
    try:
        for plane, fluid, count in SLOTS:
            rec = pick_series(rows, plane, fluid, used)
            if rec is None:
                offset += count; continue
            used.add(rec["SeriesInstanceUID"])
            files, med_ps, info = order_and_meta(f"{sdir}/{sid}/{rec['SeriesInstanceUID']}")
            info.update(slot=offset, plane=plane, fluid=fluid, n=len(files),
                        series=rec["SeriesInstanceUID"])
            infos.append(info)
            if not files:
                offset += count; continue
            n = len(files)
            lo = int(n * SPAN_LO); hi = max(int(n * SPAN_HI) - 1, lo)
            picks = (np.linspace(lo, hi, count).round().astype(int) if n > 1
                     else np.zeros(count, dtype=int))
            cache = {}
            for p in sorted(set(picks.tolist())):
                p = min(int(p), n - 1)
                try:
                    cache[p] = read_px(files[p][0])
                except Exception:
                    cache[p] = None
            fill(vol, geom, offset, picks, cache, files, med_ps)
            offset += count
            if offset >= MAXS:
                break
        err = ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, geom, infos, err


