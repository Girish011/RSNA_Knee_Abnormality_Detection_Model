#!/usr/bin/env python3
"""Offline, bounded-memory submission runner for fixed-EMA definition readers.

Use the original dense512 pixel pipeline and the checkpoint's saved view protocol.
No labels/reports are read. Reject missing/corrupt studies instead of silently
substituting another model. A receipt identifies every deployed checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import multiprocessing
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import torch

import definition_cache as cache
import definition_reader as reader


def validate_submission(frame, ids):
    if list(frame.columns) != ["StudyInstanceUID", *reader.LAB]:
        raise ValueError("submission column order differs from competition")
    if frame.StudyInstanceUID.astype(str).tolist() != list(ids) or len(set(ids)) != len(ids):
        raise ValueError("submission study order/uniqueness mismatch")
    values = frame[reader.LAB].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("invalid submission probabilities")


def write_chunk(root, ids, built):
    if len(ids) != len(built):
        raise ValueError("cache result count differs")
    for uid, (vol, mask, geom, info, error) in zip(ids, built):
        if error:
            raise RuntimeError(f"DICOM preprocessing failed for {uid}: {error}")
        if not any(len(reader.valid_centres(mask.astype(bool), lo, hi)) for lo, hi, _ in reader.BLOCKS):
            raise RuntimeError(f"no usable MRI triplets for {uid}")
    for name, values in (("vols", [b[0] for b in built]), ("masks", [b[1] for b in built]),
                         ("geom", [b[2] for b in built]), ("ids", ids)):
        np.save(root / f"{name}_s0.npy", np.asarray(values))
    (root / "series_s0.json").write_text(json.dumps({uid: b[3] for uid, b in zip(ids, built)}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--competition-root", required=True)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--receipt", default="definition_receipt.json")
    ap.add_argument("--batch-studies", type=int, default=8)
    ap.add_argument("--decode-workers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=8.5*3600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if min(a.batch_studies, a.decode_workers, a.bs, a.max_seconds) <= 0:
        raise ValueError("batch sizes, workers and time budget must be positive")
    started = time.monotonic()
    cv2.setNumThreads(1)
    root = Path(a.competition_root)
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    ids = sample.StudyInstanceUID.tolist()
    # Match sample schema before decoding any costly images.
    validate_submission(sample, ids)
    series_path = root / "test_series.csv"
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    rows = {u: g.to_dict("records") for u, g in series.groupby("StudyInstanceUID", sort=False)}
    if set(ids) - set(rows):
        raise ValueError("test series metadata missing required study IDs")
    image_root = next((root / n for n in ("test_series", "test_images") if (root / n).is_dir()), None)
    if image_root is None:
        raise FileNotFoundError("test DICOM directory missing")
    models, receipts = [], []
    for path in a.checkpoint:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["protocol"]
        if cfg["version"] != reader.PROTOCOL_VERSION or ck.get("checkpoint_policy") != "fixed_ema":
            raise ValueError("unsupported checkpoint policy/protocol")
        if cfg.get("dummy") or ck.get("step", 0) < 1:
            raise ValueError("test/dummy or untrained checkpoint cannot submit")
        model = reader.Reader(cfg["arch"], hidden=cfg["hidden"]).to(a.device)
        model.load_state_dict(ck["model"], strict=True)
        models.append((model, cfg))
        receipts.append({"filename": Path(path).name, "sha256": reader.sha256(path),
                         "epoch": ck["epoch"], "step": ck["step"], "protocol": cfg})
    pred = np.zeros((len(ids), 12), np.float32)
    with tempfile.TemporaryDirectory(prefix="definition_test_") as td, ProcessPoolExecutor(
        max_workers=a.decode_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        chunk_root = Path(td)
        for start in range(0, len(ids), a.batch_studies):
            if time.monotonic() - started > a.max_seconds:
                raise TimeoutError("submission budget exceeded before all studies were scored")
            chunk_ids = ids[start:start + a.batch_studies]
            jobs = [(uid, rows[uid], str(image_root)) for uid in chunk_ids]
            built = list(pool.map(cache.build_study, jobs))
            write_chunk(chunk_root, chunk_ids, built)
            del built
            corpus = reader.Corpus(chunk_root, n_shards=1, series_csv=series_path)
            for model, cfg in models:
                remaining = a.max_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("submission budget exhausted")
                pred[start:start + len(chunk_ids)] += reader.predict(
                    model, corpus, chunk_ids, cfg, bs=a.bs, device=a.device, max_seconds=remaining
                ) / len(models)
            del corpus
            gc.collect()
            print(json.dumps({"completed": start + len(chunk_ids), "total": len(ids),
                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
    frame = pd.DataFrame(pred, columns=reader.LAB)
    frame.insert(0, "StudyInstanceUID", ids)
    validate_submission(frame, ids)
    dest = Path(a.out)
    temp = dest.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(dest)
    reader.atomic_json({"status": "complete", "studies": len(ids), "checkpoints": receipts,
                        "cache_source_sha256": cache.SOURCE_SHA256, "reader_sha256": reader.sha256(reader.__file__),
                        "series_sha256": reader.sha256(series_path), "submission_sha256": reader.sha256(dest),
                        "seconds": round(time.monotonic() - started, 2)}, a.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
