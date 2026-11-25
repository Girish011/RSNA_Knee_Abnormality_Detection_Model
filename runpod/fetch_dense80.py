#!/usr/bin/env python3
"""Download dense80 / dense512 kernel-output shards with HTTP Range resume.

DENSE_NAME=dense512 DENSE_SHARDS=8 fetches the native-resolution cache (adds geom/series files;
shards are fetched DENSE_PAR at a time).

``kaggle kernels output`` dies with IncompleteRead near the end of ~10 GB vols.
Refreshing signed URLs + Range resumes finishes the last ~1 GB instead of restarting.
"""

from __future__ import annotations

import os
import json
import math
import re
import sys
import time
from pathlib import Path

import requests

USER = "girishbose"
NAME = os.environ.get("DENSE_NAME", "dense80")
N_SHARDS = int(os.environ.get("DENSE_SHARDS", "4"))
PAR = int(os.environ.get("DENSE_PAR", "4" if NAME == "dense80" else "3"))   # all four dense80 shards at once
EXTRA = [] if NAME == "dense80" else ["geom_s{s}.npy", "series_s{s}.json"]
OUT = Path(os.environ.get("DENSE80_DIR", f"/workspace/input/{NAME}"))
TOKEN = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
TRIES = int(os.environ.get("DENSE80_TRIES", "10"))
CHUNK = 8 << 20


def validate_file(path: Path, expected_bytes: int | None = None) -> dict:
    """Validate exact NPY payload length without reading a 12GB volume; never unpickle IDs."""
    import numpy as np
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"{path.name}: length {size} != {expected_bytes}")
    name = path.name.removesuffix(".part")
    if name.endswith(".npy"):
        with path.open("rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            elif version in ((2, 0), (3, 0)):
                shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f"unsupported NPY version {version}")
            if dtype.hasobject:
                raise ValueError("object-dtype cache is not allowed")
            want = f.tell() + math.prod(shape) * dtype.itemsize
        if size != want:
            raise ValueError(f"{name}: incomplete NPY payload {size} != {want}")
        return {"bytes": size, "shape": list(shape), "dtype": str(dtype)}
    if name.endswith(".json"):
        with path.open() as f:
            value = json.load(f)
        if not isinstance(value, (dict, list)):
            raise ValueError(f"{name}: invalid JSON cache")
        return {"bytes": size}
    raise ValueError(f"unsupported cache file {name}")


def validate_shard(shard: int) -> dict:
    import numpy as np
    records = {p.name: validate_file(p) for p in OUT.glob(f"*_s{shard}.*")
               if p.suffix in (".npy", ".json") and not p.name.startswith("verified_")}
    vol, mask, ids = [records[f"{k}_s{shard}.npy"] for k in ("vols", "masks", "ids")]
    n = ids["shape"][0]
    if len(ids["shape"]) != 1 or vol["shape"][:2] != [n, 80] or mask["shape"] != [n, 80]:
        raise ValueError("cache IDs, masks and volume shapes disagree")
    expected_res = 512 if NAME == "dense512" else 336
    if vol["shape"][2:] != [expected_res, expected_res] or vol["dtype"] != "uint8":
        raise ValueError("unexpected cache resolution/dtype")
    with (OUT / f"meta_s{shard}.json").open() as f:
        meta = json.load(f)
    if meta.get("n") != n or meta.get("done") != n or meta.get("img", expected_res) != expected_res:
        raise ValueError("cache builder did not complete all studies")
    id_values = np.load(OUT / f"ids_s{shard}.npy", allow_pickle=False)
    if id_values.dtype.kind not in ("U", "S") or len(np.unique(id_values)) != n or mask["dtype"] != "uint8":
        raise ValueError("invalid/duplicate cache IDs or mask dtype")
    if NAME == "dense512" and records[f"geom_s{shard}.npy"]["shape"][:2] != [n, 80]:
        raise ValueError("geometry and volume shapes disagree")
    return records


def _token() -> str:
    if TOKEN:
        return TOKEN.strip().splitlines()[0]
    p = Path.home() / ".kaggle" / "access_token"
    if p.exists():
        return p.read_text().strip().splitlines()[0]
    raise SystemExit("KAGGLE_API_TOKEN missing")


def list_files(shard: int) -> list[dict]:
    r = requests.get(
        "https://www.kaggle.com/api/v1/kernels/output",
        params={"userName": USER, "kernelSlug": f"rsna-knee-{NAME}-s{shard}"},
        headers={"Authorization": f"Bearer {_token()}"},
        timeout=60,
    )
    r.raise_for_status()
    files = r.json().get("files") or []
    if not files:
        raise RuntimeError(f"no output files for shard {shard}")
    return files


def download_resumable(url: str, dest: Path, tries: int = TRIES) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.exists():
        try:
            validate_file(dest)
            return
        except (ValueError, OSError, EOFError):
            # Retain an interrupted legacy download for safe Range resumption.
            if not part.exists():
                dest.rename(part)
            else:
                dest.unlink()
    for t in range(1, tries + 1):
        pos = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as r:
                if r.status_code == 416:
                    # Can follow a complete transfer whose connection died at EOF.
                    try:
                        validate_file(part)
                        os.replace(part, dest)
                        return
                    except (ValueError, OSError, EOFError):
                        part.unlink(missing_ok=True)
                        raise ValueError("invalid partial file exceeds remote length") from None
                if r.status_code == 404:
                    raise RuntimeError("url expired/404")
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                # If server ignores Range and returns 200, restart file.
                total = None
                if r.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != pos:
                        raise ValueError("invalid Range response")
                    total = int(match[3])
                elif r.headers.get("Content-Length"):
                    total = int(r.headers["Content-Length"])
                mode = "ab" if (pos and r.status_code == 206) else "wb"
                if mode == "wb":
                    pos = 0
                wrote = 0
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                            wrote += len(chunk)
                validate_file(part, total)
                os.replace(part, dest)
                print(f"  {dest.name}: verified {dest.stat().st_size} bytes", flush=True)
                return
        except Exception as e:
            # Exception strings can contain signed private download URLs.
            print(f"  try {t}/{tries} {dest.name} @{pos}: {type(e).__name__}", flush=True)
            time.sleep(min(60, 5 * t))
    raise RuntimeError(f"failed to download {dest.name}")


def fetch_shard(shard: int) -> None:
    print(f"=== {NAME} shard {shard} ===", flush=True)
    needed = {f"vols_s{shard}.npy", f"masks_s{shard}.npy", f"ids_s{shard}.npy", f"meta_s{shard}.json",
              *(e.format(s=shard) for e in EXTRA)}
    for t in range(1, TRIES + 1):
        try:
            files = {f["fileName"]: f["url"] for f in list_files(shard)}
        except Exception as e:
            print(f"list_files try {t}: {type(e).__name__}", flush=True)
            time.sleep(5 * t)
            continue
        missing = needed - set(files)
        if missing:
            raise RuntimeError(f"shard {shard} missing {missing}")
        for name in sorted(needed):
            dest = OUT / name
            if dest.exists():
                try:
                    validate_file(dest)
                    print(f"  verified existing {name}", flush=True)
                    continue
                except (ValueError, OSError, EOFError):
                    pass
            print(f"  fetching {name} (attempt refresh {t})", flush=True)
            try:
                download_resumable(files[name], dest)
            except Exception as e:
                print(f"  refresh URLs after error: {type(e).__name__}", flush=True)
                break  # refresh signed URLs
        else:
            receipt = validate_shard(shard)
            (OUT / f"verified_s{shard}.json").write_text(json.dumps(receipt, indent=2))
            print(f"shard {shard} VERIFIED", flush=True)
            return
        time.sleep(10)
    raise SystemExit(f"{NAME} shard {shard} failed after retries")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shards = [int(x) for x in sys.argv[1:]] or list(range(N_SHARDS))
    if PAR > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(PAR) as ex:
            list(ex.map(fetch_shard, shards))
    else:
        for s in shards:
            fetch_shard(s)
    print("ALL SHARDS OK", flush=True)


if __name__ == "__main__":
    main()
