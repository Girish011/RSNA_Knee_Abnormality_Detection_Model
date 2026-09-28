#!/usr/bin/env python3
"""Download dense80 kernel-output shards with HTTP Range resume.

``kaggle kernels output`` dies with IncompleteRead near the end of ~10 GB vols.
Refreshing signed URLs + Range resumes finishes the last ~1 GB instead of restarting.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

USER = "girishbose"
OUT = Path(os.environ.get("DENSE80_DIR", "/workspace/input/dense80"))
TOKEN = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
MIN_VOL_BYTES = int(os.environ.get("DENSE80_MIN_VOL", str(5_000_000_000)))
TRIES = int(os.environ.get("DENSE80_TRIES", "10"))
CHUNK = 8 << 20


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
        params={"userName": USER, "kernelSlug": f"rsna-knee-dense80-s{shard}"},
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
    for t in range(1, tries + 1):
        pos = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=600) as r:
                if r.status_code == 404:
                    raise RuntimeError("url expired/404")
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                # If server ignores Range and returns 200, restart file.
                mode = "ab" if (pos and r.status_code == 206) else "wb"
                if mode == "wb":
                    pos = 0
                wrote = 0
                with open(dest, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                            wrote += len(chunk)
                print(f"  {dest.name}: +{wrote} bytes (now {dest.stat().st_size})", flush=True)
                return
        except Exception as e:
            print(f"  try {t}/{tries} {dest.name} @{pos}: {e}", flush=True)
            time.sleep(min(60, 5 * t))
    raise RuntimeError(f"failed to download {dest.name}")


def fetch_shard(shard: int) -> None:
    print(f"=== dense80 shard {shard} ===", flush=True)
    needed = {f"vols_s{shard}.npy", f"masks_s{shard}.npy", f"ids_s{shard}.npy", f"meta_s{shard}.json"}
    for t in range(1, TRIES + 1):
        try:
            files = {f["fileName"]: f["url"] for f in list_files(shard)}
        except Exception as e:
            print(f"list_files try {t}: {e}", flush=True)
            time.sleep(5 * t)
            continue
        missing = needed - set(files)
        if missing:
            raise RuntimeError(f"shard {shard} missing {missing}")
        for name in sorted(needed):
            dest = OUT / name
            if name.startswith("vols_") and dest.exists() and dest.stat().st_size >= MIN_VOL_BYTES:
                print(f"  skip {name} already {dest.stat().st_size}", flush=True)
                continue
            if (not name.startswith("vols_")) and dest.exists() and dest.stat().st_size > 100:
                print(f"  skip {name} already {dest.stat().st_size}", flush=True)
                continue
            print(f"  fetching {name} (attempt refresh {t})", flush=True)
            try:
                download_resumable(files[name], dest)
            except Exception as e:
                print(f"  refresh URLs after error: {e}", flush=True)
                break  # refresh signed URLs
        else:
            vol = OUT / f"vols_s{shard}.npy"
            sz = vol.stat().st_size if vol.exists() else 0
            if sz >= MIN_VOL_BYTES:
                print(f"shard {shard} OK vols={sz}", flush=True)
                return
            print(f"shard {shard} vols too small ({sz}); retry", flush=True)
            vol.unlink(missing_ok=True)
        time.sleep(10)
    raise SystemExit(f"dense80 shard {shard} failed after retries")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shards = [int(x) for x in (sys.argv[1:] or ["0", "1", "2", "3"])]
    for s in shards:
        fetch_shard(s)
    print("ALL SHARDS OK", flush=True)


if __name__ == "__main__":
    main()
