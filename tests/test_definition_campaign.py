"""Operational regression tests; no provider calls and no paid resource creation."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def module(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


C = module("scripts/definition_campaign.py", "definition_campaign_ops")
F = module("runpod/fetch_dense80.py", "safe_cache_fetch")


def test_reservation_enforces_campaign_and_one_pod():
    st = {"jobs": []}
    first = C.reserve(st, 3)
    assert first["max_wall_seconds"] * C.RATE_CEILING / 3600 + C.RESERVE_MARGIN <= 3
    with pytest.raises(RuntimeError, match="unresolved"):
        C.reserve(st, 3)
    first["status"] = "unknown"
    with pytest.raises(RuntimeError, match="unresolved"):
        C.reserve(st, 3)
    first["status"] = "terminated"
    for _ in range(9):
        C.reserve(st, 3)["status"] = "terminated"
    assert C.committed(st) == 30
    with pytest.raises(RuntimeError, match="ceiling"):
        C.reserve(st, .5)


@pytest.mark.parametrize("amount", [-1, 0, 3.01, float("nan"), float("inf")])
def test_cap_values_fail_closed(amount):
    with pytest.raises(ValueError):
        C.reserve({"jobs": []}, amount)


def test_ledger_keeps_unknown_reservation_after_failure(tmp_path):
    p = tmp_path / "ledger.json"
    with pytest.raises(RuntimeError):
        with C.ledger(p) as s:
            C.reserve(s, 3)["status"] = "unknown"
            raise RuntimeError("provider response lost")
    saved = json.loads(p.read_text())
    assert saved["jobs"][0]["status"] == "unknown"
    with C.ledger(p) as s:
        with pytest.raises(RuntimeError): C.reserve(s, 3)


def test_ledger_rejects_changed_ceiling(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"limit_usd": 31, "jobs": []}))
    with pytest.raises(RuntimeError, match="limit changed"):
        with C.ledger(p): pass


def test_npy_exact_shape_bytes_and_json(tmp_path):
    p = tmp_path / "vols.npy"
    np.save(p, np.zeros((2, 3, 4), np.uint8))
    assert F.validate_file(p)["shape"] == [2, 3, 4]
    with p.open("r+b") as f: f.truncate(p.stat().st_size - 1)
    with pytest.raises(ValueError, match="incomplete"):
        F.validate_file(p)
    j = tmp_path / "meta.json"; j.write_text('{"valid":')
    with pytest.raises(ValueError): F.validate_file(j)
    np.save(p, np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError, match="object"):
        F.validate_file(p)


class Response:
    ok = True
    def __init__(self, code=200, headers=None, data=b"", payload=None):
        self.status_code, self.headers, self.data, self.payload = code, headers or {}, data, payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def iter_content(self, size): yield self.data
    def json(self): return self.payload


def test_truncated_legacy_file_resumes_and_atomically_promotes(tmp_path, monkeypatch):
    full = tmp_path / "full.npy"; np.save(full, np.zeros((20, 4), np.uint8)); blob = full.read_bytes()
    dest = tmp_path / "vols.npy"; split = len(blob)-17; dest.write_bytes(blob[:split])
    def get(url, **kw):
        assert kw["headers"] == {"Range": f"bytes={split}-"}
        return Response(206, {"Content-Range": f"bytes {split}-{len(blob)-1}/{len(blob)}"}, blob[split:])
    monkeypatch.setattr(F.requests, "get", get)
    F.download_resumable("unused", dest, tries=1)
    assert dest.read_bytes() == blob and not dest.with_name("vols.npy.part").exists()


def test_wrong_range_never_promotes(tmp_path, monkeypatch):
    dest = tmp_path / "vols.npy"; part = dest.with_name("vols.npy.part"); part.write_bytes(b"123")
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: Response(206, {"Content-Range": "bytes 0-2/3"}, b"123"))
    monkeypatch.setattr(F.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError): F.download_resumable("unused", dest, tries=1)
    assert not dest.exists() and part.read_bytes() == b"123"


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    full = tmp_path / "full.npy"; np.save(full, np.zeros((20, 4), np.uint8)); blob = full.read_bytes()
    dest = tmp_path / "vols.npy"; dest.with_name("vols.npy.part").write_bytes(b"junk")
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: Response(200, {"Content-Length": str(len(blob))}, blob))
    F.download_resumable("unused", dest, tries=1)
    assert dest.read_bytes() == blob


def test_creation_has_provider_deadline_and_is_single_mutation(monkeypatch):
    sent = []
    def post(url, **kw):
        sent.append(kw["json"])
        inp = kw["json"]["variables"]["input"]
        return Response(payload={"data": {"podFindAndDeployOnDemand": {"id": "p1", "costPerHr": .44,
                                   "terminateAfter": inp["terminateAfter"]}}})
    monkeypatch.setattr(C.requests, "post", post)
    monkeypatch.setattr(C.runpod, "_h", lambda: {})
    out = C.create_deadline_pod({"name": "test", "imageName": "image", "cloudType": "SECURE",
                                "gpuTypeIds": ["A40"], "env": {}, "dockerStartCmd": ["python", "-c", "print('ok')"]}, time.time()+3600)
    assert out["id"] == "p1" and len(sent) == 1
    assert sent[0]["variables"]["input"]["terminateAfter"].endswith("Z")
    assert "desiredStatus terminateAfter" not in sent[0]["query"]  # live Pod has no deadline output field


def test_quote_reservation_uses_rate_plus_storage_and_four_hour_ttl():
    j = C.reserve({"jobs": []}, 3, quoted_gpu_hr=.34)
    assert j["max_wall_seconds"] == 4 * 3600
    assert j["max_wall_seconds"] * j["all_in_hr_ceiling"] / 3600 + C.RESERVE_MARGIN <= 3


def test_unknown_quote_and_high_rate_refuse(monkeypatch):
    monkeypatch.setattr(C.runpod, "_h", lambda: {})
    for value in [None, .76, float("nan")]:
        monkeypatch.setattr(C.requests, "post", lambda *a, **k: Response(payload={"data": {"gpuTypes": [{"lowestPrice": {"uninterruptablePrice": value}}]}}))
        with pytest.raises(RuntimeError): C.quote("A40", "SECURE")


def test_status_network_failure_never_releases_reservation(tmp_path, monkeypatch):
    p = tmp_path / "ledger.json"; monkeypatch.setattr(C, "STATE", p)
    with C.ledger() as s:
        C.reserve(s, 3)["status"] = "unknown"
    before = p.read_bytes()
    def fail(): raise OSError("network unavailable")
    monkeypatch.setattr(C.runpod, "list_pods", fail)
    with pytest.raises(OSError): C.status()
    assert p.read_bytes() == before
