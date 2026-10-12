"""Free launch-payload and failure-receipt checks; all provider calls are mocked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("definition_launch_test", REPO / "scripts/definition_campaign.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class Response:
    def __init__(self, payload=None, status=200, raw=None):
        self.payload = payload
        self.status_code = status
        self.ok = 200 <= status < 400
        self.content = raw if raw is not None else json.dumps(payload).encode()

    def json(self):
        if self.payload is None:
            raise ValueError("non-json body with private data")
        return self.payload


def launch_spec():
    return {"name": "own-test", "imageName": "image", "cloudType": "COMMUNITY",
            "gpuTypeIds": ["GPU"], "env": {"SECRET": "private-env-value"},
            "dockerStartCmd": ["python", "-c", "print('test')"]}


def test_compact_startup_runs_exact_utf8_source_and_checks_hash(tmp_path):
    source = "# Quotes and UTF-8 must survive transport\nvalue = 'knee: café \\\"MRI\\\"'\nprint(value)\n"
    cmd, receipt = C.compact_startup(source)
    assert receipt["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert shlex.split(shlex.join(cmd)) == cmd
    completed = subprocess.run([sys.executable, *cmd[1:]], capture_output=True, text=True)
    assert completed.returncode == 0 and completed.stdout == 'knee: café "MRI"\n'
    altered = cmd[-1].replace(receipt["source_sha256"], "0" * 64)
    rejected = subprocess.run([sys.executable, "-c", altered], capture_output=True, text=True)
    assert rejected.returncode != 0 and rejected.stdout == ""
    assert "supervisor source checksum mismatch" in rejected.stderr
    actual = (REPO / "runpod/definition_job.py").read_text()
    compressed, _ = C.compact_startup(actual)
    assert len(shlex.join(compressed)) < len(shlex.join(["python", "-u", "-c", actual])) * .6


def test_validation_failure_receipt_has_no_secret_and_private_raw_file(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REPO", tmp_path)
    monkeypatch.setattr(C.runpod, "_h", lambda: {})
    payload = {"errors": [{"message": 'Variable "$input" got invalid value private-env-value https://private/?signature=unknown-token',
                           "extensions": {"code": "GRAPHQL_VALIDATION_FAILED", "private": "hidden-extension"}}]}
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return Response(payload, 400)
    monkeypatch.setattr(C.requests, "post", post)
    with pytest.raises(C.PodCreationError) as caught:
        C.create_deadline_pod(launch_spec(), time.time() + 300)
    receipt = caught.value.receipt
    assert receipt["outcome"] == "definite_preexecution_rejection" and receipt["http_status"] == 400
    assert receipt["graphql_errors"] == [{"code": "GRAPHQL_VALIDATION_FAILED", "messages": ["graphql_schema_or_input_validation"]}]
    assert len(calls) == 1
    exposed = str(caught.value) + json.dumps(receipt)
    assert all(token not in exposed for token in ("private-env-value", "unknown-token", "hidden-extension", "https://"))
    raw = Path(receipt["private_response_file"])
    assert "private-env-value" in raw.read_text()
    assert stat.S_IMODE(raw.stat().st_mode) == 0o600
    assert stat.S_IMODE(raw.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("payload,status", [
    ({"errors": [{"message": "There are no longer any instances available with the requested specifications", "extensions": {"code": "BAD_USER_INPUT"}}]}, 200),
    ({"errors": [{"message": "private unknown error", "extensions": {"code": "secret-as-code"}}]}, 500),
    ({"errors": [{"message": "validation in resolver", "extensions": {"code": "GRAPHQL_VALIDATION_FAILED"}, "path": ["podFindAndDeployOnDemand"]}]}, 200),
    ({"data": {"someMutation": {}}, "errors": [{"extensions": {"code": "GRAPHQL_VALIDATION_FAILED"}}]}, 200),
    ({"data": {"podFindAndDeployOnDemand": None}}, 200),
    (None, 502),
])
def test_nondefinite_failure_never_claims_no_allocation(tmp_path, monkeypatch, payload, status):
    monkeypatch.setattr(C, "REPO", tmp_path)
    monkeypatch.setattr(C.runpod, "_h", lambda: {})
    calls = []
    def post(*a, **k):
        calls.append(k)
        return Response(payload, status, b"private-non-json-response" if payload is None else None)
    monkeypatch.setattr(C.requests, "post", post)
    with pytest.raises(C.PodCreationError) as caught:
        C.create_deadline_pod(launch_spec(), time.time() + 300)
    assert caught.value.receipt["outcome"] == "unknown" and len(calls) == 1
    assert "secret-as-code" not in str(caught.value)
    assert "private-non-json-response" not in str(caught.value)


def test_transport_failure_is_safe_unknown_and_not_retried(monkeypatch):
    monkeypatch.setattr(C.runpod, "_h", lambda: {})
    calls = []
    def post(*a, **k):
        calls.append(k)
        raise C.requests.Timeout("private-env-value signed URL")
    monkeypatch.setattr(C.requests, "post", post)
    with pytest.raises(C.PodCreationError) as caught:
        C.create_deadline_pod(launch_spec(), time.time() + 300)
    assert caught.value.receipt == {"outcome": "unknown", "http_status": None, "transport_error": "provider_request_failed"}
    assert "private-env-value" not in str(caught.value) and len(calls) == 1


def test_reviewed_absent_pod_is_terminated_if_it_reappears_foreign_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STATE", tmp_path / "ledger.json")
    job = {"job": "own", "pod_name": "own", "status": "reconciled_absent", "reserved_usd": 3.0}
    C.atomic_json(C.STATE, {"limit_usd": 30., "jobs": [job]})
    monkeypatch.setattr(C.runpod, "list_pods", lambda: [{"id": "delayed-own", "name": "own"}, {"id": "foreign", "name": "another-project"}])
    terminated = []
    monkeypatch.setattr(C.runpod, "terminate_pod", lambda pod: terminated.append(pod))
    C.status()
    after = json.loads(C.STATE.read_text())["jobs"][0]
    assert terminated == ["delayed-own"] and after["status"] == "terminate_pending"
    assert after["reserved_usd"] == 3.0


def test_reviewed_absence_stays_reviewed_when_only_foreign_pod_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STATE", tmp_path / "ledger.json")
    job = {"job": "own", "pod_name": "own", "status": "reconciled_absent", "reserved_usd": 3.0}
    C.atomic_json(C.STATE, {"limit_usd": 30., "jobs": [job]})
    monkeypatch.setattr(C.runpod, "list_pods", lambda: [{"id": "foreign", "name": "another-project"}])
    terminated = []
    monkeypatch.setattr(C.runpod, "terminate_pod", lambda pod: terminated.append(pod))
    C.status()
    assert not terminated and json.loads(C.STATE.read_text())["jobs"][0] == job
