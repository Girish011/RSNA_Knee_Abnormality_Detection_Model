"""Minimal RunPod REST client for the agent loop (create / get / stop / terminate).

Keys come from ``~/.rsna_agent/secrets.env`` and are never logged or returned.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
SECRETS = Path(os.path.expanduser("~/.rsna_agent/secrets.env"))
GPU_4090 = "NVIDIA GeForce RTX 4090"
GPU_3090 = "NVIDIA GeForce RTX 3090"


def secret(name: str) -> str:
    for line in SECRETS.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise KeyError(f"{name} missing from {SECRETS}")


def _h() -> dict:
    return {"Authorization": f"Bearer {secret('RUNPOD_API_KEY')}"}


def balance() -> dict:
    q = "query { myself { clientBalance currentSpendPerHr } }"
    r = requests.post(GRAPHQL, json={"query": q}, headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json()["data"]["myself"]


def create_pod(spec: dict) -> dict:
    r = requests.post(f"{REST}/pods", json=spec, headers=_h(), timeout=60)
    if not r.ok:
        raise RuntimeError(f"create_pod {r.status_code}: {r.text[:500]}")
    return r.json()


def get_pod(pod_id: str) -> dict:
    r = requests.get(f"{REST}/pods/{pod_id}", headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json()


def list_pods() -> list[dict]:
    r = requests.get(f"{REST}/pods", headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json()


def stop_pod(pod_id: str) -> None:
    requests.post(f"{REST}/pods/{pod_id}/stop", headers=_h(), timeout=30).raise_for_status()


def terminate_pod(pod_id: str) -> None:
    requests.delete(f"{REST}/pods/{pod_id}", headers=_h(), timeout=30).raise_for_status()


def proxy_url(pod_id: str, port: int = 8000) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"
