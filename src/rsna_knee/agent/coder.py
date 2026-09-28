"""DeepSeek coder tools: read/write allowlisted repo paths and run a smoke test suite.

User-approved 2026-09-27: planner may edit harness + trainer. Hard rails remain — no secrets,
no paths outside the allowlist, no leaderboard submit, campaign budget / one-pod / >$10 gates
still enforced by the host when a train is launched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rsna_knee.agent.activity import log
from rsna_knee.agent.genome import REPO

# Writable / readable tree for autonomous code changes.
ALLOW_PREFIXES = (
    "src/rsna_knee/agent/",
    "runpod/",
    "tests/",
    "agent/memory/",
    "agent/labels/",
)
# Never touch these even if nested under an allow prefix.
BLOCK_NAMES = {
    "secrets.env", "kaggle.json", "access_token", ".env", "credentials",
}
MAX_WRITE_BYTES = 400_000
MAX_READ_CHARS = 80_000


def _resolve(rel: str) -> Path:
    rel = (rel or "").strip().lstrip("/")
    if not rel or ".." in Path(rel).parts or rel.startswith(".."):
        raise ValueError(f"invalid path {rel!r}")
    name = Path(rel).name.lower()
    if name in BLOCK_NAMES or any(b in rel.lower() for b in ("secret", "credential", "kaggle.json")):
        raise ValueError(f"refusing secret-like path {rel!r}")
    if not any(rel.startswith(p) for p in ALLOW_PREFIXES):
        raise ValueError(f"path not in allowlist: {rel}")
    path = (REPO / rel).resolve()
    if not str(path).startswith(str(REPO.resolve())):
        raise ValueError(f"path escapes repo: {rel}")
    return path


def list_repo(path: str = "src/rsna_knee/agent", limit: int = 80) -> dict:
    """List files under an allowlisted directory."""
    try:
        root = _resolve(path)
        if not root.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}
        entries = []
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(REPO))
                entries.append(rel)
                if len(entries) >= limit:
                    break
        return {"ok": True, "path": path, "files": entries, "truncated": len(entries) >= limit}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def read_repo_file(path: str, offset: int = 0, limit: int = 400) -> dict:
    """Read a text file (line window) from the allowlist."""
    try:
        p = _resolve(path)
        if not p.is_file():
            return {"ok": False, "error": f"missing file: {path}"}
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 800))
        chunk = lines[offset: offset + limit]
        text = "\n".join(chunk)
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n…[truncated]"
        return {"ok": True, "path": path, "offset": offset, "lines": len(chunk),
                "total_lines": len(lines), "content": text}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def write_repo_file(path: str, content: str, *, reason: str = "") -> dict:
    """Create or overwrite an allowlisted file. Logs to activity.jsonl."""
    try:
        if content is None:
            return {"ok": False, "error": "content required"}
        raw = content if isinstance(content, str) else str(content)
        if len(raw.encode("utf-8")) > MAX_WRITE_BYTES:
            return {"ok": False, "error": f"content exceeds {MAX_WRITE_BYTES} bytes"}
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        prev = p.read_text(encoding="utf-8") if p.exists() else None
        p.write_text(raw, encoding="utf-8")
        log("code", f"deepseek wrote {path}" + (f" ({reason})" if reason else ""),
            path=path, bytes=len(raw.encode("utf-8")),
            created=prev is None, reason=(reason or "")[:200])
        return {"ok": True, "path": path, "bytes": len(raw.encode("utf-8")),
                "created": prev is None,
                "hint": "Call run_agent_tests before launch_runpod_train if you changed trainer/harness."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def run_agent_tests(extra: str = "") -> dict:
    """Run the agent unit tests; return ok + truncated output."""
    import os
    targets = ["tests/test_loop.py"]
    for part in (extra or "").split():
        if not part.startswith("tests/") or ".." in part:
            return {"ok": False, "error": f"extra path not allowed: {part}"}
        targets.append(part)
    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path("python3")
    cmd = [str(py), "-m", "pytest", *targets, "-q", "--tb=line"]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    try:
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=180, env=env)
        out = ((r.stdout or "") + (r.stderr or ""))[-4000:]
        ok = r.returncode == 0
        log("code", f"deepseek run_agent_tests {'PASS' if ok else 'FAIL'}", returncode=r.returncode)
        return {"ok": ok, "returncode": r.returncode, "output": out}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


CODER_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_repo",
            "description": "List allowlisted source files (agent harness, runpod trainer, tests).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to repo root"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_repo_file",
            "description": "Read a slice of an allowlisted file (harness / trainer / tests).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "0-based start line"},
                    "limit": {"type": "integer", "description": "max lines"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_repo_file",
            "description": "Create/overwrite an allowlisted file (agent harness, runpod/*, tests). "
                           "After edits call run_agent_tests, then launch_runpod_train if a GPU run is needed. "
                           "Trainer edits ship on the next RunPod bundle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Full new file contents"},
                    "reason": {"type": "string", "description": "Why this change helps AUC / reliability"},
                },
                "required": ["path", "content", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agent_tests",
            "description": "Run agent unit tests after code edits. Prefer PASS before launching a pod.",
            "parameters": {
                "type": "object",
                "properties": {
                    "extra": {"type": "string", "description": "Optional extra tests/… paths"},
                },
                "additionalProperties": False,
            },
        },
    },
]


def handlers() -> dict:
    return {
        "list_repo": list_repo,
        "read_repo_file": read_repo_file,
        "write_repo_file": write_repo_file,
        "run_agent_tests": run_agent_tests,
    }
