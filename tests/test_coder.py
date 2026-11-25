"""Allowlist coder tools for DeepSeek."""

from rsna_knee.agent import coder


def test_coder_rejects_secrets_and_escapes(tmp_path, monkeypatch):
    monkeypatch.setattr(coder, "REPO", tmp_path)
    (tmp_path / "src/rsna_knee/agent").mkdir(parents=True)
    (tmp_path / "runpod").mkdir()
    assert coder.write_repo_file("secrets.env", "x", reason="no")["ok"] is False
    assert coder.write_repo_file("../outside.py", "x", reason="no")["ok"] is False
    assert coder.write_repo_file("docs/STATUS.md", "x", reason="no")["ok"] is False


def test_coder_writes_and_reads_allowlisted(tmp_path, monkeypatch):
    monkeypatch.setattr(coder, "REPO", tmp_path)
    (tmp_path / "runpod").mkdir(parents=True)
    # activity log needs RUNS — stub log
    monkeypatch.setattr(coder, "log", lambda *a, **k: None)
    w = coder.write_repo_file("runpod/hello.py", "print(1)\n", reason="smoke")
    assert w["ok"] and w["created"]
    r = coder.read_repo_file("runpod/hello.py")
    assert r["ok"] and "print(1)" in r["content"]
