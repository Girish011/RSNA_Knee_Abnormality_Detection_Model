#!/usr/bin/env python3
"""Bounded definition-reader pod supervisor; stdlib-only bootstrap.

Embedded in dockerStartCmd so the termination watchdog runs even when bundle/pip
downloads fail. Provider-side expiry plus local campaign status monitor are the
independent backstops for a host which never starts this Python process.
"""
from __future__ import annotations
import functools
import hashlib
import hmac
import csv
import http.server
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error

B = Path("/workspace/bundle")
OUT = Path("/workspace/out")
DEADLINE = float(os.environ.get("DEF_DEADLINE", "0"))
REST = "https://rest.runpod.io/v1/pods/"
UPLOAD_MARGIN = 12 * 60


class OperationalError(RuntimeError):
    """Only explicit, nonsecret fields may be written to status/logs."""
    def __init__(self, code, **details):
        self.safe_details = {"error": type(self).__name__, "code": code, **details}
        super().__init__(code)


def error_details(exc):
    return exc.safe_details if isinstance(exc, OperationalError) else {"error": type(exc).__name__}


def atomic(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, indent=2)); os.replace(temp, path)


def state(phase, **kw):
    atomic(OUT / "STATUS.json", {"phase": phase, "time": time.time(), "deadline": DEADLINE, **kw})
    print(phase, json.dumps(kw), flush=True)


def pod_api(method="GET", suffix=""):
    # RUNPOD_API_KEY is provider-injected and pod-scoped. The launcher's explicit
    # credential uses a separate namespace; never fall back to an unknown key.
    api_key = os.environ.get("DEF_RUNPOD_API_KEY")
    if not api_key:
        raise OperationalError("missing_campaign_api_key")
    req = urllib.request.Request(REST + os.environ["RUNPOD_POD_ID"] + suffix, method=method,
                                 headers={"Authorization": "Bearer " + api_key,
                                          "User-Agent": "definition-campaign/1.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"deleted": True}
        raise OperationalError("pod_api_http", method=method, http_status=e.code) from None
    except urllib.error.URLError:
        raise OperationalError("pod_api_transport", method=method) from None


def terminate():
    """Bounded DELETE retries with positive GET verification; status monitor retries independently."""
    for _ in range(4):
        try:
            pod_api("DELETE")
            if pod_api().get("deleted"):
                return True
        except Exception:
            pass
        time.sleep(3)
    # Stop prevents ongoing GPU burn while the external monitor deletes the volume.
    try:
        pod_api("POST", "/stop")
    except Exception:
        pass
    return False


def watchdog():
    while time.time() < DEADLINE:
        (OUT / "HEARTBEAT").write_text(str(int(time.time())))
        time.sleep(min(10, max(0, DEADLINE - time.time())))
    state("hard_deadline")
    terminate()
    os._exit(124)


def log_failure_categories(path):
    """Extract fixed categories, never arbitrary subprocess text or signed URLs."""
    try:
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - 16384))
            text = f.read().decode("utf-8", errors="replace").lower()
    except OSError:
        return []
    patterns = {
        "dependency_unavailable": r"no matching distribution|could not find a version",
        "authentication": r"unauthorized|authentication failed|invalid api token|could not find kaggle.json",
        "access_denied": r"forbidden|permission denied",
        "missing_file": r"no such file or directory|filenotfounderror",
        "disk_full": r"no space left on device",
        "missing_module": r"modulenotfounderror|importerror",
        "invalid_arguments": r"unrecognized arguments|invalid choice",
        "network": r"connectionerror|connection refused|temporary failure|timed out|name resolution",
        "http_401": r"\b401\b", "http_403": r"\b403\b", "http_404": r"\b404\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def run(cmd, timeout, logname="setup.log", env=None, stage="subprocess"):
    remaining = DEADLINE - time.time() - 90
    if remaining <= 0:
        raise TimeoutError("deadline reached")
    with (OUT / logname).open("a") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=min(timeout, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            details = {"stage": stage, "returncode": p.returncode, "timeout": True,
                       "failure_categories": log_failure_categories(OUT / logname)}
            atomic(OUT / "subprocess_receipt.json", details)
            raise OperationalError("subprocess_timeout", **details) from None
    if rc:
        details = {"stage": stage, "returncode": rc,
                   "failure_categories": log_failure_categories(OUT / logname)}
        atomic(OUT / "subprocess_receipt.json", details)
        raise OperationalError("subprocess_failed", **details)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def prepare_upload(up, reason, diagnostic_only=False):
    shutil.rmtree(up, ignore_errors=True); up.mkdir(parents=True)
    safe_diagnostics = {"STATUS.json", "failure.json", "subprocess_receipt.json"}
    for p in OUT.rglob("*"):
        eligible = p.name in safe_diagnostics if diagnostic_only else p.suffix in (".pt", ".npz", ".json", ".log", ".txt")
        if p.is_file() and eligible and not p.name.endswith(".tmp"):
            dest = up / p.relative_to(OUT); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    if diagnostic_only:
        (up / "NOTICE.txt").write_text("Private operational diagnostics only. No competition images, labels, model weights, raw logs, credentials or environment values are included.\n")
    else:
        shutil.copy2(B / "NOTICE.txt", up / "NOTICE.txt")
        shutil.copy2(B / "definition_v1.manifest.json", up / "definition_v1.manifest.json")
    receipt = {"reason": reason, "time": time.time(), "diagnostic_only": diagnostic_only,
               "sha256": {str(p.relative_to(up)): sha(p) for p in up.rglob("*") if p.is_file()}}
    atomic(up / "result_manifest.json", receipt)
    atomic(up / "dataset-metadata.json", {"title": os.environ["DEF_RESULT"].split("/")[-1],
           "id": os.environ["DEF_RESULT"], "licenses": [{"name": "other"}]})


def publish(reason, lock, diagnostic_only=False):
    """Snapshot atomically saved checkpoint files, then verify remote manifest after upload.

    Whole model uploads occur only after first completed epoch / >=30 minutes /
    finalization, never every polling tick. Logs stay in private account dataset.
    """
    with lock:
        up = Path("/workspace/upload")
        prepare_upload(up, reason, diagnostic_only)
        exists = (OUT / "UPLOADED").exists()
        cmd = ["kaggle", "datasets", "version", "-p", str(up), "--dir-mode", "zip", "-m", reason, "-q"] if exists else [
            "kaggle", "datasets", "create", "-p", str(up), "--dir-mode", "zip", "-q"]
        try:
            run(cmd, 300, "upload.log")
        except RuntimeError:
            if exists: raise
            # Previous upload may have succeeded before its response was lost.
            run(["kaggle", "datasets", "version", "-p", str(up), "--dir-mode", "zip", "-m", reason, "-q"], 300, "upload.log")
        verify = Path("/workspace/verify")
        for _ in range(4):
            shutil.rmtree(verify, ignore_errors=True); verify.mkdir()
            try:
                run(["kaggle", "datasets", "download", os.environ["DEF_RESULT"], "-f", "result_manifest.json",
                     "-p", str(verify), "--force", "-q"], 60, "upload.log")
                if (verify / "result_manifest.json").exists() and sha(verify / "result_manifest.json") == sha(up / "result_manifest.json"):
                    (OUT / "UPLOADED").write_text(str(time.time()))
                    atomic(OUT / "upload_receipt.json", {"verified": True, "time": time.time(), "reason": reason,
                                                         "manifest_sha256": sha(up / "result_manifest.json")})
                    return True
            except Exception:
                pass
            time.sleep(10)
        raise RuntimeError("remote snapshot manifest not yet verified")


def private_handler(status_token, directory):
    """Capture the immutable HTTP token independently of bootstrap credentials."""
    class PrivateHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + status_token):
                self.send_error(403); return
            super().do_GET()
        def do_HEAD(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + status_token):
                self.send_error(403); return
            super().do_HEAD()
        def log_message(self, *args): pass
    return functools.partial(PrivateHandler, directory=str(directory))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(DEADLINE) or DEADLINE <= time.time():
        terminate(); return
    threading.Thread(target=watchdog, daemon=True).start()
    status_token = os.environ.get("DEF_STATUS_TOKEN")
    if not status_token:
        terminate(); return
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), private_handler(status_token, OUT))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    lock = threading.Lock(); upload_thread = None; uploaded_once = False
    phase = "checking_quote"; failed = False; bundle_verified = False
    try:
        state("checking_quote")
        p = pod_api()
        try:
            price = float(p.get("costPerHr", 0))
            ceiling = float(os.environ["DEF_MAX_GPU_HR"])
        except (TypeError, ValueError):
            raise OperationalError("pod_quote_invalid") from None
        if not math.isfinite(price) or not math.isfinite(ceiling) or not 0 < price <= ceiling:
            raise OperationalError("pod_quote_outside_cap", observed_gpu_hr=price if math.isfinite(price) else None,
                                   ceiling_gpu_hr=ceiling if math.isfinite(ceiling) else None)
        phase = "installing_kaggle"; state(phase, gpu_hr=price)
        kagdir = Path.home() / ".kaggle"; kagdir.mkdir(exist_ok=True)
        kaggle_token_file = kagdir / "access_token"
        kaggle_token_file.write_text(os.environ["KAGGLE_API_TOKEN"]); kaggle_token_file.chmod(0o600)
        run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "kaggle==2.2.4"], 300, stage=phase)
        phase = "bundle_download"; state(phase)
        run(["kaggle", "datasets", "download", os.environ["DEF_BUNDLE"], "-p", str(B), "--unzip", "-q"], 300, stage=phase)
        phase = "bundle_verify"; state(phase)
        for name, digest in json.loads((B / "bundle_manifest.json").read_text())["sha256"].items():
            if not (B / name).resolve().is_relative_to(B.resolve()) or sha(B / name) != digest:
                raise RuntimeError("bundle hash mismatch")
        cfg = json.loads((B / "job.json").read_text())
        if sha(B / "definition_v1.npz") != cfg["labels_sha256"]:
            raise RuntimeError("label hash mismatch")
        bundle_verified = True
        label_names = json.loads((B / "definition_v1.manifest.json").read_text())["labels"]
        with (B / "train.csv").open() as f:
            expert = [r["StudyInstanceUID"] for r in csv.DictReader(f)
                      if all(r[c].strip().lower() not in ("", "nan") for c in label_names)]
        if len(expert) != 58:
            raise RuntimeError("expert diagnostic set differs from 58")
        (B / "expert_ids.txt").write_text("\n".join(expert) + "\n")
        # Pin versions used by this isolated new trainer; torch stays the image's build.
        phase = "installing_training_dependencies"; state(phase)
        run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "numpy==1.26.4",
             "pandas==2.2.3", "scikit-learn==1.5.2", "timm==1.0.30", "opencv-python-headless==4.10.0.84",
             "pyarrow==19.0.1", "requests==2.32.3"], 600, stage=phase)
        phase = "cuda_import_check"; state(phase)
        run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); import timm,cv2,numpy,pandas"], 60, stage=phase)
        phase = "cache_download"; state(phase)
        env = dict(os.environ, DENSE_NAME="dense512", DENSE_SHARDS="8", DENSE_PAR="4", DENSE80_DIR="/workspace/input/dense512", DENSE80_TRIES="2")
        run([sys.executable, str(B / "fetch_dense80.py")], min(2400, max(1, DEADLINE - time.time() - 1800)), env=env, stage=phase)
        if DEADLINE - time.time() < 1800:
            raise RuntimeError("setup used pilot window; no time for meaningful training")
        last_publish = 0.0
        def snapshot(reason):
            try: publish(reason, lock)
            except Exception as e: state("snapshot_failed", **error_details(e))
        for i, mode in enumerate(cfg["modes"]):
            folder = OUT / mode; folder.mkdir(exist_ok=True)
            # Reserve up to 5min report-fold eval + 5min gold diagnostic per arm.
            seconds = int((DEADLINE - time.time() - UPLOAD_MARGIN) / (len(cfg["modes"]) - i)) - 600
            if seconds < 300: raise RuntimeError("training budget exhausted")
            cmd = [sys.executable, str(B / "definition_reader.py"), "--data-root", "/workspace/input/dense512",
                   "--n-shards", "8", "--labels", str(B / "definition_v1.npz"), "--train-csv", str(B / "train.csv"),
                   "--series-csv", str(B / "train_series.csv"), "--out", str(folder), "--tag", mode,
                   "--target-mode", mode, "--views", "global-local", "--aux-weight", "0", "--fold", "0",
                   "--epochs", str(cfg["epochs"]), "--max-seconds", str(seconds), "--arch", "convnext_tiny.fb_in1k",
                   "--bb-file", str(B / "convnext_tiny.fb_in1k.pt"),
                   "--size", "224", "--windows", "2", "--bs", "2", "--workers", "4", "--accum", "4",
                   "--grad-checkpoint", "--eval-max-seconds", "260"]
            resume = folder / f"definition_{mode}_resume.pt"
            if not resume.exists() and (B / f"resume_{mode}.pt").exists():
                shutil.copy2(B / f"resume_{mode}.pt", resume)
            if resume.exists(): cmd += ["--resume", str(resume)]
            phase = "training"; state(phase, mode=mode, max_seconds=seconds)
            with (folder / "train.log").open("a") as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.time()
                last_epoch = -1
                while proc.poll() is None:
                    if time.time() > min(DEADLINE - UPLOAD_MARGIN - 300, started + seconds + 300):
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(30)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        break
                    progress_path = folder / "progress.json"
                    if progress_path.exists():
                        try:
                            progress = json.loads(progress_path.read_text())
                            epoch = int(progress["epoch"])
                            if epoch > last_epoch:
                                last_epoch = epoch
                                state("training", mode=mode, progress=progress, allocated_train_seconds=seconds,
                                      projected_train_seconds=(round(progress["elapsed_seconds"] / epoch * cfg["epochs"]) if epoch else None))
                        except (ValueError, KeyError, OSError):
                            pass
                    if resume.exists() and (not uploaded_once or time.time() - last_publish > 1800) and (upload_thread is None or not upload_thread.is_alive()):
                        upload_thread = threading.Thread(target=snapshot, args=(f"epoch-{mode}",), daemon=True)
                        upload_thread.start(); last_publish = time.time(); uploaded_once = True
                    time.sleep(10)
            atomic(folder / "process_receipt.json", {"returncode": proc.returncode, "wall_seconds": time.time()-started,
                                                    "cmd": cmd, "time": time.time()})
            if proc.returncode != 0:
                raise RuntimeError(f"trainer failed rc={proc.returncode}; no automatic paid retry")
            ema = folder / f"definition_{mode}_ema.pt"
            if not ema.exists(): raise RuntimeError("trainer left no deployable EMA")
            phase = "expert_diagnostic"; state(phase, mode=mode)
            run([sys.executable, str(B / "definition_reader.py"), "--data-root", "/workspace/input/dense512",
                 "--n-shards", "8", "--labels", str(B / "definition_v1.npz"), "--train-csv", str(B / "train.csv"),
                 "--series-csv", str(B / "train_series.csv"), "--out", str(folder), "--tag", mode,
                 "--predict", str(ema), "--predict-ids", str(B / "expert_ids.txt"), "--prediction-out", str(folder / f"definition_{mode}_expert58.npz"),
                 "--workers", "4", "--bs", "2", "--eval-max-seconds", "260"], 300, logname=f"{mode}/expert_predict.log", stage=phase)
        state("arms_returned", note="Check per-arm reports for completed epochs vs bounded early stop; expert58 is reused diagnostic, not fresh validation")
    except Exception as e:
        failed = True
        details = {"failed_phase": phase, **error_details(e),
                   "frames": [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                              for f in traceback.extract_tb(e.__traceback__)[-5:]]}
        atomic(OUT / "failure.json", details)
        state("failed", **details)  # no URLs, env or credential-containing exception repr
    finally:
        if upload_thread is not None:
            upload_thread.join(timeout=max(0, min(360, DEADLINE - time.time() - 120)))
        if (bundle_verified or (failed and shutil.which("kaggle"))) and time.time() < DEADLINE - 90:
            try:
                publish("bootstrap-failed" if not bundle_verified else "final", lock, diagnostic_only=not bundle_verified)
                state("final_uploaded", diagnostic_only=not bundle_verified, training_failed=failed)
            except Exception as e:
                state("final_upload_failed", **error_details(e))
                # Preserve volume until hard deadline so host can collect live files;
                # GPU and storage still fit the original total-wall reservation.
                recovery_end = min(DEADLINE - 60, time.time() + 120) if not bundle_verified else DEADLINE - 60
                while time.time() < recovery_end:
                    time.sleep(10)
        elif failed:
            # CLI bootstrap itself may have failed. Keep authenticated diagnostics
            # briefly available without spending the rest of the training window.
            recovery_end = min(DEADLINE - 60, time.time() + 120)
            state("bootstrap_diagnostic_grace", seconds=max(0, int(recovery_end-time.time())))
            while time.time() < recovery_end:
                time.sleep(10)
        terminate()


if __name__ == "__main__":
    main()
