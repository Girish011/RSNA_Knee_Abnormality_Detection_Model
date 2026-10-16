import glob, os, subprocess, time, shutil
from concurrent.futures import ThreadPoolExecutor
print(subprocess.run("df -h /kaggle/working /tmp /kaggle/temp /dev/shm 2>&1; free -g | head -2; nproc", shell=True, capture_output=True, text=True).stdout, flush=True)
fs = sorted(glob.glob("/kaggle/input/**/*.npz", recursive=True))
print("npz files", len(fs), "| total GB", round(sum(os.path.getsize(f) for f in fs) / 1e9, 2), flush=True)
def rd(f):
    with open(f, "rb") as h:
        return len(h.read())
for nthr, part in ((1, fs[:20]), (4, fs[20:100]), (16, fs[100:420]), (32, fs[420:735])):
    t = time.time()
    with ThreadPoolExecutor(nthr) as ex:
        n = sum(ex.map(rd, part))
    dt = time.time() - t
    print(f"threads {nthr}: {len(part)} files, {n / 1e6:.0f} MB in {dt:.1f}s = {n / 1e6 / dt:.1f} MB/s", flush=True)
os.makedirs("/tmp/cx", exist_ok=True)
t = time.time(); n = 0
for f in fs[:40]:
    shutil.copy(f, "/tmp/cx/"); n += os.path.getsize(f)
print(f"re-read (cached) + write to /tmp: {n / 1e6:.0f} MB in {time.time() - t:.1f}s", flush=True)
import numpy as np
t = time.time()
for f in sorted(glob.glob("/tmp/cx/*.npz"))[:20]:
    z = np.load(f); [z[k] for k in z.files]
print(f"decompress 20 studies from local disk: {time.time() - t:.1f}s", flush=True)
