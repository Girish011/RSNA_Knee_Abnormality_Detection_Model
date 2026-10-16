
import os, shutil, subprocess, sys, time
W = "/kaggle/working"
ARMS = [('r1-ksf-f0', 'r1ksf0', '--arch convnext_tiny.in12k_ft_in1k --res 224 --epochs 40 --bs 8 --k 26 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0 --aug_erase 0.25 --drop_path 0.2 --seed 42 --folds 5 --fold 0 --chan seqfuse'), ('r1-ksf-f1', 'r1ksf1', '--arch convnext_tiny.in12k_ft_in1k --res 224 --epochs 40 --bs 8 --k 26 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0 --aug_erase 0.25 --drop_path 0.2 --seed 42 --folds 5 --fold 1 --chan seqfuse')]            # [(job, tag, args)]
LABELS = 'labels_r1.parquet'


def find(name):
    for d, _, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(d, name)
    raise SystemExit(name + " not found under /kaggle/input")


subprocess.call("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
trainer = find("train_knee.py")
proto = f"--labels {find(LABELS)} --fold_file {find('folds_nonexpert_5.json')} --holdout {find('expert_holdout.json')}"


def launch(smoke):
    procs = []
    for g, (job, tag, args) in enumerate(ARMS):
        d = f"{W}/{tag}" + ("_smoke" if smoke else "")
        os.makedirs(d, exist_ok=True)
        shutil.copy2(trainer, d + "/train_knee.py")
        arch = args.split("--arch ")[1].split()[0]
        cmd = (f"{sys.executable} -u train_knee.py --tag {tag} {proto} {args} --bb_file {arch}.pth --amp fp16 --workers 2"
               + (" --smoke" if smoke else ""))
        print(f"[ktrain] gpu {g} | {job} | {cmd}", flush=True)
        procs.append((tag, subprocess.Popen(cmd, shell=True, cwd=d, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                                            stdout=open(d + "/train.log", "w"), stderr=subprocess.STDOUT)))
    return procs


# smoke run first (trainer's --smoke: 40 studies, 2 epochs): a wiring failure costs one mount, not a session
t0 = time.time()
smoke = launch(True)
for tag, p in smoke:
    p.wait()
    print(f"[ktrain] smoke {tag} exit {p.returncode} | {(time.time() - t0) / 60:.0f} min", flush=True)
    for ln in open(f"{W}/{tag}_smoke/train.log").read().splitlines()[-25:]:
        print(f"[smoke {tag}] {ln}", flush=True)
    shutil.copy2(f"{W}/{tag}_smoke/train.log", f"{W}/smoke_{tag}.log")
    shutil.rmtree(f"{W}/{tag}_smoke")
if any(p.returncode for _, p in smoke):
    raise SystemExit("smoke run failed")
procs = launch(False)
seen = {tag: 0 for tag, _ in procs}
t0 = time.time()
while True:
    alive = False
    for tag, p in procs:
        alive |= p.poll() is None
        try:
            lines = open(f"{W}/{tag}/train.log").read().splitlines()
        except OSError:
            lines = []
        for ln in lines[seen[tag]:]:
            if "Warning" not in ln and "warn" not in ln and ln.strip():
                print(f"[{tag}] {ln}", flush=True)
        seen[tag] = len(lines)
    if not alive:
        break
    time.sleep(60)
for tag, p in procs:
    print(f"[ktrain] {tag} exit {p.returncode} | {(time.time() - t0) / 60:.0f} min", flush=True)
    os.remove(f"{W}/{tag}/train_knee.py")
if any(p.returncode for _, p in procs):
    raise SystemExit("an arm failed")
