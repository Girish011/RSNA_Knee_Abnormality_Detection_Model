# ===== FORK: DINO warm-start fine-tune on OUR fused labels (cross-architecture distillation) =====
# Appended after the public trainer's definitions (pilkwang/rsna-knee-baseline-v1) by
# scripts/build_dino_ft_kernel.py; replaces its main(). Each of the 20 public members
# (pilkwang/rsna-knee-weights, 4 seeds x 5 folds) is loaded, fingerprint-checked, fine-tuned
# for FT_EPOCHS at 1/5 of the public LR on its own training folds (grp != fold) with targets =
# our fused labels (image-read gold wins at weight 3), then predicts its holdout fold before and
# after (paired OOF). The frozen prefix (embeddings + first 6 blocks) never moves, so the stack's
# SHARED_DINO_PREFIX_SHA256 gate still holds. Output: /kaggle/working/weights (manifest v2 +
# m_<id>.pt, drop-in for the stack) and ft_oof.npz.
import queue
import threading

FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "3"))
FT_LR_HEAD = LR_HEAD / 5
FT_LR_BACKBONE = LR_BACKBONE / 5
FT_PCT_START = 0.10
FT_LABELS = "labels_fused_full.csv"
FT_TAG = "ft-fused"
FT_SMOKE = os.environ.get("FT_SMOKE") == "1"   # local test: synthetic cache, no DICOMs
TIME_BUDGET = 11.0 * 3600                      # 12 h session; build_cache scales its order budget on this
OUT = Path(os.environ.get("FT_OUT", "/kaggle/working"))
BUILD_LOCK = threading.Lock()
if os.environ.get("FT_DINO"):                 # local smoke: HF snapshot dir instead of /kaggle/input
    find_dinov2 = lambda variant="small": Path(os.environ["FT_DINO"])
LOG_LOCK = threading.Lock()


def find_input(name):
    for root in [Path("/kaggle/input"), Path(os.environ.get("FT_INPUT", "."))]:
        if root.is_dir():
            for p in root.rglob(name):
                return p
    raise FileNotFoundError(name)


def prefix_sha(state):
    """Same key set as the stack's _shared_prefix_state_sha256 (embeddings + blocks 0-5)."""
    pre = ("backbone.embeddings.", *(f"backbone.encoder.layer.{i}." for i in range(6)))
    h = hashlib.sha256()
    for k in sorted(k for k in state if k.startswith(pre)):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def cache_dir_for(nbytes):
    import shutil
    cands = ["/kaggle/tmp", "/tmp", "/root", "/kaggle"]
    free = {}
    for c in cands:
        try:
            Path(c).mkdir(parents=True, exist_ok=True)
            free[c] = shutil.disk_usage(c).free
        except OSError:
            pass
    log("disk free: " + ", ".join(f"{c} {v / 1024 ** 3:.0f} GB" for c, v in free.items()))
    ok = [c for c, v in free.items() if v > nbytes * 1.05 + 3 * 1024 ** 3]
    if not ok:
        raise RuntimeError(f"no disk with {nbytes / 1024 ** 3:.1f} GB free for the train cache")
    return max(ok, key=lambda c: free[c])


def ft_targets(st_tr, train_df):
    uid = "StudyInstanceUID"
    fused = pd.read_csv(find_input(FT_LABELS), dtype={uid: str}).set_index(uid)
    if fused.index.duplicated().any():
        raise ValueError("duplicate study in fused labels")
    gold = train_df.set_index(uid)[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    Y = np.zeros((len(st_tr), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    n_f = 0
    for i, st in enumerate(st_tr):
        if st in gold.index:
            Y[i], W[i] = gold.loc[st].values, 3.0
        elif st in fused.index:
            y = fused.loc[st, TARGETS].to_numpy(np.float32)
            if np.isnan(y).any():
                continue
            Y[i] = y
            W[i] = 0.25 + 0.75 * np.clip(np.abs(2 * y - 1), 0, 1)
            n_f += 1
    log(f"[ours] targets: {n_f} fused + {int(np.isin(st_tr, gold.index).sum())} gold of {len(st_tr)}")
    return Y, W, gold


def report_groups(st_tr, train_df):
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    return np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5
                     for s in st_tr])


def fine_tune_member(m, pkg, dev, Ctr, Mtr, Y, W, grp, gold_idx, gold_y):
    tag = f"{m['id']} f{m['fold']}: "
    fold = int(m["fold"])
    with BUILD_LOCK:
        ck = torch.load(Path(pkg) / m["file"], map_location="cpu", weights_only=False)
        model = build_model(int(m["config"]["unfreeze_last"]), variant=m["config"]["variant"],
                            pool=m["config"].get("pool", "cls_mean"),
                            prior=bool(m["config"].get("prior", False)))
    model.load_state_dict(ck["model"])
    model = model.to(dev)
    check_fingerprint(model, dev, IMG, ck["fingerprint"], tag=tag)
    sha0 = prefix_sha(model.state_dict())

    ho = np.where(grp == fold)[0]
    tr = np.where((grp != fold) & (W.sum(1) > 0))[0]
    # The checkpoint's own holdout must be exactly this fold, or its OOF is not OOF.
    ck_ids = set(map(str, ck["oof"]["ids"])) if isinstance(ck.get("oof"), dict) else None
    if ck_ids is not None and not FT_SMOKE and ck_ids != set(ST_TR[ho].tolist()):
        raise RuntimeError(f"{tag}checkpoint holdout != md5 fold {fold}")
    pre = predict(model, Ctr, Mtr, ho, dev, IMG)
    if ck_ids is not None and not FT_SMOKE:
        pos = {s: i for i, s in enumerate(map(str, ck["oof"]["ids"]))}
        ref = np.asarray(ck["oof"]["pred"], np.float32)[[pos[s] for s in ST_TR[ho]]]
        corr = float(np.mean([np.corrcoef(ref[:, j], pre[:, j])[0, 1] for j in range(len(TARGETS))]))
        log(f"{tag}pixel check vs stored OOF: mean corr {corr:.4f}, max |d| {np.abs(ref - pre).max():.3f}")

    gsel = np.array([i for i in gold_idx if grp[i] == fold])
    gy = gold_y[np.isin(gold_idx, gsel)] if len(gsel) else None

    seed = int(m.get("seed", SEED)) + 7 * fold + 1
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": FT_LR_BACKBONE},
        {"params": model.head.parameters(), "lr": FT_LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    per_ep = max(len(tr) // BATCH_STUDIES, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[FT_LR_BACKBONE, FT_LR_HEAD], total_steps=FT_EPOCHS * per_ep,
        pct_start=FT_PCT_START)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    t0 = time.time()
    hist = []
    for ep in range(FT_EPOCHS):
        model.train()
        perm = rng.permutation(tr)
        tot, n = 0.0, 0
        for b in range(0, per_ep * BATCH_STUDIES, BATCH_STUDIES):
            sel = np.sort(perm[b:b + BATCH_STUDIES])          # sorted: sequential memmap reads
            g = int(rng.integers(N_GROUP))
            rows = torch.from_numpy(np.ascontiguousarray(Ctr[sel, :, g * GROUP:(g + 1) * GROUP])).to(dev)
            imgs = augment(rows)
            mk = torch.from_numpy(Mtr[sel]).to(dev)
            y = torch.from_numpy(Y[sel]).to(dev)
            w = torch.from_numpy(W[sel]).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = (F.binary_cross_entropy_with_logits(
                    model(imgs, mk, IMG), y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            n += 1
        ga = macro_auc(gy, predict(model, Ctr, Mtr, gsel, dev, IMG)) if gy is not None else float("nan")
        hist.append({"epoch": ep + 1, "loss": tot / max(n, 1), "gold_ho": ga})
        log(f"{tag}epoch {ep + 1}/{FT_EPOCHS} loss {tot / max(n, 1):.4f} gold-holdout(n={len(gsel)}) "
            f"{ga:.4f}  [{time.time() - t0:.0f}s]")

    post = predict(model, Ctr, Mtr, ho, dev, IMG)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if prefix_sha(state) != sha0:
        raise RuntimeError(f"{tag}frozen prefix moved during fine-tune")
    fp = fingerprint(model, dev, IMG)
    new_id = hashlib.sha256(f"{m['id']}|{FT_TAG}|{FT_EPOCHS}".encode()).hexdigest()[:10]
    yb = (Y[ho] > 0.5).astype(int)
    out = dict(ck)
    out.update(model=state, fingerprint=fp.tolist(), epochs_done=int(ck.get("epochs_done", 0)) + FT_EPOCHS,
               oof={"ids": ST_TR[ho].tolist(), "pred": post},
               ft={"parent": m["id"], "labels": FT_LABELS, "epochs": FT_EPOCHS, "lr_head": FT_LR_HEAD,
                   "lr_backbone": FT_LR_BACKBONE, "hist": hist, "prefix_sha": sha0})
    (OUT / "weights").mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT / "weights" / f"m_{new_id}.pt")
    entry = {k: v for k, v in m.items()}
    entry.update(id=new_id, file=f"m_{new_id}.pt", fingerprint=fp.tolist(), parent=m["id"], tag=FT_TAG,
                 epochs_done=out["epochs_done"], holdout=macro_auc(yb, post),
                 holdout_parent=macro_auc(yb, pre))
    log(f"{tag}-> {new_id}: fused-holdout {entry['holdout_parent']:.4f} -> {entry['holdout']:.4f} "
        f"in {time.time() - t0:.0f}s")
    del model, opt, sched, scaler, ck
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return entry, ho, pre, post


def ft_main():
    global ST_TR
    pkg = Path(os.environ["FT_PKG"]) if os.environ.get("FT_PKG") else find_weights()
    if pkg is None:
        raise WeightsError("pilkwang/rsna-knee-weights not attached")
    man = json.loads((Path(pkg) / "manifest.json").read_text())
    members = man["members"]
    if os.environ.get("FT_LIMIT"):
        members = members[:int(os.environ["FT_LIMIT"])]
    keys = {m["pixel_group"] for m in members}
    if len(keys) != 1:
        raise WeightsError(f"expected one pixel group, found {len(keys)}")
    adopt_config_globals(json.loads(keys.pop()))
    log(f"{len(members)} members; pixels {IMG}px x {CACHE_SLICES} slices (groups {N_GROUP}); "
        f"FT {FT_EPOCHS} ep, lr head {FT_LR_HEAD:g} backbone {FT_LR_BACKBONE:g}")

    train_df = pd.read_csv(ROOT / "train.csv")
    if FT_SMOKE:
        n = int(os.environ.get("FT_SMOKE_N", "40"))
        st_tr = np.array(sorted(train_df["StudyInstanceUID"].astype(str))[:n])
        Ctr = np.random.default_rng(0).integers(0, 255, (n, N_SLOT, CACHE_SLICES, IMG, IMG), dtype=np.uint8)
        Mtr = np.ones((n, N_SLOT), np.float32)
    else:
        train_series = pd.read_csv(ROOT / "train_series.csv")
        plane_map = dict(zip(train_series["SeriesInstanceUID"], train_series["Anatomical_Plane"]))
        log("header pass: train")
        htr = annotate(walk("train_series"))
        slots_tr = pick_slots(htr, plane_map)
        nbytes = len(slots_tr) * N_SLOT * CACHE_SLICES * IMG * IMG
        CACHE_MEMMAP[0] = cache_dir_for(nbytes)
        st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_of(htr, "train "), "train")
        st_tr = np.array(st_tr)
    ST_TR = st_tr
    Y, W, gold = ft_targets(st_tr, train_df)
    grp = report_groups(st_tr, train_df)
    pos = {s: i for i, s in enumerate(st_tr)}
    gold_idx = np.array([pos[s] for s in gold.index if s in pos])
    gold_y = gold.loc[st_tr[gold_idx]].values.astype(int) if len(gold_idx) else None

    devs = ([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            or [torch.device("cpu")])
    log(f"devices: {devs}")
    todo = queue.Queue()
    for m in members:
        todo.put(m)
    results, errors = [], []
    t_member = []

    def worker(dev):
        while True:
            try:
                m = todo.get_nowait()
            except queue.Empty:
                return
            left = TIME_BUDGET - (time.time() - T0)
            est = (max(t_member) if t_member else 0) * 1.3
            if est and left < est:
                log(f"{m['id']}: skipped, {left / 60:.0f} min left < {est / 60:.0f} min estimate")
                continue
            t = time.time()
            try:
                if dev.type == "cuda":
                    with torch.cuda.device(dev):
                        r = fine_tune_member(m, pkg, dev, Ctr, Mtr, Y, W, grp, gold_idx, gold_y)
                else:
                    r = fine_tune_member(m, pkg, dev, Ctr, Mtr, Y, W, grp, gold_idx, gold_y)
            except Exception as e:                       # keep the other members going
                traceback.print_exc()
                errors.append((m["id"], repr(e)))
                continue
            t_member.append(time.time() - t)
            with LOG_LOCK:
                results.append(r)
                write_outputs(man, members, results, st_tr, gold_idx, gold_y, grp)

    th = [threading.Thread(target=worker, args=(d,)) for d in devs]
    for x in th:
        x.start()
    for x in th:
        x.join()
    log(f"done: {len(results)}/{len(members)} members fine-tuned; errors {errors}")
    if errors:
        raise RuntimeError(f"{len(errors)} member(s) failed: {errors[:3]}")


def write_outputs(man, members, results, st_tr, gold_idx, gold_y, grp):
    """Manifest v2 in the public member order + paired OOF arrays; rewritten after every member."""
    by_parent = {r[0]["parent"]: r for r in results}
    new_members = [by_parent[m["id"]][0] for m in members if m["id"] in by_parent]
    man2 = {k: v for k, v in man.items() if k != "members"}
    man2["members"] = new_members
    man2["history"] = list(man.get("history", []))
    man2["history"].append({"run": FT_TAG, "tag": FT_TAG,
                                           "added": [m["id"] for m in new_members],
                                           "replaced": [m["parent"] for m in new_members],
                                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (OUT / "weights").mkdir(parents=True, exist_ok=True)
    (OUT / "weights" / "manifest.json").write_text(json.dumps(man2, indent=1))
    n = len(st_tr)
    pre = np.full((len(members), n, len(TARGETS)), np.nan, np.float32)
    post = np.full_like(pre, np.nan)
    order = {m["id"]: i for i, m in enumerate(members)}
    for e, ho, p0, p1 in results:
        pre[order[e["parent"]], ho] = p0
        post[order[e["parent"]], ho] = p1
    np.savez_compressed(OUT / "ft_oof.npz", ids=np.asarray(st_tr), pre=pre, post=post,
                        member_ids=np.array([m["id"] for m in members]),
                        fold=np.array([int(m["fold"]) for m in members]), grp=grp, targets=np.array(TARGETS))
    with np.errstate(all="ignore"):
        mp0, mp1 = np.nanmean(pre, 0), np.nanmean(post, 0)
    have = ~np.isnan(mp1).any(1)
    gi = np.array([i for i in gold_idx if have[i]])
    if gold_y is not None and len(gi) >= 10:
        gy = gold_y[np.isin(gold_idx, gi)]
        log(f"[gate so far] gold OOF macro (n={len(gi)}): parent {macro_auc(gy, mp0[gi]):.4f} -> "
            f"ft {macro_auc(gy, mp1[gi]):.4f}  ({len(results)} members)")


try:
    ft_main()
except Exception:
    traceback.print_exc()
    raise
log("done")
