# Kaggle GPU T4x2 — expert fine-tune fold0 from FROZEN DINOv2-B (0.759)
# Train on gold experts outside fold0; validate on FULL fold0 (cache_v1).
#
# Attach:
#   competition rsna-knee-abnormality-detection
#   girishbose/rsna-knee-code
#   girishbose/dinov2-vitb14-rsna-knee          ← B weights, not vits14
#   girishbose/rsna-knee-cache-v1               ← not cache_v2
#   notebook OUTPUT: main-dinov2b-fold0-frozen-weak-v1  (must contain fold0_best.pt)
#
# Settings: GPU T4 x2. Do NOT pip install torch.
# Save Version: expert-ft-b-fold0-cache-v1
# Gate: keep only if best > before-FT (~0.759). Do not compare to the missing 0.764 S ckpt.

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("Set Accelerator to GPU T4 x2 and Save Version again. Do not pip install torch.")

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
WEIGHTS = Path("/kaggle/input/datasets/girishbose/dinov2-vitb14-rsna-knee/dinov2_vitb14_pretrain.pth")
CACHE = Path("/kaggle/input/notebooks/girishbose/rsna-knee-cache-v1/cache_v1")
SRC = CODE / "src"

if not WEIGHTS.exists():
    hits = list(Path("/kaggle/input").rglob("dinov2_vitb14_pretrain.pth"))
    WEIGHTS = hits[0] if hits else WEIGHTS
if not CACHE.exists():
    hits = [p for p in Path("/kaggle/input").rglob("cache_v1") if p.is_dir() and "cache-v2" not in str(p) and "cache_v2" not in str(p)]
    CACHE = hits[0] if hits else CACHE

ckpts = sorted(Path("/kaggle/input").rglob("fold0_best.pt"))
print("found fold0_best.pt:", [str(p) for p in ckpts], flush=True)
prefer = [
    p for p in ckpts
    if "cache_v2" not in str(p)
    and ("dinov2_b" in str(p) or "vitb" in str(p) or "frozen" in str(p) or "main_dinov2" in str(p))
]
INIT_CKPT = (prefer or ckpts)[0] if (prefer or ckpts) else None
assert INIT_CKPT is not None, (
    "Attach the frozen-B notebook output (main-dinov2b-fold0-frozen-weak-v1) so fold0_best.pt is under /kaggle/input. "
    f"Inputs: {list(Path('/kaggle/input').iterdir())}"
)
print("WEIGHTS", WEIGHTS, WEIGHTS.exists(), flush=True)
print("CACHE", CACHE, CACHE.exists(), flush=True)
print("INIT_CKPT", INIT_CKPT, flush=True)
assert WEIGHTS.exists() and CACHE.exists()

os.environ["PYTHONPATH"] = str(SRC)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dino = CODE / "third_party" / "dinov2"
if (dino / "hubconf.py").exists():
    os.environ["DINOV2_REPO"] = str(dino)
sys.path.insert(0, str(SRC))

from rsna_knee.constants import LABEL_COLS
from rsna_knee.data.cached_dataset import CachedStudyDataset, attach_folds, merge_weak_labels
from rsna_knee.data.dataset import collate_studies
from rsna_knee.metrics import summarize_metrics
from rsna_knee.models.multiseries import create_multiseries_model
from rsna_knee.training.loss import masked_bce_with_logits

device = torch.device("cuda")
folds = CODE / "data/folds/folds_v1.csv"
weak = CODE / "data/processed/weak_labels_v1.csv"

train = pd.read_csv(COMP / "train.csv")
train = merge_weak_labels(train, weak)
train = attach_folds(train, folds)
has_any = train[LABEL_COLS].notna().any(axis=1)
train = train.loc[has_any].copy()
print("any-label", len(train), "expert", int(train[LABEL_COLS].notna().all(axis=1).sum()), flush=True)

val_df = train[train["fold"] == 0].reset_index(drop=True)
tr_all = train[train["fold"] != 0].reset_index(drop=True)
tr_df = tr_all[tr_all[LABEL_COLS].notna().all(axis=1)].reset_index(drop=True)


def _cached(df):
    keep = [(CACHE / f"{uid}.npz").exists() for uid in df["StudyInstanceUID"].astype(str)]
    return df.loc[keep].reset_index(drop=True)


tr_df = _cached(tr_df)
val_df = _cached(val_df)
print(f"expert train (not fold0)={len(tr_df)}  full fold0 val={len(val_df)}", flush=True)
assert len(tr_df) > 0 and len(val_df) > 0

tr_loader = DataLoader(CachedStudyDataset(tr_df, CACHE), batch_size=1, shuffle=True, collate_fn=collate_studies)
va_loader = DataLoader(CachedStudyDataset(val_df, CACHE), batch_size=1, shuffle=False, collate_fn=collate_studies)

model = create_multiseries_model(
    "dinov2_vitb14",
    weights_path=str(WEIGHTS),
    freeze_backbone=True,
    pretrained=False,
    dropout=0.15,
)
state = torch.load(INIT_CKPT, map_location="cpu")
sd = state["model"] if isinstance(state, dict) and "model" in state else state
missing, unexpected = model.load_state_dict(sd, strict=False)
print(
    "loaded ckpt missing", len(missing), "unexpected", len(unexpected),
    "ckpt_auc", state.get("macro_auc") if isinstance(state, dict) else None,
    flush=True,
)
if len(missing) > 20:
    raise SystemExit("Checkpoint does not match ViT-B. Attach the frozen-B fold0_best.pt, not an S run.")

model.to(device)
for p in model.encoder.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=0.05)

out = Path("/kaggle/working/expert_ft_b_fold0")
out.mkdir(parents=True, exist_ok=True)
best = -1.0
epochs = 8


def eval_auc(loader):
    model.eval()
    probs_list, y_list, m_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["images"].to(device),
                batch["plane_ids"].to(device),
                batch["fluid"].to(device),
                batch["fat_sup"].to(device),
                batch["series_mask"].to(device),
                batch["slice_mask"].to(device),
            )
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
            y_list.append(batch["labels"].numpy())
            m_list.append(batch["label_mask"].numpy())
    y, p, m = np.concatenate(y_list), np.concatenate(probs_list), np.concatenate(m_list)
    y_metric = y.copy()
    y_metric[m <= 0] = np.nan
    return summarize_metrics(y_metric, p)["macro_auc"], p


base_auc, _ = eval_auc(va_loader)
print(f"before FT full-fold0 val_macro_auc={base_auc}", flush=True)

for epoch in range(epochs):
    model.train()
    losses = []
    for batch in tr_loader:
        opt.zero_grad(set_to_none=True)
        logits = model(
            batch["images"].to(device),
            batch["plane_ids"].to(device),
            batch["fluid"].to(device),
            batch["fat_sup"].to(device),
            batch["series_mask"].to(device),
            batch["slice_mask"].to(device),
        )
        loss = masked_bce_with_logits(
            logits,
            batch["labels"].to(device),
            batch["label_mask"].to(device),
            batch["label_confidence"].to(device),
            pos_weight=1.0,
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    score, p = eval_auc(va_loader)
    print(f"epoch {epoch}: loss={np.mean(losses):.4f} val_macro_auc={score}", flush=True)
    if np.isfinite(score) and score > best:
        best = score
        torch.save(
            {"model": model.state_dict(), "fold": 0, "macro_auc": score, "epoch": epoch},
            out / "fold0_best.pt",
        )
        np.save(out / "fold0_oof_probs.npy", p)

print(f"best macro_auc={best} (start {base_auc}) → {out}", flush=True)
print("keep if best > before-FT; else kill FT and keep the original B ckpt", flush=True)
