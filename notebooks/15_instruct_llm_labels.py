# Kaggle GPU T4x2 + Internet ON — precision-first train-only report labels
# Public model: Qwen/Qwen2.5-7B-Instruct (Apache-2.0; never used at inference)
#
# Attach: competition + girishbose/rsna-knee-code (weak_labels_v1.csv)
# Settings: GPU T4 x2, Internet ON. This notebook uses GPU 0 only.
# Never pip-install torch.
# Save Version: weak-labels-v4-qwen-instruct
#
# Outputs:
#   /kaggle/working/weak_labels_v4_candidate.csv only when the precision gate passes
#   /kaggle/working/weak_labels_v4_raw.csv
#   /kaggle/working/weak_label_v4_audit.csv
#   /kaggle/working/weak_label_v4_summary.json

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LICENSE = "Apache-2.0"
LABEL_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Fixed before looking at this model's expert audit.
POS_CONFIDENCE = 0.90
NEG_CONFIDENCE = 0.95
PRECISION_GATE = 0.69
# Historical pre-override audit in docs/audit/weak_label_vs_expert.csv.
# The distributed weak_v1 CSV contains expert overrides, so joining that file
# back to the 58 experts would leak gold labels and falsely report precision 1.
V1_BASELINE_PRECISION = 0.6875534188034188
MIN_POSITIVE_PREDICTIONS = 12
MAX_REPORT_CHARS = 5000
BATCH_SIZE = 4
CHECKPOINT_EVERY = 50

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
WORK = Path("/kaggle/working")
PROGRESS_PATH = WORK / "weak_labels_v4_progress.jsonl"
RAW_PATH = WORK / "weak_labels_v4_raw.csv"
AUDIT_PATH = WORK / "weak_label_v4_audit.csv"
SUMMARY_PATH = WORK / "weak_label_v4_summary.json"
CANDIDATE_PATH = WORK / "weak_labels_v4_candidate.csv"

SYSTEM_PROMPT = """You are a conservative multilingual musculoskeletal radiology report annotator.
Read one knee MRI report in any language. Classify the 12 findings in the exact order supplied.

Use:
- 1 only when the report explicitly supports that finding in the examined knee.
- 0 only when the report explicitly says that finding is absent or intact.
- null for omission, ambiguity, uncertainty, postoperative ambiguity, or wrong anatomy.

Rules:
- Handle negation and uncertainty literally. "Cannot exclude", "possible", and "suspected" are null.
- ACL and MCL require tear, rupture, or sprain; degeneration alone is not a tear.
- Meniscus labels require a tear; degeneration or extrusion alone is not a tear.
- OA labels include compartment-specific osteoarthritis or definite cartilage/chondral loss.
- Do not transfer medial, lateral, or patellofemoral findings between compartments.
- Contusion means traumatic bone contusion/bone bruise, not nonspecific marrow edema.
- Baker's means Baker/popliteal cyst.
- Do not infer a negative finding merely because it is not mentioned.

Return JSON only, with exactly:
{"labels":[12 values of 1, 0, or null],"confidence":[12 numbers from 0 to 1]}
Confidence is confidence that the chosen 1 or 0 is explicitly supported. Use 0 for null."""


def locate_competition() -> Path:
    if (COMP / "train.csv").exists():
        return COMP
    hits = list(Path("/kaggle/input").rglob("train.csv"))
    matches = [p.parent for p in hits if "rsna-knee" in str(p).lower()]
    if not matches:
        raise FileNotFoundError("Could not locate competition train.csv")
    return matches[0]


def clean_report(value: object) -> str:
    text = " ".join(str(value if pd.notna(value) else "").split())
    if len(text) <= MAX_REPORT_CHARS:
        return text
    # Impressions/conclusions are commonly at the end of a report.
    return text[-MAX_REPORT_CHARS:]


def make_messages(report: str) -> list[dict[str, str]]:
    order = " | ".join(f"{i + 1}:{label}" for i, label in enumerate(LABEL_COLS))
    user = f"Label order: {order}\n\nREPORT:\n{report}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_response(text: str) -> tuple[list[object], list[float]]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match is None:
        raise ValueError("no JSON object")
    obj = json.loads(match.group(0))
    labels = obj.get("labels")
    confidence = obj.get("confidence")
    if not isinstance(labels, list) or not isinstance(confidence, list):
        raise ValueError("labels/confidence are not lists")
    if len(labels) != len(LABEL_COLS) or len(confidence) != len(LABEL_COLS):
        raise ValueError("wrong output length")

    parsed_labels: list[object] = []
    parsed_confidence: list[float] = []
    for value, conf in zip(labels, confidence):
        if value not in (0, 1, None):
            raise ValueError(f"invalid label value {value!r}")
        score = float(conf)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid confidence {conf!r}")
        parsed_labels.append(value)
        parsed_confidence.append(score if value is not None else 0.0)
    return parsed_labels, parsed_confidence


def load_progress() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not PROGRESS_PATH.exists():
        return rows
    with PROGRESS_PATH.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[str(row["StudyInstanceUID"])] = row
            except (json.JSONDecodeError, KeyError):
                continue
    print("resuming completed reports", len(rows), flush=True)
    return rows


def append_progress(rows: list[dict[str, object]]) -> None:
    with PROGRESS_PATH.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_batch(
    reports: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(make_messages(report), tokenize=False, add_generation_prompt=True)
        for report in reports
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=180,
            repetition_penalty=1.02,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[:, inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def rows_to_raw(rows: dict[str, dict[str, object]]) -> pd.DataFrame:
    records = []
    for uid, row in rows.items():
        record: dict[str, object] = {
            "StudyInstanceUID": uid,
            "__parse_ok": bool(row.get("parse_ok", False)),
            "__raw_response": row.get("raw_response", ""),
        }
        labels = row.get("labels", [None] * len(LABEL_COLS))
        confidence = row.get("confidence", [0.0] * len(LABEL_COLS))
        for label, value, conf in zip(LABEL_COLS, labels, confidence):
            score = float(conf)
            accepted: object = np.nan
            if value == 1 and score >= POS_CONFIDENCE:
                accepted = 1.0
            elif value == 0 and score >= NEG_CONFIDENCE:
                accepted = 0.0
            record[label] = accepted
            record[f"{label}__conf"] = score
        records.append(record)
    return pd.DataFrame(records)


def load_v1() -> pd.DataFrame:
    candidates = [
        CODE / "data/processed/weak_labels_v1.csv",
        Path("/kaggle/input/rsna-knee-code/data/processed/weak_labels_v1.csv"),
    ]
    for path in candidates:
        if path.exists():
            print("weak_v1", path, flush=True)
            return pd.read_csv(path)
    raise FileNotFoundError("Attach rsna-knee-code containing data/processed/weak_labels_v1.csv")


def combine_v1_and_llm(v1: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Preserve every accepted v1 label; use the LLM only to fill v1 abstentions."""
    llm = raw.set_index("StudyInstanceUID")
    out = v1.copy().set_index("StudyInstanceUID")
    for uid in llm.index:
        if uid not in out.index:
            out.loc[uid, :] = np.nan
        for label in LABEL_COLS:
            if label not in out.columns:
                out[label] = np.nan
            conf_col = f"{label}__conf"
            if conf_col not in out.columns:
                out[conf_col] = np.nan
            if pd.isna(out.at[uid, label]) and pd.notna(llm.at[uid, label]):
                out.at[uid, label] = llm.at[uid, label]
                out.at[uid, conf_col] = llm.at[uid, conf_col]
    return out.reset_index()


def audit_source(name: str, pred: pd.DataFrame, expert: pd.DataFrame) -> list[dict[str, object]]:
    merged = expert.merge(pred, on="StudyInstanceUID", suffixes=("_gold", "_pred"))
    rows = []
    for label in LABEL_COLS:
        gold = merged[f"{label}_gold"].astype(int)
        prediction = merged[f"{label}_pred"]
        known = prediction.notna()
        positive = known & prediction.eq(1)
        if known.any():
            y_known = gold[known].to_numpy()
            p_known = prediction[known].astype(int).to_numpy()
            f1 = float(f1_score(y_known, p_known, zero_division=0))
        else:
            f1 = np.nan
        rows.append(
            {
                "source": name,
                "label": label,
                "n_known": int(known.sum()),
                "n_pred_pos": int(positive.sum()),
                "positive_precision": (
                    float(precision_score(gold[positive], np.ones(int(positive.sum())), zero_division=0))
                    if positive.any()
                    else np.nan
                ),
                "positive_recall_all": float(
                    recall_score(gold.to_numpy(), positive.astype(int).to_numpy(), zero_division=0)
                ),
                "f1_known": f1,
            }
        )
    return rows


def apply_expert_override(candidate: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    out = candidate.set_index("StudyInstanceUID")
    expert = train.loc[train[LABEL_COLS].notna().all(axis=1)].set_index("StudyInstanceUID")
    for uid, row in expert.iterrows():
        if uid not in out.index:
            out.loc[uid, :] = np.nan
        for label in LABEL_COLS:
            out.at[uid, label] = float(row[label])
            out.at[uid, f"{label}__conf"] = 1.0
    keep = LABEL_COLS + [f"{label}__conf" for label in LABEL_COLS]
    return out.reset_index()[["StudyInstanceUID"] + keep]


print(
    "cuda",
    torch.cuda.is_available(),
    "torch",
    torch.__version__,
    "visible GPUs",
    torch.cuda.device_count(),
    flush=True,
)
if not torch.cuda.is_available():
    raise RuntimeError("GPU torch is unavailable. Select T4x2; do not pip-install torch.")

train = pd.read_csv(locate_competition() / "train.csv")
if train["StudyInstanceUID"].astype(str).duplicated().any():
    raise ValueError("StudyInstanceUID must be unique")
print("studies", len(train), "expert", int(train[LABEL_COLS].notna().all(axis=1).sum()), flush=True)

# Kaggle's current image omits bitsandbytes. Install only that wheel and use
# --no-deps so pip cannot replace the working CUDA torch build.
try:
    bnb_version = importlib.metadata.version("bitsandbytes")
    bnb_ok = tuple(int(part) for part in bnb_version.split(".")[:2]) >= (0, 46)
except importlib.metadata.PackageNotFoundError:
    bnb_version, bnb_ok = "missing", False
if not bnb_ok:
    print("installing bitsandbytes only; existing torch will not be modified", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "bitsandbytes>=0.46.1",
        ]
    )
    bnb_version = importlib.metadata.version("bitsandbytes")
print("bitsandbytes", bnb_version, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization,
    dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
model.eval()

completed = load_progress()
pending = train.loc[
    ~train["StudyInstanceUID"].astype(str).isin(completed),
    ["StudyInstanceUID", "Report"],
].reset_index(drop=True)
print("pending", len(pending), flush=True)

for start in range(0, len(pending), BATCH_SIZE):
    batch = pending.iloc[start : start + BATCH_SIZE]
    reports = [clean_report(value) for value in batch["Report"]]
    responses = infer_batch(reports, tokenizer, model)
    new_rows = []
    for uid, response in zip(batch["StudyInstanceUID"].astype(str), responses):
        try:
            labels, confidence = parse_response(response)
            row = {
                "StudyInstanceUID": uid,
                "parse_ok": True,
                "labels": labels,
                "confidence": confidence,
                "raw_response": response,
            }
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            row = {
                "StudyInstanceUID": uid,
                "parse_ok": False,
                "labels": [None] * len(LABEL_COLS),
                "confidence": [0.0] * len(LABEL_COLS),
                "raw_response": response,
                "error": str(exc),
            }
        completed[uid] = row
        new_rows.append(row)
    append_progress(new_rows)

    done = start + len(batch)
    if done % CHECKPOINT_EVERY < BATCH_SIZE or done == len(pending):
        rows_to_raw(completed).to_csv(RAW_PATH, index=False)
        parse_rate = float(np.mean([bool(row.get("parse_ok")) for row in completed.values()]))
        print(f"processed {done}/{len(pending)} parse_rate={parse_rate:.4f}", flush=True)

raw = rows_to_raw(completed)
raw.to_csv(RAW_PATH, index=False)
v1 = load_v1()
candidate = combine_v1_and_llm(v1, raw)

expert = train.loc[
    train[LABEL_COLS].notna().all(axis=1),
    ["StudyInstanceUID"] + LABEL_COLS,
].copy()
audit_rows = []
audit_rows.extend(audit_source("llm_only", raw, expert))
audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT_PATH, index=False)
print(audit.to_string(index=False), flush=True)

llm_audit = audit[audit["source"] == "llm_only"]
valid_precision = llm_audit["positive_precision"].dropna()
macro_precision = float(valid_precision.mean()) if len(valid_precision) else 0.0
positive_predictions = int(llm_audit["n_pred_pos"].sum())
v1_known = int(v1[LABEL_COLS].notna().sum().sum())
candidate_known = int(candidate[LABEL_COLS].notna().sum().sum())
parse_rate = float(raw["__parse_ok"].mean())
passed = (
    macro_precision > max(PRECISION_GATE, V1_BASELINE_PRECISION)
    and candidate_known > v1_known
    and positive_predictions >= MIN_POSITIVE_PREDICTIONS
    and parse_rate >= 0.98
)

summary = {
    "model": MODEL_ID,
    "model_license": MODEL_LICENSE,
    "reports": int(len(raw)),
    "parse_rate": parse_rate,
    "pos_confidence": POS_CONFIDENCE,
    "neg_confidence": NEG_CONFIDENCE,
    "expert_studies": int(len(expert)),
    "weak_v1_historical_macro_positive_precision": V1_BASELINE_PRECISION,
    "llm_macro_positive_precision": macro_precision,
    "weak_v1_total_known_labels": v1_known,
    "candidate_total_known_labels": candidate_known,
    "llm_expert_positive_predictions": positive_predictions,
    "precision_gate_strictly_greater_than": max(PRECISION_GATE, V1_BASELINE_PRECISION),
    "requires_more_total_known_labels_than_v1": True,
    "passed": bool(passed),
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

if passed:
    final = apply_expert_override(candidate, train)
    final.to_csv(CANDIDATE_PATH, index=False)
    print("GATE PASSED — wrote", CANDIDATE_PATH, flush=True)
else:
    CANDIDATE_PATH.unlink(missing_ok=True)
    print("GATE FAILED — no training candidate written", flush=True)

del model
gc.collect()
torch.cuda.empty_cache()
print("done", flush=True)
