# Kaggle GPU T4x2 + Internet ON — consensus labels (fill unlabeled cells only)
# Public model: Qwen/Qwen2.5-7B-Instruct (Apache-2.0; never used at inference)
#
# Attach: competition + girishbose/rsna-knee-code (must include consensus_labels.py)
# Settings: GPU T4 x2, Internet ON. This notebook uses GPU 0 only.
# Never pip-install torch.
# Save Version / kernel: weak-labels-v5-consensus
#
# Outputs:
#   /kaggle/working/weak_labels_v5_candidate.csv only when the combined gate passes
#   /kaggle/working/weak_labels_v5_raw.csv
#   /kaggle/working/weak_labels_v5_skeleton.csv
#   /kaggle/working/weak_label_v5_audit.csv
#   /kaggle/working/weak_label_v5_summary.json

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "src/rsna_knee/text/consensus_labels.py").exists():
    hits = list(INPUT.rglob("consensus_labels.py"))
    if not hits:
        raise FileNotFoundError(
            "Attach a rsna-knee-code version that contains src/rsna_knee/text/consensus_labels.py"
        )
    CODE = hits[0].parents[3]
sys.path.insert(0, str(CODE / "src"))

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.consensus_labels import (
    NEG_CONFIDENCE,
    POS_CONFIDENCE,
    apply_expert_override,
    audit_source,
    combine_skeleton_and_fills,
    count_known,
    evaluate_gate,
    fills_from_records,
    known_context,
    llm_fill_only,
    parse_named_findings,
    pending_labels,
    skeleton_without_expert_leak,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LICENSE = "Apache-2.0"
MAX_REPORT_CHARS = 5000
BATCH_SIZE = 4
CHECKPOINT_EVERY = 50

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WORK = Path("/kaggle/working")
PROGRESS_PATH = WORK / "weak_labels_v5_progress.jsonl"
RAW_PATH = WORK / "weak_labels_v5_raw.csv"
SKELETON_PATH = WORK / "weak_labels_v5_skeleton.csv"
AUDIT_PATH = WORK / "weak_label_v5_audit.csv"
SUMMARY_PATH = WORK / "weak_label_v5_summary.json"
CANDIDATE_PATH = WORK / "weak_labels_v5_candidate.csv"

SYSTEM_PROMPT = """You are a conservative multilingual musculoskeletal radiology report annotator.
Fill only the requested unlabeled findings. Do not change already-known labels.

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

Return JSON only:
{"findings": {"<exact label>": {"value": 1, 0, or null, "confidence": 0.0-1.0}, ...}}
Include every requested label as a key, using the exact label strings. Extra keys are ignored.
Confidence is confidence that the chosen 1 or 0 is explicitly supported. Use 0 for null."""


def locate_competition() -> Path:
    if (COMP / "train.csv").exists():
        return COMP
    hits = list(Path("/kaggle/input").rglob("train.csv"))
    matches = [p.parent for p in hits if "rsna-knee" in str(p).lower()]
    if not matches:
        raise FileNotFoundError("Could not locate competition train.csv")
    return matches[0]


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


def clean_report(value: object) -> str:
    text = " ".join(str(value if pd.notna(value) else "").split())
    if len(text) <= MAX_REPORT_CHARS:
        return text
    return text[-MAX_REPORT_CHARS:]


def make_messages(report: str, pending: list[str], known: dict[str, int]) -> list[dict[str, str]]:
    user = (
        "Already-known labels (read-only, do not repeat or change):\n"
        f"{json.dumps(known, ensure_ascii=False)}\n\n"
        "Requested unlabeled labels:\n"
        f"{json.dumps(pending, ensure_ascii=False)}\n\n"
        f"REPORT:\n{report}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


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
    messages_batch: list[list[dict[str, str]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
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
            max_new_tokens=512,
            repetition_penalty=1.02,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[:, inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


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
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
if train["StudyInstanceUID"].duplicated().any():
    raise ValueError("StudyInstanceUID must be unique")
print("studies", len(train), "expert", int(train[LABEL_COLS].notna().all(axis=1).sum()), flush=True)

v1 = load_v1()
skeleton = skeleton_without_expert_leak(v1, train)
skeleton.to_csv(SKELETON_PATH, index=False)
print(
    "skeleton known",
    count_known(skeleton),
    "studies with pending",
    int(skeleton[LABEL_COLS].isna().any(axis=1).sum()),
    flush=True,
)

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
reports = train.set_index("StudyInstanceUID")["Report"]
pending_rows = []
for _, row in skeleton.iterrows():
    uid = str(row["StudyInstanceUID"])
    pending = pending_labels(row)
    if not pending or uid in completed:
        continue
    pending_rows.append(
        {
            "StudyInstanceUID": uid,
            "Report": reports.get(uid, ""),
            "pending": pending,
            "known": known_context(row),
        }
    )
print("pending studies", len(pending_rows), flush=True)

for start in range(0, len(pending_rows), BATCH_SIZE):
    batch = pending_rows[start : start + BATCH_SIZE]
    messages_batch = [
        make_messages(clean_report(item["Report"]), item["pending"], item["known"])
        for item in batch
    ]
    responses = infer_batch(messages_batch, tokenizer, model)
    new_rows = []
    for item, response in zip(batch, responses):
        uid = item["StudyInstanceUID"]
        pending = item["pending"]
        try:
            parsed = parse_named_findings(response, pending)
            findings = {
                label: {"value": parsed[label][0], "confidence": parsed[label][1]}
                for label in pending
            }
            row = {
                "StudyInstanceUID": uid,
                "parse_ok": True,
                "pending": pending,
                "findings": findings,
                "raw_response": response,
            }
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            row = {
                "StudyInstanceUID": uid,
                "parse_ok": False,
                "pending": pending,
                "findings": {},
                "raw_response": response,
                "error": str(exc),
            }
        completed[uid] = row
        new_rows.append(row)
    append_progress(new_rows)

    done = start + len(batch)
    if done % CHECKPOINT_EVERY < BATCH_SIZE or done == len(pending_rows):
        parse_rate = float(np.mean([bool(row.get("parse_ok")) for row in completed.values()])) if completed else 0.0
        print(f"processed {done}/{len(pending_rows)} parse_rate={parse_rate:.4f}", flush=True)

raw = fills_from_records(list(completed.values()))
raw.to_csv(RAW_PATH, index=False)
candidate = combine_skeleton_and_fills(skeleton, raw)
fill_only = llm_fill_only(skeleton, candidate)

expert = train.loc[
    train[LABEL_COLS].notna().all(axis=1),
    ["StudyInstanceUID"] + LABEL_COLS,
].copy()
audit = pd.concat(
    [
        audit_source("skeleton", skeleton, expert),
        audit_source("combined", candidate, expert),
        audit_source("llm_fill", fill_only, expert),
    ],
    ignore_index=True,
)
audit.to_csv(AUDIT_PATH, index=False)
print(audit.to_string(index=False), flush=True)

parse_rate = float(raw["__parse_ok"].mean()) if len(raw) else 1.0
summary = evaluate_gate(
    skeleton_audit=audit[audit["source"] == "skeleton"],
    combined_audit=audit[audit["source"] == "combined"],
    fill_audit=audit[audit["source"] == "llm_fill"],
    parse_rate=parse_rate,
    skeleton_known=count_known(skeleton),
    combined_known=count_known(candidate),
)
summary.update(
    {
        "model": MODEL_ID,
        "model_license": MODEL_LICENSE,
        "reports_attempted": int(len(raw)),
        "pos_confidence": POS_CONFIDENCE,
        "neg_confidence": NEG_CONFIDENCE,
        "expert_studies": int(len(expert)),
        "never_overwrite_skeleton": True,
    }
)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

if summary["passed"]:
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
