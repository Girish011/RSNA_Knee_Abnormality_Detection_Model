"""Conservative multilingual report → weak label extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from rsna_knee.constants import LABEL_COLS

NEGATION = re.compile(
    r"\b(no|without|denies|negative\s+for|absence\s+of|unremarkable|failed\s+to\s+demonstrate)\b",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b(possible|possibly|may|might|suggestive|cannot\s+exclude|equivocal)\b", re.I
)
CHRONIC = re.compile(r"\b(chronic|remote|old|mucoid\s+degeneration|degenerative)\b", re.I)

PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "ACL": [
        re.compile(r"\b(acl|anterior\s+cruciate)\b.{0,40}\b(tear|rupture|discontinuity)\b", re.I)
    ],
    "MCL": [
        re.compile(r"\b(mcl|medial\s+collateral)\b.{0,40}\b(tear|rupture|sprain)\b", re.I)
    ],
    "Medial Meniscus": [
        re.compile(r"\bmedial\s+meniscus\b.{0,40}\b(tear|torn|truncated)\b", re.I),
        re.compile(r"\b(tear|torn)\b.{0,40}\bmedial\s+meniscus\b", re.I),
    ],
    "Lateral Meniscus": [
        re.compile(r"\blateral\s+meniscus\b.{0,40}\b(tear|torn|truncated)\b", re.I),
        re.compile(r"\b(tear|torn)\b.{0,40}\blateral\s+meniscus\b", re.I),
    ],
    "Medial OA": [
        re.compile(r"\bmedial\b.{0,30}\b(oa|osteoarthritis|cartilage\s+loss|chondral)\b", re.I)
    ],
    "Lateral OA": [
        re.compile(r"\blateral\b.{0,30}\b(oa|osteoarthritis|cartilage\s+loss|chondral)\b", re.I)
    ],
    "PF OA": [
        re.compile(
            r"\b(patellofemoral|pf)\b.{0,30}\b(oa|osteoarthritis|cartilage\s+loss)\b", re.I
        )
    ],
    "Effusion": [re.compile(r"\b(joint\s+)?effusion\b", re.I)],
    "Synovitis": [re.compile(r"\bsynovitis\b", re.I)],
    "Baker's": [re.compile(r"\b(baker'?s?\s+cyst|popliteal\s+cyst)\b", re.I)],
    "Contusion": [re.compile(r"\b(bone\s+)?(contusion|bruise|marrow\s+edema)\b", re.I)],
    "Fracture": [re.compile(r"\b(fracture|cortical\s+break)\b", re.I)],
}


@dataclass
class WeakLabelResult:
    label: str
    value: int
    confidence: float
    reason: str


def _sentence_windows(text: str) -> list[str]:
    parts = re.split(r"[.\n;]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_label_from_report(text: str, label: str) -> WeakLabelResult:
    if not text or not str(text).strip():
        return WeakLabelResult(label, 0, 0.0, "empty")
    patterns = PATTERNS.get(label, [])
    best = WeakLabelResult(label, 0, 0.0, "no_match")
    for sent in _sentence_windows(str(text)):
        if not any(p.search(sent) for p in patterns):
            continue
        if NEGATION.search(sent):
            return WeakLabelResult(label, 0, 0.7, "negated")
        if CHRONIC.search(sent) and label in {"ACL", "MCL", "Contusion", "Fracture"}:
            return WeakLabelResult(label, 0, 0.65, "chronic_remote")
        conf = 0.55
        reason = "keyword"
        if UNCERTAIN.search(sent):
            conf = 0.35
            reason = "uncertain"
        if conf > best.confidence:
            best = WeakLabelResult(label, 1, conf, reason)
    return best


def weak_labels_for_report(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for label in LABEL_COLS:
        res = extract_label_from_report(text, label)
        out[label] = float(res.value)
        out[f"{label}__conf"] = float(res.confidence)
        out[f"{label}__reason"] = res.reason
    return out


def apply_weak_labels(
    train_df: pd.DataFrame,
    *,
    min_confidence: float = 0.5,
    expert_override: bool = True,
) -> pd.DataFrame:
    """Fill missing expert labels from reports; never overwrite expert if set."""
    df = train_df.copy()
    for label in LABEL_COLS:
        if label not in df.columns:
            df[label] = pd.NA
        conf_col = f"{label}__conf"
        if conf_col not in df.columns:
            df[conf_col] = 1.0

    for idx, row in df.iterrows():
        report = row.get("Report", "")
        extracted = weak_labels_for_report("" if pd.isna(report) else str(report))
        for label in LABEL_COLS:
            conf_col = f"{label}__conf"
            expert_val = row.get(label)
            has_expert = expert_override and pd.notna(expert_val)
            if has_expert:
                df.at[idx, conf_col] = 1.0
                continue
            conf = float(extracted[f"{label}__conf"])
            if conf < min_confidence:
                df.at[idx, label] = pd.NA
                df.at[idx, conf_col] = conf
            else:
                df.at[idx, label] = extracted[label]
                df.at[idx, conf_col] = conf
    return df
