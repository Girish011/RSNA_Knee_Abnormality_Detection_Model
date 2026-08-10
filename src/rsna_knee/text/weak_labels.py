"""Conservative multilingual report → weak label extraction.

v1: English + Spanish clinical phrases (dataset is multilingual; ES common).
Expert labels always override when applying to train.csv.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from rsna_knee.constants import LABEL_COLS

NEGATION = re.compile(
    r"\b("
    r"no|without|denies|negative\s+for|absence\s+of|unremarkable|failed\s+to\s+demonstrate|"
    r"sin|ausencia\s+de|negativo\s+para|no\s+se\s+(observa|identifica|evidencia)|descarta"
    r")\b",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b("
    r"possible|possibly|may|might|suggestive|cannot\s+exclude|equivocal|"
    r"posible|probablemente|sugiere|no\s+se\s+puede\s+excluir|dudoso"
    r")\b",
    re.I,
)
CHRONIC = re.compile(
    r"\b("
    r"chronic|remote|old|mucoid\s+degeneration|degenerative|"
    r"cr[oó]nic[oa]|antigua|degenerativ[oa]|degeneraci[oó]n\s+mucoide"
    r")\b",
    re.I,
)

# Positive finding patterns (EN + ES). Keep conservative; prefer precision.
PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "ACL": [
        re.compile(
            r"\b(acl|anterior\s+cruciate(\s+ligament)?)\b.{0,50}\b(tear|torn|rupture|discontinuity)\b",
            re.I,
        ),
        re.compile(
            r"\b(tear|torn|rupture|discontinuity)\b.{0,50}\b(acl|anterior\s+cruciate)\b",
            re.I,
        ),
        re.compile(r"\b(rotura|ruptura|desgarro)\b.{0,40}\b(lca|ligamento\s+cruzado\s+anterior)\b", re.I),
        re.compile(r"\b(lca|ligamento\s+cruzado\s+anterior)\b.{0,40}\b(rotura|ruptura|desgarro)\b", re.I),
    ],
    "MCL": [
        re.compile(
            r"\b(mcl|medial\s+collateral(\s+ligament)?)\b.{0,50}\b(tear|torn|rupture|sprain)\b",
            re.I,
        ),
        re.compile(
            r"\b(tear|torn|rupture|sprain)\b.{0,50}\b(mcl|medial\s+collateral)\b",
            re.I,
        ),
        re.compile(
            r"\b(rotura|ruptura|desgarro)\b.{0,40}\b(lcm|ligamento\s+colateral\s+medial)\b",
            re.I,
        ),
        re.compile(
            r"\b(lcm|ligamento\s+colateral\s+medial)\b.{0,40}\b(rotura|ruptura|desgarro)\b",
            re.I,
        ),
    ],
    "Medial Meniscus": [
        re.compile(r"\bmedial\s+meniscus\b.{0,50}\b(tear|torn|truncated)\b", re.I),
        re.compile(r"\b(tear|torn|truncated)\b.{0,50}\bmedial\s+meniscus\b", re.I),
        re.compile(
            r"\b(rotura|ruptura|desgarro)\b.{0,40}\b(menisco\s+interno|menisco\s+medial)\b",
            re.I,
        ),
        re.compile(
            r"\b(menisco\s+interno|menisco\s+medial)\b.{0,40}\b(rotura|ruptura|desgarro)\b",
            re.I,
        ),
    ],
    "Lateral Meniscus": [
        re.compile(r"\blateral\s+meniscus\b.{0,50}\b(tear|torn|truncated)\b", re.I),
        re.compile(r"\b(tear|torn|truncated)\b.{0,50}\blateral\s+meniscus\b", re.I),
        re.compile(
            r"\b(rotura|ruptura|desgarro)\b.{0,40}\b(menisco\s+externo|menisco\s+lateral)\b",
            re.I,
        ),
        re.compile(
            r"\b(menisco\s+externo|menisco\s+lateral)\b.{0,40}\b(rotura|ruptura|desgarro)\b",
            re.I,
        ),
    ],
    "Medial OA": [
        re.compile(
            r"\bmedial\b.{0,40}\b(oa|osteoarthritis|cartilage\s+loss|chondral|chondrosis)\b",
            re.I,
        ),
        re.compile(
            r"\b(artrosis|osteoartritis|condropat[ií]a)\b.{0,40}\b(medial|femorotibial\s+medial|compartimento\s+medial)\b",
            re.I,
        ),
        re.compile(
            r"\b(medial|femorotibial\s+medial|compartimento\s+medial)\b.{0,40}\b(artrosis|osteoartritis|condropat[ií]a)\b",
            re.I,
        ),
    ],
    "Lateral OA": [
        re.compile(
            r"\blateral\b.{0,40}\b(oa|osteoarthritis|cartilage\s+loss|chondral|chondrosis)\b",
            re.I,
        ),
        re.compile(
            r"\b(artrosis|osteoartritis|condropat[ií]a)\b.{0,40}\b(lateral|femorotibial\s+lateral|compartimento\s+lateral)\b",
            re.I,
        ),
        re.compile(
            r"\b(lateral|femorotibial\s+lateral|compartimento\s+lateral)\b.{0,40}\b(artrosis|osteoartritis|condropat[ií]a)\b",
            re.I,
        ),
    ],
    "PF OA": [
        re.compile(
            r"\b(patellofemoral|pf)\b.{0,40}\b(oa|osteoarthritis|cartilage\s+loss|chondral|chondrosis)\b",
            re.I,
        ),
        re.compile(
            r"\b(artrosis|osteoartritis|condropat[ií]a)\b.{0,40}\b(patelofemoral|femoropatelar)\b",
            re.I,
        ),
        re.compile(
            r"\b(patelofemoral|femoropatelar)\b.{0,40}\b(artrosis|osteoartritis|condropat[ií]a)\b",
            re.I,
        ),
    ],
    "Effusion": [
        re.compile(r"\b(joint\s+)?effusion\b", re.I),
        re.compile(r"\b(derrame(\s+articular)?|hidrartrosis)\b", re.I),
    ],
    "Synovitis": [
        re.compile(r"\bsynovitis\b", re.I),
        re.compile(r"\bsinovitis\b", re.I),
    ],
    "Baker's": [
        re.compile(r"\b(baker'?s?\s+cyst|popliteal\s+cyst)\b", re.I),
        re.compile(r"\b(quiste\s+de\s+baker|quiste\s+popl[ií]teo)\b", re.I),
    ],
    "Contusion": [
        re.compile(r"\b(bone\s+)?(contusion|bruise|marrow\s+edema)\b", re.I),
        re.compile(r"\b(contusi[oó]n(es)?\s+[oó]seas?|edema\s+[oó]seo|edema\s+medular)\b", re.I),
    ],
    "Fracture": [
        re.compile(r"\b(fracture|cortical\s+break)\b", re.I),
        re.compile(r"\b(fractura|trazo\s+de\s+fractura)\b", re.I),
    ],
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
