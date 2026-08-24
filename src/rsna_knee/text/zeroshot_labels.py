"""Zero-shot NLI hypotheses for RSNA knee report labels (train-only).

The teacher is noisy: labels come from multilingual reports (FR >> EN/ES, plus
DE/PT/NL) while the exam is graded by radiologists on the images. To make the
zero-shot signal steadier we ensemble several paraphrased hypotheses per label
and aggregate their entailment scores before thresholding. Never used at test time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rsna_knee.constants import LABEL_COLS

if TYPE_CHECKING:
    import pandas as pd

# English hypotheses; mDeBERTa-XNLI is multilingual so FR/ES/DE reports still work.
HYPOTHESES: dict[str, str] = {
    "ACL": "an ACL or anterior cruciate ligament tear or rupture is present",
    "MCL": "an MCL or medial collateral ligament tear or sprain is present",
    "Medial Meniscus": "a medial meniscus tear is present",
    "Lateral Meniscus": "a lateral meniscus tear is present",
    "Medial OA": "medial compartment osteoarthritis or cartilage loss is present",
    "Lateral OA": "lateral compartment osteoarthritis or cartilage loss is present",
    "PF OA": "patellofemoral osteoarthritis or cartilage loss is present",
    "Effusion": "a knee joint effusion is present",
    "Synovitis": "synovitis is present",
    "Baker's": "a Baker cyst or popliteal cyst is present",
    "Contusion": "a bone contusion or bone bruise is present",
    "Fracture": "a fracture is present",
}

assert list(HYPOTHESES) == LABEL_COLS

# Multiple paraphrases per label. Ensembling several entailment probes is more
# robust to phrasing/translation quirks than a single hypothesis. The first entry
# mirrors HYPOTHESES so single-hypothesis behaviour stays reproducible.
HYPOTHESIS_VARIANTS: dict[str, list[str]] = {
    "ACL": [
        HYPOTHESES["ACL"],
        "the anterior cruciate ligament is torn or ruptured",
        "there is a rupture of the ACL",
    ],
    "MCL": [
        HYPOTHESES["MCL"],
        "the medial collateral ligament is torn or sprained",
        "there is an injury of the MCL",
    ],
    "Medial Meniscus": [
        HYPOTHESES["Medial Meniscus"],
        "the medial meniscus is torn",
        "there is a tear of the internal meniscus",
    ],
    "Lateral Meniscus": [
        HYPOTHESES["Lateral Meniscus"],
        "the lateral meniscus is torn",
        "there is a tear of the external meniscus",
    ],
    "Medial OA": [
        HYPOTHESES["Medial OA"],
        "there is cartilage loss in the medial compartment",
        "medial femorotibial osteoarthritis is present",
    ],
    "Lateral OA": [
        HYPOTHESES["Lateral OA"],
        "there is cartilage loss in the lateral compartment",
        "lateral femorotibial osteoarthritis is present",
    ],
    "PF OA": [
        HYPOTHESES["PF OA"],
        "there is patellofemoral cartilage loss",
        "retropatellar osteoarthritis is present",
    ],
    "Effusion": [
        HYPOTHESES["Effusion"],
        "there is a joint effusion in the knee",
        "fluid is present within the knee joint",
    ],
    "Synovitis": [
        HYPOTHESES["Synovitis"],
        "there is synovial inflammation",
        "synovial thickening is present",
    ],
    "Baker's": [
        HYPOTHESES["Baker's"],
        "there is a popliteal cyst",
        "a Baker cyst is seen behind the knee",
    ],
    "Contusion": [
        HYPOTHESES["Contusion"],
        "there is bone marrow edema",
        "a bone bruise is present",
    ],
    "Fracture": [
        HYPOTHESES["Fracture"],
        "there is a bone fracture",
        "a cortical break is present",
    ],
}

assert list(HYPOTHESIS_VARIANTS) == LABEL_COLS

# Conservative: abstain in the middle (precision over coverage).
POS_THRESHOLD = 0.70
NEG_THRESHOLD = 0.30


def impression_text(report: str, max_chars: int = 1800) -> str:
    """Prefer the end of the report (impression / conclusion)."""
    t = " ".join(str(report or "").split())
    if len(t) <= max_chars:
        return t
    return t[-max_chars:]


def aggregate_variant_scores(scores: list[float], *, method: str = "mean") -> float:
    """Combine entailment scores from several hypotheses for one label."""
    if not scores:
        return 0.0
    vals = [float(s) for s in scores]
    if method == "mean":
        return sum(vals) / len(vals)
    if method == "max":
        return max(vals)
    if method == "min":
        return min(vals)
    raise ValueError(f"unknown aggregate method: {method!r} (expected mean|max|min)")


def scores_to_label(score: float, *, pos_th: float = POS_THRESHOLD, neg_th: float = NEG_THRESHOLD):
    """Return (value or None, confidence). None = abstain."""
    s = float(score)
    if s >= pos_th:
        return 1.0, s
    if s <= neg_th:
        return 0.0, 1.0 - s
    return None, s


def merge_pseudo_labels(
    zs: pd.DataFrame,
    *,
    v1: pd.DataFrame | None = None,
    expert: pd.DataFrame | None = None,
    min_v1_conf: float = 0.5,
):
    """Combine zero-shot pseudo labels with a v1 keyword fallback + expert override.

    Priority per (study, label): expert gold > zero-shot NLI decision > v1 keyword
    (only when the NLI model abstained, i.e. the label is NaN). This is the exact
    layering used to build ``weak_labels_v3.csv``, extracted here so it is unit
    tested and reused by the Kaggle notebook instead of being re-implemented inline.

    All frames are keyed by ``StudyInstanceUID`` and use the 12 ``LABEL_COLS`` plus
    optional ``<label>__conf`` columns. Returns a new frame; inputs are untouched.
    """
    import pandas as pd

    out = zs.copy()
    for c in LABEL_COLS:
        if c not in out.columns:
            out[c] = pd.NA
        conf_col = f"{c}__conf"
        if conf_col not in out.columns:
            out[conf_col] = pd.NA

    if v1 is not None and len(v1):
        vi = v1.set_index("StudyInstanceUID")
        for i, row in out.iterrows():
            uid = str(row["StudyInstanceUID"])
            if uid not in vi.index:
                continue
            vr = vi.loc[uid]
            for c in LABEL_COLS:
                if pd.notna(out.at[i, c]):
                    continue  # NLI already decided this label
                if c not in vr.index or pd.isna(vr[c]):
                    continue
                conf_col = f"{c}__conf"
                conf = float(vr[conf_col]) if conf_col in vr.index and pd.notna(vr[conf_col]) else 0.0
                if conf >= min_v1_conf:
                    out.at[i, c] = vr[c]
                    out.at[i, conf_col] = conf

    if expert is not None and len(expert):
        ei = expert.set_index("StudyInstanceUID")
        for i, row in out.iterrows():
            uid = str(row["StudyInstanceUID"])
            if uid not in ei.index:
                continue
            er = ei.loc[uid]
            if not er[LABEL_COLS].notna().all():
                continue  # only fully-labeled expert studies override
            for c in LABEL_COLS:
                out.at[i, c] = er[c]
                out.at[i, f"{c}__conf"] = 1.0

    return out
