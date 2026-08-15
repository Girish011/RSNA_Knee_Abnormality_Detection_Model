"""Zero-shot NLI hypotheses for RSNA knee report labels (train-only)."""

from __future__ import annotations

from rsna_knee.constants import LABEL_COLS

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

# Conservative: abstain in the middle (precision over coverage).
POS_THRESHOLD = 0.70
NEG_THRESHOLD = 0.30


def impression_text(report: str, max_chars: int = 1800) -> str:
    """Prefer the end of the report (impression / conclusion)."""
    t = " ".join(str(report or "").split())
    if len(t) <= max_chars:
        return t
    return t[-max_chars:]


def scores_to_label(score: float, *, pos_th: float = POS_THRESHOLD, neg_th: float = NEG_THRESHOLD):
    """Return (value or None, confidence). None = abstain."""
    s = float(score)
    if s >= pos_th:
        return 1.0, s
    if s <= neg_th:
        return 0.0, 1.0 - s
    return None, s
