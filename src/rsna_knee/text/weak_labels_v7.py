"""v7 multilingual weak-label extraction: adds Turkish + Greek to the v2 extractor.

Why: the corpus is 7 languages. Our v2 keyword extractor covers EN/ES/FR/DE/NL/PT
but has **no Turkish or Greek** vocabulary and no normalcy handling — yet Turkish is
the 2nd-largest language (~600 studies) and Greek adds ~320. Real reports in these
languages usually state findings as *normal* ("menisküs normaldir", "Χωρίς συλλογή
υγρού"), so the dominant signal is language-correct negation/normalcy, not just
positive keywords. A left-only negation window (the v2 default) also silently
inverts Turkish, which negates/normalizes *after* the term.

v7 detects the language and, for Turkish/Greek, uses a sentence-level rule:
- anatomy/term present + abnormality cue (and not normalized) -> POSITIVE
- anatomy/term present + normalcy/negation cue -> NEGATIVE
- otherwise abstain
For all other languages it delegates to the validated v2 extractor unchanged.

Vocabulary is best-effort MSK terminology validated against sampled real reports;
it is intentionally high-precision (abstain when unsure). Train-only; never used at
inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.weak_labels import extract_label_from_report as _extract_v2

GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
# Case-SENSITIVE: under re.I the Turkish dotted/dotless i (İ/ı) fold onto ASCII 'i',
# which would misclassify any English text. Match these letters literally instead.
TURKISH_CHARS = re.compile(r"[şğİı]")
# Turkish-specific words only (avoid 'menisk' which collides with German 'Meniskus').
TURKISH_WORDS = re.compile(
    r"\b(diz|normaldir|yoktur|izlen\w*|çapraz|kıkırdak|mayii?|eklem|kemik|dejenerasyon)\b",
    re.I,
)


def detect_language(text: str) -> str:
    """Coarse language tag: 'el' (Greek), 'tr' (Turkish), or 'other'."""
    t = str(text or "")
    if GREEK_RE.search(t):
        return "el"
    if TURKISH_CHARS.search(t) or TURKISH_WORDS.search(t):
        return "tr"
    return "other"


# --- abnormality / normalcy cues per language -------------------------------
TR_ABNORMAL = re.compile(
    r"(yırtık|yirtik|rüptür|ruptur|kopma|yaralan|dejen\w*|kıkırdak\s+kay|kayıp|kayb|"
    r"ödem|odem|kist|kırık|kirik|fraktür|fraktur|artış|artmış|sinovit|efüzyon|hidrops|"
    r"kondromalazi|kondropati|osteoartrit|artroz|lezyon|sinyal\s+artış)",
    re.I,
)
TR_NORMAL = re.compile(
    r"(normaldir|normal(?:\b|\s+sınır)|doğal|dogal|korunmuş|korunmus|yoktur|\byok\b|"
    r"izlenmedi|izlenmemiş|saptanmadı|saptanmamış|görülmedi|gorulmedi|mevcut\s+değil|"
    r"sağlam|saglam|intakt|seçilmedi)",
    re.I,
)
EL_ABNORMAL = re.compile(
    r"(ρήξη|ρωγμ|οίδημα|οιδημα|κύστη|κυστη|κάταγμα|καταγμα|χονδροπάθεια|χονδροπαθεια|"
    r"οστεοαρθρίτιδα|αρθρίτιδα|υμενίτιδα|υμενιτιδα|συλλογή|συλλογη|ύδραρθρο|υδραρθρο|"
    r"βλάβη|βλαβη|εκφύλιση|εκφυλιση|απώλεια|απωλεια)",
    re.I,
)
EL_NORMAL = re.compile(
    r"(χωρίς|χωρις|\bδεν\b|φυσιολογικ|εντός\s+του\s+φυσιολογικού|εντος\s+του\s+φυσιολογικου|"
    r"απουσία|απουσια|ελεύθερ|ελευθερ|ακέραι|ακεραι)",
    re.I,
)
# Borderline / mild / uncertain: the competition grades "on the fence" as NEGATIVE,
# so downgrade these to abstain rather than assert a positive.
TR_UNCERTAIN = re.compile(
    r"(minimal|hafif|az\s+miktarda|silik|şüpheli|supheli|olası|olasi|olabilir|muhtemel|ılımlı|eser|hafifçe|başlangıç)",
    re.I,
)
EL_UNCERTAIN = re.compile(
    r"(ήπι|ηπι|ελαφρ|μικρή|μικρη|μικρό|μικρο|πιθαν|ενδεχομέν|ενδεχομεν|διακριτικ|ελάχιστ|ελαχιστ)",
    re.I,
)

# --- per-label anatomy / self-terms, per language ---------------------------
# "anat" needs an abnormality cue in-sentence to fire positive; "self" terms are
# themselves the finding (their presence, unless normalized, is positive).
TR_LABELS: dict[str, dict[str, str]] = {
    "ACL": {"anat": r"(ön\s+çapraz|on\s+capraz|öçb|oçb|anterior\s+krusiat)"},
    "MCL": {"anat": r"(iç\s+yan\s+bağ|ic\s+yan\s+bag|medial\s+kollateral|iç\s+kollateral)"},
    "Medial Meniscus": {"anat": r"((iç|medial|medyal)\s+menisk)"},
    "Lateral Meniscus": {"anat": r"((dış|dis|lateral)\s+menisk)"},
    "Medial OA": {"anat": r"(medial|medyal|iç\s+kompartman)"},
    "Lateral OA": {"anat": r"(lateral|dış\s+kompartman)"},
    "PF OA": {"anat": r"(patellofemoral|femoropatellar|patella\s+eklem|retropatellar)"},
    "Effusion": {"self": r"(efüzyon|efuzyon|hidrops)", "anat": r"(eklem\s+içi\s+sıvı|mayii?|sıvı|sivi)"},
    "Synovitis": {"self": r"(sinovit)"},
    "Baker's": {"self": r"(baker\s+kist|poplite\w*\s+kist)"},
    "Contusion": {"self": r"(kemik\s+iliği\s+ödem|kemik\s+ödem|kontüzyon|kontuzyon|kemik\s+kontüzyon)"},
    "Fracture": {"self": r"(kırık|kirik|fraktür|fraktur)"},
}
EL_LABELS: dict[str, dict[str, str]] = {
    "ACL": {"anat": r"(πρόσθιος\s+χιαστ|προσθιος\s+χιαστ|πρόσθιο\s+χιαστ|χιαστ)"},
    "MCL": {"anat": r"(έσω\s+πλάγιος|εσω\s+πλαγιος|έσω\s+πλάγιο)"},
    "Medial Meniscus": {"anat": r"(έσω\s+μηνίσκ|εσω\s+μηνισκ)"},
    "Lateral Meniscus": {"anat": r"(έξω\s+μηνίσκ|εξω\s+μηνισκ)"},
    "Medial OA": {"anat": r"(έσω|εσω)"},
    "Lateral OA": {"anat": r"(έξω|εξω)"},
    "PF OA": {"anat": r"(επιγονατιδομηριαί|τροχιλ|επιγονατιδ)"},
    "Effusion": {"self": r"(ύδραρθρο|υδραρθρο)", "anat": r"(συλλογή\s+υγρού|συλλογη\s+υγρου|υγρό|υγρο)"},
    "Synovitis": {"self": r"(υμενίτιδα|υμενιτιδα)"},
    "Baker's": {"self": r"(κύστη\s+baker|κυστη\s+baker|ιγνυακή\s+κύστη|ιγνυακη\s+κυστη)"},
    "Contusion": {"self": r"(οστικό\s+οίδημα|οστικο\s+οιδημα|μυελικό\s+οίδημα)"},
    "Fracture": {"self": r"(κάταγμα|καταγμα)"},
}
# OA needs a cartilage/arthritis cue in the same sentence to fire on a compartment word.
TR_OA_CUE = re.compile(r"(osteoartrit|artroz|kondromalazi|kondropati|kıkırdak\s+kay|dejen\w*)", re.I)
EL_OA_CUE = re.compile(r"(οστεοαρθρίτιδα|αρθρίτιδα|χονδροπάθεια|χονδροπαθεια|απώλεια\s+χόνδρου|εκφύλιση)", re.I)
OA_LABELS = {"Medial OA", "Lateral OA", "PF OA"}


@dataclass
class WeakLabelResult:
    label: str
    value: int
    confidence: float
    reason: str


def _normalize(text: str) -> str:
    """Fix common encoding quirks in these reports (micro-sign used for Greek mu)."""
    return str(text).replace("\u00b5", "\u03bc").replace("\u0384", "").replace("\u00b4", "")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.\n;·]+", _normalize(text))
    return [p.strip() for p in parts if p.strip()]


def _extract_trel(text: str, label: str, lang: str) -> WeakLabelResult:
    labels = TR_LABELS if lang == "tr" else EL_LABELS
    abnormal = TR_ABNORMAL if lang == "tr" else EL_ABNORMAL
    normal = TR_NORMAL if lang == "tr" else EL_NORMAL
    oa_cue = TR_OA_CUE if lang == "tr" else EL_OA_CUE
    uncertain = TR_UNCERTAIN if lang == "tr" else EL_UNCERTAIN
    spec = labels.get(label)
    if not spec:
        return WeakLabelResult(label, 0, 0.0, "no_spec")

    best = WeakLabelResult(label, 0, 0.0, "no_match")
    for sent in _sentences(text):
        anat = spec.get("anat")
        self_t = spec.get("self")
        has_anat = bool(anat and re.search(anat, sent, re.I))
        has_self = bool(self_t and re.search(self_t, sent, re.I))
        if not (has_anat or has_self):
            continue
        is_normal = bool(normal.search(sent))
        is_uncertain = bool(uncertain.search(sent))
        # Self-terms (effusion/cyst/fracture/etc.) are the finding themselves.
        if has_self:
            if is_normal and not abnormal.search(sent):
                if best.confidence < 0.7:
                    best = WeakLabelResult(label, 0, 0.7, f"{lang}_normal")
                continue
            if is_uncertain:  # borderline/mild -> abstain (graded negative by host)
                continue
            return WeakLabelResult(label, 1, 0.6, f"{lang}_self")
        # Anatomy word present: need an abnormality cue to call positive.
        if label in OA_LABELS:
            if oa_cue.search(sent) and not is_normal and not is_uncertain:
                return WeakLabelResult(label, 1, 0.55, f"{lang}_oa")
            if is_normal:
                if best.confidence < 0.65:
                    best = WeakLabelResult(label, 0, 0.65, f"{lang}_normal")
            continue
        if abnormal.search(sent) and not is_normal and not is_uncertain:
            return WeakLabelResult(label, 1, 0.6, f"{lang}_abn")
        if is_normal:
            if best.confidence < 0.7:
                best = WeakLabelResult(label, 0, 0.7, f"{lang}_normal")
    return best


def extract_label_v7(text: str, label: str) -> WeakLabelResult:
    """v7 extractor: Turkish/Greek get dedicated handling; others delegate to v2."""
    if not text or not str(text).strip():
        return WeakLabelResult(label, 0, 0.0, "empty")
    lang = detect_language(text)
    if lang in ("tr", "el"):
        return _extract_trel(text, label, lang)
    r = _extract_v2(text, label)
    return WeakLabelResult(label, r.value, r.confidence, r.reason)


def weak_labels_v7_for_report(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for label in LABEL_COLS:
        r = extract_label_v7(text, label)
        out[label] = float(r.value) if r.confidence > 0 else None
        out[f"{label}__conf"] = float(r.confidence)
        out[f"{label}__reason"] = r.reason
    return out


def apply_weak_labels_v7(
    train_df: pd.DataFrame,
    *,
    min_confidence: float = 0.5,
    expert_override: bool = True,
) -> pd.DataFrame:
    """Fill labels from reports (v7). Expert rows keep their gold when present."""
    df = train_df.copy()
    for label in LABEL_COLS:
        if label not in df.columns:
            df[label] = pd.NA
        conf_col = f"{label}__conf"
        if conf_col not in df.columns:
            df[conf_col] = 1.0
    for idx, row in df.iterrows():
        report = row.get("Report", "")
        extracted = weak_labels_v7_for_report("" if pd.isna(report) else str(report))
        for label in LABEL_COLS:
            conf_col = f"{label}__conf"
            if expert_override and pd.notna(row.get(label)):
                df.at[idx, conf_col] = 1.0
                continue
            val = extracted[label]
            conf = float(extracted[f"{label}__conf"])
            if val is None or conf < min_confidence:
                df.at[idx, label] = pd.NA
                df.at[idx, conf_col] = conf
            else:
                df.at[idx, label] = float(val)
                df.at[idx, conf_col] = conf
    return df
