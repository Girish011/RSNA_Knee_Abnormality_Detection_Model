"""Multilingual report → weak label extraction (v2).

Language mix in train reports (approx): FR >> EN/ES, plus DE/PT/NL/other.
v2 adds French/German/Portuguese/Dutch phrases on top of EN+ES.
Conservative: prefer precision; chronic/remote → negative for acute labels.
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
    r"sin|ausencia\s+de|negativo\s+para|no\s+se\s+(observa|identifica|evidencia)|descarta|"
    r"pas\s+de|sans|absence\s+d|n[eé]gatif|aucun|"
    r"kein|keine|ohne|unauff[aä]llig|"
    r"geen|zonder|"
    r"sem|aus[eê]ncia\s+de"
    r")\b",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b("
    r"possible|possibly|may|might|suggestive|cannot\s+exclude|equivocal|"
    r"posible|probablemente|sugiere|dudoso|"
    r"possible|probablement|evocateur|ne\s+permet\s+pas\s+d.exclure|"
    r"m[oö]glicherweise|verdacht|nicht\s+auszuschlie"
    r")\b",
    re.I,
)
CHRONIC = re.compile(
    r"\b("
    r"chronic|remote|old|mucoid\s+degeneration|degenerative|"
    r"cr[oó]nic[oa]|antigua|degenerativ|"
    r"chronique|d[eé]g[eé]n[eé]rativ|"
    r"chronisch|degenerativ|"
    r"chronisch|degeneratief"
    r")\b",
    re.I,
)


def _pats(*exprs: str) -> list[re.Pattern[str]]:
    return [re.compile(e, re.I) for e in exprs]


# Positive patterns: EN, ES, FR, DE, PT, NL. Prefer precision.
PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "ACL": _pats(
        r"\b(acl|anterior\s+cruciate)\b.{0,50}\b(tear|torn|rupture|discontinuity)\b",
        r"\b(tear|torn|rupture|discontinuity)\b.{0,50}\b(acl|anterior\s+cruciate)\b",
        r"\b(rotura|ruptura|desgarro)\b.{0,40}\b(lca|ligamento\s+cruzado\s+anterior)\b",
        r"\b(lca|ligamento\s+cruzado\s+anterior)\b.{0,40}\b(rotura|ruptura|desgarro)\b",
        r"\b(rupture|d[eé]chirure)\b.{0,40}\b(lca|ligament\s+crois[eé]\s+ant[eé]rieur)\b",
        r"\b(lca|ligament\s+crois[eé]\s+ant[eé]rieur)\b.{0,40}\b(rupture|d[eé]chirure)\b",
        r"\b(riss|ruptur)\b.{0,40}\b(vkb|vorderes?\s+kreuzband)\b",
        r"\b(vkb|vorderes?\s+kreuzband)\b.{0,40}\b(riss|ruptur)\b",
        r"\b(scheur|ruptuur)\b.{0,40}\b(voorste\s+kruisband|vkb)\b",
        r"\b(voorste\s+kruisband|vkb)\b.{0,40}\b(scheur|ruptuur)\b",
    ),
    "MCL": _pats(
        r"\b(mcl|medial\s+collateral)\b.{0,50}\b(tear|torn|rupture|sprain)\b",
        r"\b(tear|torn|rupture|sprain)\b.{0,50}\b(mcl|medial\s+collateral)\b",
        r"\b(rotura|ruptura|desgarro)\b.{0,40}\b(lcm|ligamento\s+colateral\s+medial)\b",
        r"\b(lcm|ligamento\s+colateral\s+medial)\b.{0,40}\b(rotura|ruptura|desgarro)\b",
        r"\b(rupture|d[eé]chirure)\b.{0,40}\b(ll[ií]|ligament\s+lat[eé]ral\s+interne|ligament\s+collat[eé]ral\s+m[eé]dial)\b",
        r"\b(ll[ií]|ligament\s+lat[eé]ral\s+interne|ligament\s+collat[eé]ral\s+m[eé]dial)\b.{0,40}\b(rupture|d[eé]chirure)\b",
        r"\b(riss|ruptur)\b.{0,40}\b(innenband|mediales?\s+kollateralband)\b",
        r"\b(innenband|mediales?\s+kollateralband)\b.{0,40}\b(riss|ruptur)\b",
    ),
    "Medial Meniscus": _pats(
        r"\bmedial\s+meniscus\b.{0,50}\b(tear|torn|truncated)\b",
        r"\b(tear|torn|truncated)\b.{0,50}\bmedial\s+meniscus\b",
        r"\b(rotura|ruptura)\b.{0,40}\b(menisco\s+interno|menisco\s+medial)\b",
        r"\b(menisco\s+interno|menisco\s+medial)\b.{0,40}\b(rotura|ruptura)\b",
        r"\b(fissure|rupture|l[eé]sion)\b.{0,40}\b(m[eé]nisque\s+(m[eé]dial|interne))\b",
        r"\b(m[eé]nisque\s+(m[eé]dial|interne))\b.{0,40}\b(fissure|rupture|l[eé]sion)\b",
        r"\b(riss|l[aä]sion)\b.{0,40}\b(innenmeniskus|medialer?\s+meniskus)\b",
        r"\b(innenmeniskus|medialer?\s+meniskus)\b.{0,40}\b(riss|l[aä]sion)\b",
        r"\b(meniscusscheur|scheur).{0,30}mediaal\b",
        r"\bmediaal.{0,30}(meniscusscheur|meniscus)\b",
    ),
    "Lateral Meniscus": _pats(
        r"\blateral\s+meniscus\b.{0,50}\b(tear|torn|truncated)\b",
        r"\b(tear|torn|truncated)\b.{0,50}\blateral\s+meniscus\b",
        r"\b(rotura|ruptura)\b.{0,40}\b(menisco\s+externo|menisco\s+lateral)\b",
        r"\b(menisco\s+externo|menisco\s+lateral)\b.{0,40}\b(rotura|ruptura)\b",
        r"\b(fissure|rupture|l[eé]sion)\b.{0,40}\b(m[eé]nisque\s+(lat[eé]ral|externe))\b",
        r"\b(m[eé]nisque\s+(lat[eé]ral|externe))\b.{0,40}\b(fissure|rupture|l[eé]sion)\b",
        r"\b(riss|l[aä]sion)\b.{0,40}\b(au[sß]enmeniskus|lateraler?\s+meniskus)\b",
        r"\b(au[sß]enmeniskus|lateraler?\s+meniskus)\b.{0,40}\b(riss|l[aä]sion)\b",
    ),
    "Medial OA": _pats(
        r"\bmedial\b.{0,40}\b(oa|osteoarthritis|cartilage\s+loss|chondral|chondrosis)\b",
        r"\b(artrosis|osteoartritis|condropat)\b.{0,40}\b(medial|femorotibial\s+medial)\b",
        r"\b(arthrose|chondropathie)\b.{0,40}\b(m[eé]dial|f[eé]moro[\s-]?tibial\s+m[eé]dial|compartiment\s+m[eé]dial)\b",
        r"\b(m[eé]dial|compartiment\s+m[eé]dial)\b.{0,40}\b(arthrose|chondropathie)\b",
        r"\b(arthrose|chondropathie)\b.{0,40}\b(medial|innen)\b",
    ),
    "Lateral OA": _pats(
        r"\blateral\b.{0,40}\b(oa|osteoarthritis|cartilage\s+loss|chondral|chondrosis)\b",
        r"\b(artrosis|osteoartritis|condropat)\b.{0,40}\b(lateral|femorotibial\s+lateral)\b",
        r"\b(arthrose|chondropathie)\b.{0,40}\b(lat[eé]ral|f[eé]moro[\s-]?tibial\s+lat[eé]ral|compartiment\s+lat[eé]ral)\b",
        r"\b(lat[eé]ral|compartiment\s+lat[eé]ral)\b.{0,40}\b(arthrose|chondropathie)\b",
    ),
    "PF OA": _pats(
        r"\b(patellofemoral|pf)\b.{0,40}\b(oa|osteoarthritis|cartilage\s+loss|chondral|chondrosis)\b",
        r"\b(artrosis|condropat)\b.{0,40}\b(patelofemoral|femoropatelar)\b",
        r"\b(arthrose|chondropathie)\b.{0,40}\b(f[eé]moro[\s-]?patellaire|patello[\s-]?f[eé]morale)\b",
        r"\b(f[eé]moro[\s-]?patellaire|patello[\s-]?f[eé]morale)\b.{0,40}\b(arthrose|chondropathie)\b",
        r"\b(retropatellar|femoropatellar)\b.{0,40}\b(arthrose|chondropathie)\b",
    ),
    "Effusion": _pats(
        r"\b(joint\s+)?effusion\b",
        r"\b(derrame(\s+articular)?|hidrartrosis)\b",
        r"\b([eé]panchement(\s+articulaire)?)\b",
        r"\b(gelenkerguss|erguss)\b",
        r"\b(gewrichtseffusie|hydrops)\b",
    ),
    "Synovitis": _pats(
        r"\bsynovitis\b",
        r"\bsinovitis\b",
        r"\bsynovite\b",
        r"\bsynovitis\b",
    ),
    "Baker's": _pats(
        r"\b(baker'?s?\s+cyst|popliteal\s+cyst)\b",
        r"\b(quiste\s+de\s+baker|quiste\s+popl[ií]teo)\b",
        r"\b(kyste\s+(de\s+)?baker|kyste\s+poplit[eé])\b",
        r"\b(bakerzyste|poplitealzyste)\b",
        r"\b(baker[\s-]?cyste)\b",
    ),
    "Contusion": _pats(
        r"\b(bone\s+)?(contusion|bruise|marrow\s+edema)\b",
        r"\b(contusi[oó]n(es)?\s+[oó]seas?|edema\s+[oó]seo|edema\s+medular)\b",
        r"\b(contusion\s+osseuse|[oœ]d[eè]me\s+m[eé]dullaire|[oœ]d[eè]me\s+osseux)\b",
        r"\b(knochenmark[oö]dem|knochenkontusion|bone\s+bruise)\b",
        r"\b(beenmergoedeem|contusie)\b",
    ),
    "Fracture": _pats(
        r"\b(fracture|cortical\s+break)\b",
        r"\b(fractura|trazo\s+de\s+fractura)\b",
        r"\b(fracture)\b",
        r"\b(fraktur|knochenbruch)\b",
        r"\b(fractuur)\b",
    ),
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
