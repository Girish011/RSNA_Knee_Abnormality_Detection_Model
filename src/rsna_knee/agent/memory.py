"""Retrospective memory and the hard ban list (MLEvolve §3.3, adapted to this campaign).

The static half is ``agent/memory/cold_start.md``: settled kills from ``docs/DECISIONS.md``
plus the public density results. The dynamic half is every finished journal node. Retrieval
fuses a lexical rank and a TF-IDF cosine rank with Reciprocal Rank Fusion (Eq. 11); FAISS is
unnecessary at a few hundred records.

The ban list is checked on a node's plan + genome *before* any GPU is rented, so the planner
cannot spend money re-running experiments this repo already killed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rsna_knee.agent.journal import Journal

_TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Ban:
    rule: str
    pattern: str
    why: str


BANS: list[Ban] = [
    Ban("reports_at_test", r"\b(report|text|nli|llm)\w*\b.*\b(infer|test[- ]time|submit)\w*",
        "test.csv has no reports; text models are train-only"),
    Ban("gated_pretrain", r"\b(mrnet|oai|fastmri)\b",
        "registration-gated datasets are not ruled public/free by the hosts"),
    Ban("unfreeze_noisy_teacher", r"\bunfreez\w*",
        "paired collapse -0.114 gold on 2026-09-21; only with a new clean teacher"),
    Ban("train_longer", r"\b(train longer|more epochs|epochs?\s*(>|to)\s*\d{2})",
        "gold peaked at epoch 4 while weak-val kept rising (gf_v0c)"),
    Ban("thin_cache", r"\b(3\s*x\s*12|cache_gf_v0|thin cache)\b",
        "thin 3x12 corpus is retired as a baseline"),
    Ban("backbone_zoo", r"\b(zoo|[4-9]\+?\s*backbones)\b",
        "three backbones bought ~0.001 LB publicly; geometry is the lever"),
    Ban("gold58_objective", r"\b(optimi[sz]e|maximi[sz]e|select)\w*\s+(on\s+)?(gold|58)\b",
        "gold-58 sd 0.021; it is a veto, not the search objective"),
]


def check_bans(text: str) -> list[Ban]:
    low = text.lower()
    return [b for b in BANS if re.search(b.pattern, low)]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def load_cold_start(path: Path) -> list[str]:
    """Split the markdown knowledge base into one record per bullet."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("- "):
            records.append(s[2:])
    return records


def journal_records(journal: Journal) -> list[str]:
    out = []
    for n in journal.nodes.values():
        if n.id == "root" or n.status == "pending":
            continue
        outcome = f"proxy={n.proxy_auc}" if n.valid else f"FAILED {n.error[:200]}"
        out.append(f"[{n.kind}/{n.operator}] {n.plan} | genome={n.genome} | {outcome}")
    return out


def retrieve(query: str, records: list[str], k: int = 5, alpha: float = 0.5, rrf_k: int = 60) -> list[str]:
    """Hybrid lexical + TF-IDF retrieval fused by Reciprocal Rank Fusion."""
    if not records:
        return []
    q = _tokens(query)
    lex = sorted(range(len(records)), key=lambda i: -len(q & _tokens(records[i])))
    vec = TfidfVectorizer().fit(records + [query])
    sims = cosine_similarity(vec.transform([query]), vec.transform(records))[0]
    sem = sorted(range(len(records)), key=lambda i: -sims[i])
    r_lex = {i: r for r, i in enumerate(lex)}
    r_sem = {i: r for r, i in enumerate(sem)}
    score = {
        i: alpha / (rrf_k + r_lex[i]) + (1 - alpha) / (rrf_k + r_sem[i]) for i in range(len(records))
    }
    return [records[i] for i in sorted(score, key=lambda i: -score[i])[:k]]
