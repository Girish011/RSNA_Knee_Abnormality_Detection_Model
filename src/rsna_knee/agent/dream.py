"""Dream-RSI replay: score alternatives from recorded outcomes instead of new GPU runs.

Two replay worlds exist in this campaign:

1. **Logit world.** A trained model saves per-window logits ``(n_studies, n_windows, 12)``.
   Window count, pooling and model blend weights are then re-scored for free. This is how
   inference-time geometry (the lever that moved the public 0.926 -> 0.932 on fixed weights)
   and the efficiency final are chosen.
2. **Tree world.** A finished journal is a discovery tree. An exploration policy replays it
   by choosing which recorded nodes to reveal, in which batches, and when to stop
   (Dream-RSI §3). The replay score trades best proxy AUC against GPU dollars.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from rsna_knee.agent.journal import ROOT_ID, Journal, Node
from rsna_knee.metrics import macro_auc

# ---------------------------------------------------------------- logit world


def window_subset(n_windows: int, k: int) -> np.ndarray:
    """k evenly spaced window indices; keeps coverage uniform when reading fewer windows."""
    if not 1 <= k <= n_windows:
        raise ValueError(f"k={k} out of range for {n_windows} windows")
    return np.unique(np.round(np.linspace(0, n_windows - 1, k)).astype(int))


def pool_windows(logits: np.ndarray, how: str = "mean") -> np.ndarray:
    """(n, w, 12) window logits -> (n, 12) study logits."""
    if how == "mean":
        return logits.mean(axis=1)
    if how == "max":
        return logits.max(axis=1)
    if how == "lse":
        m = logits.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(logits - m).mean(axis=1, keepdims=True)))[:, 0]
    raise ValueError(f"unknown pooling {how}")


def rank_blend(preds: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Per-label rank-mean blend; AUC only depends on ranks, so calibration differences vanish."""
    weights = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    ranked = [np.apply_along_axis(rankdata, 0, p) / len(p) for p in preds]
    return np.tensordot(weights / weights.sum(), np.stack(ranked), axes=1)


@dataclass(frozen=True)
class LogitPolicy:
    windows: int
    pooling: str = "mean"
    weights: tuple[float, ...] | None = None


def score_logit_policy(
    model_logits: Sequence[np.ndarray], y: np.ndarray, policy: LogitPolicy
) -> float:
    """Proxy macro AUC for one inference policy over one or more models' stored logits."""
    preds = []
    for lg in model_logits:
        idx = window_subset(lg.shape[1], min(policy.windows, lg.shape[1]))
        preds.append(pool_windows(lg[:, idx], policy.pooling))
    blended = preds[0] if len(preds) == 1 else rank_blend(preds, policy.weights)
    return macro_auc(y, blended)


def sweep_windows(
    model_logits: Sequence[np.ndarray],
    y: np.ndarray,
    counts: Sequence[int],
    pooling: str = "mean",
) -> dict[int, float]:
    return {k: score_logit_policy(model_logits, y, LogitPolicy(k, pooling)) for k in counts}


def efficiency_pick(auc_by_windows: dict[int, float], tolerance: float = 0.003) -> int:
    """Fewest windows whose AUC is within ``tolerance`` of the best (Final B)."""
    best = max(auc_by_windows.values())
    return min(k for k, a in auc_by_windows.items() if a >= best - tolerance)


# ----------------------------------------------------------------- tree world

TreePolicy = Callable[[Journal, set[str], int], list[str]]


def eligible(journal: Journal, revealed: set[str]) -> list[str]:
    """Root plus revealed nodes that still have unrevealed recorded children."""
    out = []
    for nid in [ROOT_ID, *sorted(revealed)]:
        if any(c.id not in revealed for c in journal.children(nid) if c.status != "pending"):
            out.append(nid)
    return out


def _next_child(journal: Journal, nid: str, revealed: set[str]) -> Node | None:
    for c in journal.children(nid):
        if c.status != "pending" and c.id not in revealed:
            return c
    return None


@dataclass(frozen=True)
class ReplayResult:
    score: float
    best_proxy: float
    gpu_hours: float
    revealed: int
    rounds: int


def replay(
    journal: Journal,
    policy: TreePolicy,
    *,
    workers: int = 1,
    max_rounds: int = 50,
    beta_cost: float = 0.002,
    beta_parallel: float = 0.0,
) -> ReplayResult:
    """Dream-RSI Eq. 1 with cost in GPU-hours: best proxy - b1*hours + b2*nodes/round.

    Each round the policy returns up to ``workers`` eligible node ids; each reveals its
    earliest unrevealed recorded child. An empty batch stops the replay.
    """
    revealed: set[str] = set()
    rounds = 0
    while rounds < max_rounds:
        allowed = set(eligible(journal, revealed))
        if not allowed:
            break
        batch = [n for n in policy(journal, set(revealed), workers) if n in allowed][:workers]
        if not batch:
            break
        for nid in batch:
            child = _next_child(journal, nid, revealed)
            if child is not None:
                revealed.add(child.id)
        rounds += 1
    nodes = [journal.nodes[n] for n in revealed]
    proxies = [n.proxy_auc for n in nodes if n.valid]
    best = max(proxies) if proxies else 0.0
    hours = sum(n.gpu_hours for n in nodes)
    score = best - beta_cost * hours + beta_parallel * len(nodes) / max(1, rounds)
    return ReplayResult(score, best, hours, len(nodes), rounds)


def evaluate_policies(
    worlds: Sequence[Journal], policies: dict[str, TreePolicy], **kw
) -> dict[str, float]:
    """Average replay score of each candidate policy over all recorded worlds."""
    return {
        name: float(np.mean([replay(w, p, **kw).score for w in worlds]))
        for name, p in policies.items()
    }


def greedy_best_first(stop_after_no_gain: int = 2) -> TreePolicy:
    """Expand the best revealed node; stop after ``stop_after_no_gain`` reveals without a gain."""

    def policy(journal: Journal, revealed: set[str], workers: int) -> list[str]:
        nodes = sorted((journal.nodes[n] for n in revealed), key=lambda n: n.created)
        best, since = None, 0
        for n in nodes:
            if n.valid and (best is None or n.proxy_auc > best):
                best, since = n.proxy_auc, 0
            else:
                since += 1
        if best is not None and since >= stop_after_no_gain:
            return []
        allowed = eligible(journal, revealed)
        ranked = sorted(
            allowed,
            key=lambda n: -1.0 if n == ROOT_ID else (journal.nodes[n].proxy_auc or 0.0),
            reverse=True,
        )
        return ranked[:workers]

    return policy
