"""Progressive MCGS node selection and stagnation-triggered operators (MLEvolve §3.2).

Selection walks the primary-edge tree with UCT whose exploration constant decays over the
campaign, and with probability ``1 - w(t)`` skips the walk and samples an elite node by
inverse rank. Stagnation decides which expansion operator the planner is asked for.

Time is the fraction of the campaign budget used, so the same schedule works whether the
budget is wall-clock days or GPU dollars.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from rsna_knee.agent.journal import ROOT_ID, Journal, Node


@dataclass(frozen=True)
class Schedule:
    c0: float = 1.4
    c_min: float = 0.3
    w_min: float = 0.2
    elite_k: int = 5
    tau_branch: int = 2
    tau_global: int = 4
    ref_top_n: int = 3
    hist_k: int = 3

    def c(self, t: float) -> float:
        """Piecewise exploration decay c0 -> c_min (flat first third, linear, flat last third)."""
        t = min(max(t, 0.0), 1.0)
        if t < 1 / 3:
            return self.c0
        if t > 2 / 3:
            return self.c_min
        return self.c0 + (self.c_min - self.c0) * (t - 1 / 3) * 3

    def w(self, t: float) -> float:
        """Probability of UCT exploration vs elite exploitation (Eq. 4)."""
        t = min(max(t, 0.0), 1.0)
        return max(self.w_min, 1.0 - t)


def uct(child: Node, parent_visits: int, c: float, eps: float = 1e-6) -> float:
    return child.q + c * math.sqrt(math.log(parent_visits + 1) / (child.visits + eps))


def select_uct(journal: Journal, t: float, sched: Schedule) -> Node:
    """Descend primary edges by UCT; stop at a node whose best child is not better than itself."""
    node = journal.nodes[ROOT_ID]
    c = sched.c(t)
    while True:
        kids = [k for k in journal.children(node.id) if k.status != "pending"]
        if not kids:
            return node
        best = max(kids, key=lambda k: uct(k, node.visits, c))
        if node.id != ROOT_ID and uct(best, node.visits, c) <= node.q:
            return node
        node = best


def select_elite(journal: Journal, sched: Schedule, rng: random.Random) -> Node | None:
    """Sample from the top-K valid nodes with weight 1/rank (Eq. 5)."""
    ranked = sorted(journal.valid_nodes(), key=lambda n: n.proxy_auc, reverse=True)[: sched.elite_k]
    if not ranked:
        return None
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    return rng.choices(ranked, weights=weights, k=1)[0]


def select(journal: Journal, t: float, sched: Schedule, rng: random.Random) -> tuple[Node, str]:
    if rng.random() >= sched.w(t):
        elite = select_elite(journal, sched, rng)
        if elite is not None:
            return elite, "elite"
    return select_uct(journal, t, sched), "uct"


def _stalled_steps(nodes: list[Node]) -> int:
    """Consecutive most-recent finished nodes that did not raise the running best."""
    finished = sorted((n for n in nodes if n.status != "pending"), key=lambda n: n.created)
    best, last_gain = None, -1
    for i, n in enumerate(finished):
        if n.valid and (best is None or n.proxy_auc > best):
            best, last_gain = n.proxy_auc, i
    return len(finished) - 1 - last_gain


def choose_operator(journal: Journal, node: Node, sched: Schedule) -> tuple[str, list[str]]:
    """Pick the expansion type and its reference set for the planner prompt."""
    if _stalled_steps([n for n in journal.nodes.values() if n.id != ROOT_ID]) >= sched.tau_global:
        top = sorted(journal.valid_nodes(), key=lambda n: n.proxy_auc, reverse=True)
        roots = []
        for n in top:
            b = journal.branch_root(n.id)
            if all(journal.branch_root(r) != b for r in roots):
                roots.append(n.id)
        if len(roots) >= 2:
            return "aggregate", roots[: sched.ref_top_n]
    if node.id == ROOT_ID:
        return "primary", []
    branch = journal.branch_root(node.id)
    if _stalled_steps(journal.branch_nodes(branch)) >= sched.tau_branch:
        others = [
            n for n in sorted(journal.valid_nodes(), key=lambda n: n.proxy_auc, reverse=True)
            if journal.branch_root(n.id) != branch
        ]
        if others:
            return "cross_branch", [n.id for n in others[: sched.ref_top_n]]
        hist = [n.id for n in journal.path_to_root(node.id)[1 : sched.hist_k + 1] if n.id != ROOT_ID]
        return "intra_branch", hist
    return "primary", []
