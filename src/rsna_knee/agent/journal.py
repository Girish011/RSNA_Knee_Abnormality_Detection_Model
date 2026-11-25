"""Experiment graph for the autonomous search loop (MLEvolve-style MCGS journal).

Each node is one candidate recipe. ``kind`` separates the two costs that matter on a
$300 budget:

* ``online`` - produced new weights on a rented GPU (expensive, rare).
* ``dream``  - re-scored stored per-window logits / OOF files (free, the default).

Primary edges (``parent``) carry credit assignment; reference edges (``refs``) record
cross-branch information reuse and never receive backpropagated reward (MLEvolve §3.2).
The journal is a plain JSON file so it survives spot-instance preemption and can be
committed next to ``docs/experiments.md``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT_ID = "root"


def genome_hash(genome: dict) -> str:
    """Stable id for a recipe; identical genomes on the same corpus are the same experiment."""
    blob = json.dumps(genome, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass
class Node:
    id: str
    parent: str | None
    genome: dict
    plan: str = ""
    kind: str = "online"
    refs: list[str] = field(default_factory=list)
    operator: str = "primary"
    status: str = "pending"
    proxy_auc: float | None = None
    gold_auc: float | None = None
    public_lb: float | None = None
    gpu_hours: float = 0.0
    error: str = ""
    reward: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    created: float = field(default_factory=time.time)

    @property
    def q(self) -> float:
        return self.value_sum / (self.visits + 1e-6)

    @property
    def valid(self) -> bool:
        return self.status == "done" and self.proxy_auc is not None


def reward_for(node: Node, branch_best: float | None) -> float:
    """MLEvolve Eq. 7: -1 failed, 1 valid but no gain, 2 refreshes the branch best."""
    if not node.valid:
        return -1.0
    if branch_best is None or node.proxy_auc > branch_best:
        return 2.0
    return 1.0


class Journal:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.nodes: dict[str, Node] = {ROOT_ID: Node(ROOT_ID, None, {}, status="done", kind="root")}

    def add(
        self,
        parent: str,
        genome: dict,
        *,
        plan: str = "",
        kind: str = "online",
        refs: list[str] | None = None,
        operator: str = "primary",
    ) -> Node:
        if parent not in self.nodes:
            raise KeyError(f"unknown parent {parent}")
        node_id = genome_hash(genome)
        if node_id in self.nodes:
            raise ValueError(f"duplicate genome {node_id}; reuse the recorded result instead")
        for r in refs or []:
            if r not in self.nodes:
                raise KeyError(f"unknown reference node {r}")
        node = Node(node_id, parent, genome, plan, kind, list(refs or []), operator)
        self.nodes[node_id] = node
        return node

    def children(self, node_id: str) -> list[Node]:
        return sorted(
            (n for n in self.nodes.values() if n.parent == node_id), key=lambda n: n.created
        )

    def path_to_root(self, node_id: str) -> list[Node]:
        out = []
        cur: str | None = node_id
        while cur is not None:
            node = self.nodes[cur]
            out.append(node)
            cur = node.parent
        return out

    def branch_root(self, node_id: str) -> str:
        """The root's child that heads this node's branch."""
        path = self.path_to_root(node_id)
        return path[-2].id if len(path) >= 2 else ROOT_ID

    def branch_nodes(self, branch: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.id != ROOT_ID and self.branch_root(n.id) == branch]

    def branch_best(self, branch: str, exclude: str | None = None) -> float | None:
        scores = [
            n.proxy_auc for n in self.branch_nodes(branch) if n.valid and n.id != exclude
        ]
        return max(scores) if scores else None

    def record(
        self,
        node_id: str,
        *,
        proxy_auc: float | None = None,
        gold_auc: float | None = None,
        public_lb: float | None = None,
        gpu_hours: float = 0.0,
        error: str = "",
    ) -> float:
        """Store an outcome, assign the Eq. 7 reward and backpropagate on primary edges only."""
        node = self.nodes[node_id]
        node.proxy_auc, node.gold_auc, node.public_lb = proxy_auc, gold_auc, public_lb
        node.gpu_hours, node.error = gpu_hours, error
        node.status = "failed" if error or proxy_auc is None else "done"
        best = self.branch_best(self.branch_root(node_id), exclude=node_id)
        node.reward = reward_for(node, best)
        for anc in self.path_to_root(node_id):
            anc.visits += 1
            anc.value_sum += node.reward
        return node.reward

    def valid_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.id != ROOT_ID and n.valid]

    def best(self) -> Node | None:
        valid = self.valid_nodes()
        return max(valid, key=lambda n: n.proxy_auc) if valid else None

    def gpu_hours_spent(self) -> float:
        return sum(n.gpu_hours for n in self.nodes.values())

    def save(self, path: Path | None = None) -> Path:
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(n) for n in self.nodes.values()], indent=1))
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path) -> Journal:
        j = cls(path)
        j.nodes = {d["id"]: Node(**d) for d in json.loads(Path(path).read_text())}
        return j
