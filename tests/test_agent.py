import random
from pathlib import Path

import numpy as np
import pytest

from rsna_knee.agent import activity, dream
from rsna_knee.agent.journal import ROOT_ID, Journal
from rsna_knee.agent.memory import check_bans, load_cold_start, retrieve
from rsna_knee.agent.search import Schedule, choose_operator, select, select_uct

REPO = Path(__file__).resolve().parents[1]


def _chain(j, parent, scores, branch="a", hours=1.0):
    ids = []
    for i, s in enumerate(scores):
        n = j.add(parent, {"branch": branch, "step": i})
        j.record(n.id, proxy_auc=s, gpu_hours=hours)
        ids.append(n.id)
        parent = n.id
    return ids


def test_reward_and_backprop_primary_only():
    j = Journal()
    a1, a2, a3 = _chain(j, ROOT_ID, [0.80, 0.85, 0.84])
    assert [j.nodes[i].reward for i in (a1, a2, a3)] == [2.0, 2.0, 1.0]
    ref = j.add(ROOT_ID, {"branch": "b"}, refs=[a2], operator="cross_branch")
    assert j.record(ref.id, error="OOM") == -1.0
    assert j.nodes[a2].visits == 2  # own record + a3; the reference edge did not add a visit
    assert j.nodes[ROOT_ID].visits == 4


def test_duplicate_genome_rejected_and_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.json")
    _chain(j, ROOT_ID, [0.7])
    with pytest.raises(ValueError):
        j.add(ROOT_ID, {"branch": "a", "step": 0})
    j.save()
    j2 = Journal.load(tmp_path / "j.json")
    assert j2.best().proxy_auc == 0.7
    assert j2.gpu_hours_spent() == 1.0


def test_schedule_decays_and_elite_takeover():
    s = Schedule()
    assert s.c(0.0) == s.c0 and s.c(1.0) == s.c_min
    assert s.w(0.0) == 1.0 and s.w(1.0) == s.w_min
    j = Journal()
    _chain(j, ROOT_ID, [0.80, 0.90])
    rng = random.Random(0)
    modes = {select(j, 1.0, s, rng)[1] for _ in range(50)}
    assert modes == {"uct", "elite"}


def test_select_uct_descends_to_node():
    j = Journal()
    _chain(j, ROOT_ID, [0.8, 0.9])
    assert select_uct(j, 0.9, Schedule()).id != ROOT_ID


def test_stagnation_operators():
    s = Schedule(tau_branch=2, tau_global=10)
    j = Journal()
    a = _chain(j, ROOT_ID, [0.90, 0.85, 0.86], branch="a")
    op, refs = choose_operator(j, j.nodes[a[-1]], s)
    assert op == "intra_branch" and refs
    b = _chain(j, ROOT_ID, [0.88], branch="b")
    op, refs = choose_operator(j, j.nodes[a[-1]], s)
    assert op == "cross_branch" and refs == b
    s2 = Schedule(tau_branch=2, tau_global=2)
    _chain(j, b[0], [0.80, 0.81], branch="b2")
    op, refs = choose_operator(j, j.nodes[a[-1]], s2)
    assert op == "aggregate" and len(refs) == 2


def test_bans():
    assert {b.rule for b in check_bans("Unfreeze last 4 blocks at lr x0.1")} == {"unfreeze_noisy_teacher"}
    assert check_bans("use report text at test time via NLI")
    assert check_bans("pretrain on MRNet")
    assert not check_bans("dense 80-slice corpus, soft labels from reports at train time")


def test_cold_start_retrieval():
    recs = load_cold_start(REPO / "agent" / "memory" / "cold_start.md")
    assert len(recs) > 15
    hits = retrieve("should we unfreeze the backbone", recs, k=3)
    assert any("unfreeze" in h.lower() for h in hits)


def test_window_sweep_and_efficiency_pick():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(200, 12)).astype(float)
    signal = (y[:, None, :] * 2 - 1) * 0.5
    logits = signal + rng.normal(0, 2, size=(200, 40, 12))
    aucs = dream.sweep_windows([logits], y, [2, 10, 40])
    assert aucs[40] > aucs[2]
    pick = dream.efficiency_pick({10: 0.950, 20: 0.955, 40: 0.956})
    assert pick == 20
    assert dream.window_subset(10, 1).tolist() == [0]


def test_rank_blend_bounds():
    a = np.random.default_rng(1).random((50, 12))
    out = dream.rank_blend([a, a[::-1]], [3, 1])
    assert out.shape == (50, 12) and out.min() > 0 and out.max() <= 1


def test_activity_log_budget_and_secret_guard(tmp_path):
    p = tmp_path / "a.jsonl"
    activity.log("train", "fold0", cost_usd=120, path=p, gpu_hours=350)
    activity.log("note", "hello", path=p)
    assert activity.spent(activity.read(p)) == 120
    activity.assert_budget(100, path=p)
    with pytest.raises(RuntimeError):
        activity.assert_budget(200, path=p)
    with pytest.raises(ValueError):
        activity.log("note", "token rpa_ABCDEFGHIJKLMNOP1234", path=p)
    with pytest.raises(ValueError):
        activity.log("bogus", "x", path=p)


def test_tree_replay_stops_early_and_charges_gpu():
    j = Journal()
    _chain(j, ROOT_ID, [0.90, 0.89, 0.88, 0.87], branch="a", hours=5)
    _chain(j, ROOT_ID, [0.80], branch="b", hours=5)
    greedy = dream.replay(j, dream.greedy_best_first(stop_after_no_gain=1), beta_cost=0.001)
    exhaustive = dream.replay(j, dream.greedy_best_first(stop_after_no_gain=99), beta_cost=0.001)
    assert greedy.best_proxy == exhaustive.best_proxy == 0.90
    assert greedy.gpu_hours < exhaustive.gpu_hours
    assert greedy.score > exhaustive.score
    scores = dream.evaluate_policies(
        [j], {"g1": dream.greedy_best_first(1), "g99": dream.greedy_best_first(99)}, beta_cost=0.001
    )
    assert scores["g1"] > scores["g99"]
