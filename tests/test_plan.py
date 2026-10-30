import copy

import networkx as nx
import numpy as np
import pytest

from ilredistrict.build import PROCESSED
from ilredistrict.metrics import lean
from ilredistrict.model import Plan, load_params, seat_targets

needs_data = pytest.mark.skipif(not (PROCESSED / "graph.json").exists(),
                                reason="run `python -m ilredistrict fetch` and `build` first")


def test_weaver_seat_targets():
    assert seat_targets(118, (0.30, 0.32, 0.38)) == (35, 38, 45)
    assert sum(seat_targets(59, (0.30, 0.32, 0.38))) == 59


def test_lean_bands():
    assert lean(-5.01, 5) == "R"
    assert lean(-5.0, 5) == "toss-up"
    assert lean(5.0, 5) == "toss-up"
    assert lean(5.01, 5) == "D"


@pytest.fixture(scope="module")
def problem():
    from ilredistrict.model import load_problem
    params = load_params()
    params["search"]["steps"] = 1500
    return load_problem(params)


def _solve(problem, mode):
    from ilredistrict.optimize import solve
    return solve(problem, mode, log=lambda *a: None)


@needs_data
@pytest.mark.parametrize("mode", ["hjrca41", "weaver"])
def test_plan_is_legal_and_score_is_exact(problem, mode):
    plan, _ = _solve(problem, mode)
    p = problem
    # every populated unit assigned, water left out
    assert (plan.asg[p.active] >= 0).all() and (plan.asg[~p.active] == -1).all()
    # population band: each district within +/- tolerance, so range <= 2 * tolerance
    assert plan.dpop.min() >= p.lo and plan.dpop.max() <= p.hi
    # contiguity
    G = nx.Graph()
    G.add_edges_from(zip(p.eu.tolist(), p.ev.tolist()))
    for d in range(p.k):
        assert nx.is_connected(G.subgraph(np.flatnonzero(plan.asg == d).tolist()))
    # running score equals a from-scratch score
    s = plan.score()
    fresh = Plan(p, plan.asg, mode)
    assert fresh.score() == pytest.approx(s, abs=1e-6)


@needs_data
def test_same_seed_same_map(problem):
    a, _ = _solve(problem, "weaver")
    b, _ = _solve(problem, "weaver")
    assert np.array_equal(a.asg, b.asg)


@needs_data
def test_different_seed_different_map(problem):
    from ilredistrict.optimize import solve
    p2 = copy.copy(problem)
    p2.params = copy.deepcopy(problem.params)
    p2.params["plan"]["seed"] += 1
    a, _ = _solve(problem, "hjrca41")
    b, _ = solve(p2, "hjrca41", log=lambda *a: None)
    assert not np.array_equal(a.asg, b.asg)


@needs_data
def test_hjrca41_mode_never_reads_votes(problem):
    """HJRCA 41 forbids using election results; prove the optimizer ignores them."""
    from ilredistrict.optimize import solve
    a, _ = _solve(problem, "hjrca41")
    p2 = copy.copy(problem)
    p2.votes = np.random.default_rng(0).random(problem.votes.shape) * 1000
    b, _ = solve(p2, "hjrca41", log=lambda *a: None)
    assert np.array_equal(a.asg, b.asg)
