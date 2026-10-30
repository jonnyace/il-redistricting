"""Seeded search: draw a first plan inside the population band, then anneal.

Every random choice comes from one numpy Generator built from the seed in
config/params.toml, so the same seed and data always give the same map.

Moves (both keep every district contiguous and inside the population band):
  * recombination ("ReCom"): merge two neighboring districts, draw a random
    spanning tree of the merged area, cut one tree edge so both halves are in
    the population band. Edges that cross county or municipal lines get a
    surcharge, so the tree prefers to cross them rarely and cuts tend to
    follow those lines.
  * flip: move one precinct on a district border to the neighboring district.
A move is kept if it lowers the score, or with probability exp(-increase/T)
(simulated annealing), with T falling from temperature_start to temperature_end.
"""
from __future__ import annotations

import math
import sys
import time
from collections import deque

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from .model import Plan, Problem


def edge_surcharge(p: Problem) -> np.ndarray:
    s = p.params["score"]
    cross_county = p.county_code[p.eu] != p.county_code[p.ev]
    cross_muni = p.muni_code[p.eu] != p.muni_code[p.ev]
    return s["county_surcharge"] * cross_county + s["municipality_surcharge"] * cross_muni


def tree_split(p: Problem, esur: np.ndarray, nodes: np.ndarray, lo_t: float, hi_t: float,
               rng: np.random.Generator, tries: int):
    """Return a bool mask over `nodes` for a connected part with population in
    [lo_t, hi_t] whose complement is also connected, or None."""
    L = len(nodes)
    idx = np.full(p.n, -1, np.int64)
    idx[nodes] = np.arange(L)
    em = (idx[p.eu] >= 0) & (idx[p.ev] >= 0)
    a, b, sur = idx[p.eu[em]], idx[p.ev[em]], esur[em]
    pops = p.pop[nodes].astype(float)
    total = pops.sum()
    for _ in range(tries):
        w = rng.random(len(a)) + sur + 1e-9
        tree = minimum_spanning_tree(coo_matrix((w, (a, b)), shape=(L, L)).tocsr())
        order, pred = breadth_first_order(tree, 0, directed=False, return_predecessors=True)
        if len(order) < L:
            raise RuntimeError("merged area is not connected")
        sub = pops.copy()
        for v in order[:0:-1]:
            sub[pred[v]] += sub[v]
        sub[0] = -1  # root cannot be cut
        c1 = np.flatnonzero((sub >= lo_t) & (sub <= hi_t))                               # subtree is the part
        c2 = np.flatnonzero((total - sub >= lo_t) & (total - sub <= hi_t) & (sub >= 0))  # complement is
        nc = len(c1) + len(c2)
        if nc == 0:
            continue
        j = int(rng.integers(nc))
        cut, flip = (c1[j], False) if j < len(c1) else (c2[j - len(c1)], True)
        in_sub = np.zeros(L, bool)
        in_sub[cut] = True
        for v in order:
            pv = pred[v]
            if pv >= 0 and in_sub[pv]:
                in_sub[v] = True
        return ~in_sub if flip else in_sub
    return None


def initial_plan(p: Problem, esur, rng, log=print) -> np.ndarray:
    """Recursive bisection. A piece that will hold k districts must have an
    average district population within eps / (1 + log2 k) of ideal, so every
    final district (k = 1) lands inside the band. If a piece cannot be split,
    its parent split is redrawn."""
    eps = float(p.params["plan"]["population_tolerance"])
    asg = np.full(p.n, -1, np.int64)

    def band(k: int) -> float:
        # Allowed average deviation for a piece that will hold k districts. It
        # shrinks as k grows, so a piece at the edge of its band can still be split.
        return eps / (1 + math.log2(k))

    def rec(nodes, labels) -> bool:
        k = len(labels)
        if k == 1:
            asg[nodes] = labels[0]
            return True
        k1, k2 = k // 2, k - k // 2
        e1, e2 = band(k1), band(k2)
        T = p.pop[nodes].sum()
        lo_t = max(k1 * p.ideal * (1 - e1), T - k2 * p.ideal * (1 + e2))
        hi_t = min(k1 * p.ideal * (1 + e1), T - k2 * p.ideal * (1 - e2))
        for _ in range(25):
            mask = tree_split(p, esur, nodes, lo_t, hi_t, rng, tries=40)
            if mask is not None and rec(nodes[mask], labels[:k1]) and rec(nodes[~mask], labels[k1:]):
                return True
        return False

    if not rec(np.flatnonzero(p.active), list(range(p.k))):
        raise RuntimeError("could not draw a first plan")
    return asg


def district_graph(asg, p: Problem) -> dict[int, set]:
    nb = {d: set() for d in range(p.k)}
    a, b = asg[p.eu], asg[p.ev]
    cut = a != b
    for x, y in zip(a[cut].tolist(), b[cut].tolist()):
        nb[x].add(y)
        nb[y].add(x)
    return nb


def _connected_without(plan: Plan, u: int) -> bool:
    p, asg = plan.p, plan.asg
    A = asg[u]
    nbrs = [v for v in p.adj_nodes[p.indptr[u]:p.indptr[u + 1]].tolist() if asg[v] == A]
    if len(nbrs) == 0:
        return False
    if len(nbrs) == 1:
        return True
    want = set(nbrs)
    seen = {u, nbrs[0]}
    want.discard(nbrs[0])
    q = deque([nbrs[0]])
    while q and want:
        x = q.popleft()
        for y in p.adj_nodes[p.indptr[x]:p.indptr[x + 1]].tolist():
            if y not in seen and asg[y] == A:
                seen.add(y)
                want.discard(y)
                q.append(y)
    return not want


def anneal(plan: Plan, esur, rng, log=print) -> dict:
    p, s = plan.p, plan.p.params["search"]
    steps, rshare = int(s["steps"]), float(s["recom_share"])
    T0, T1 = float(s["temperature_start"]), float(s["temperature_end"])
    cur = plan.score()
    best, best_asg = cur, plan.asg.copy()
    stats = {"recom_tried": 0, "recom_accepted": 0, "flip_tried": 0, "flip_accepted": 0}
    m = len(p.eu)
    t0 = time.time()
    for step in range(steps):
        if step % 5000 == 0 or step == steps - 1:
            sp = plan.score_parts()
            log(f"  step {step:>6} score={cur:9.1f} best={best:9.1f} border_km={sp['border_km']:.0f} "
                f"split_pen={sp['split_penalty']:.0f} partisan_pen={sp['partisan_penalty']:.1f} "
                f"({time.time() - t0:.0f}s)")
        T = T0 * (T1 / T0) ** (step / max(steps - 1, 1))
        for _ in range(200):  # find a cut edge
            e = int(rng.integers(m))
            u, v = int(p.eu[e]), int(p.ev[e])
            if plan.asg[u] != plan.asg[v]:
                break
        else:
            continue
        if rng.random() < rshare:
            stats["recom_tried"] += 1
            A, B = int(plan.asg[u]), int(plan.asg[v])
            tot = plan.dpop[A] + plan.dpop[B]
            lo_t, hi_t = max(p.lo, tot - p.hi), min(p.hi, tot - p.lo)
            if lo_t > hi_t:
                continue
            nodes = np.flatnonzero((plan.asg == A) | (plan.asg == B))
            mask = tree_split(p, esur, nodes, lo_t, hi_t, rng, int(s["recom_tries"]))
            if mask is None:
                continue
            isA = plan.asg[nodes] == A
            keep = (mask & isA).sum() + (~mask & ~isA).sum()
            new = np.where(mask, A, B) if keep * 2 >= len(nodes) else np.where(mask, B, A)
            kind = "recom"
        else:
            stats["flip_tried"] += 1
            if rng.random() < 0.5:
                u, v = v, u
            A, B = int(plan.asg[u]), int(plan.asg[v])
            if plan.dpop[A] - p.pop[u] < p.lo or plan.dpop[B] + p.pop[u] > p.hi:
                continue
            if not _connected_without(plan, u):
                continue
            nodes, new = np.array([u]), np.array([B])
            kind = "flip"
        d, change = plan.delta(nodes, new)
        if change is None:
            continue
        if d <= 0 or rng.random() < math.exp(-d / T):
            plan.apply(change)
            cur += d
            stats[f"{kind}_accepted"] += 1
            if cur < best - 1e-9:
                best, best_asg = cur, plan.asg.copy()
    plan.asg = best_asg
    plan.recompute()
    stats["final_score"] = round(plan.score(), 3)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def solve(p: Problem, mode: str, log=lambda *a: print(*a, file=sys.stderr, flush=True)) -> tuple[Plan, dict]:
    seed = int(p.params["plan"]["seed"])
    rng = np.random.default_rng(seed)
    esur = edge_surcharge(p)
    t = time.time()
    log(f"[{mode}] seed {seed}: drawing first plan")
    plan = Plan(p, initial_plan(p, esur, rng, log), mode)
    if plan.dpop.min() < p.lo or plan.dpop.max() > p.hi:
        raise RuntimeError("first plan outside the population band")
    log(f"  first plan in {time.time() - t:.0f}s, score {plan.score():.1f}; annealing "
        f"{p.params['search']['steps']} steps")
    stats = anneal(plan, esur, rng, log)
    stats["total_seconds"] = round(time.time() - t, 1)
    return plan, stats
