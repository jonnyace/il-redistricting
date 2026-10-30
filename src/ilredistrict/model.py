"""The problem (units, graph, data) and a plan's running score.

A plan is an integer array `asg` with one district label (0..k-1) per unit.
Lake Michigan water VTDs (population 0) are left out of every district, the
same way the Census labels them "ZZZ" in the enacted map.
"""
from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import dataclass, field

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .build import PROCESSED
from .fetch import ROOT

YEARS = ("14", "18", "22")
KINDS = ("county", "municipality", "township")


def load_params(path=None) -> dict:
    with open(path or ROOT / "config" / "params.toml", "rb") as f:
        return tomllib.load(f)


def seat_targets(k: int, shares: tuple[float, float, float]) -> tuple[int, int, int]:
    """Largest-remainder rounding of (R, toss-up, D) shares to whole seats."""
    raw = [s * k for s in shares]
    base = [int(np.floor(x)) for x in raw]
    order = sorted(range(3), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[: k - sum(base)]:
        base[i] += 1
    return tuple(base)


@dataclass
class Problem:
    units: gpd.GeoDataFrame          # all VTDs, row i = unit i
    active: np.ndarray               # bool, False for lake water VTDs
    pop: np.ndarray
    votes: np.ndarray                # shape (3 years, 2 parties [D, R], n)
    eu: np.ndarray                   # edge endpoints (active units only)
    ev: np.ndarray
    ew: np.ndarray                   # shared border length, meters
    indptr: np.ndarray               # CSR adjacency over units -> edge ids
    adj_nodes: np.ndarray
    adj_edges: np.ndarray
    county_code: np.ndarray
    muni_code: np.ndarray
    unit_kind: list[str]             # per split unit
    node_units: list[list[int]]      # per VTD: split-unit indices it has people in
    k: int
    ideal: float
    lo: float
    hi: float
    params: dict = field(repr=False, default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.pop)


def load_problem(params: dict) -> Problem:
    units = gpd.read_file(PROCESSED / "units.gpkg")
    with open(PROCESSED / "graph.json") as f:
        g = nx.node_link_graph(json.load(f), edges="edges")
    n = len(units)
    active = ~units["GEOID20"].str.endswith("ZZZZZZ").to_numpy()
    pop = units["pop"].to_numpy(np.int64)
    if pop[~active].sum() != 0:
        raise SystemExit("a water VTD has population; refusing to drop it")
    votes = np.stack([np.stack([units[f"d{y}"].to_numpy(float), units[f"r{y}"].to_numpy(float)]) for y in YEARS])

    edges = sorted((min(u, v), max(u, v), float(d.get("shared_perim", 0.0))) for u, v, d in g.edges(data=True)
                   if active[u] and active[v])
    eu = np.array([e[0] for e in edges], np.int64)
    ev = np.array([e[1] for e in edges], np.int64)
    ew = np.array([e[2] for e in edges], float)
    sub = nx.Graph()
    sub.add_nodes_from(np.flatnonzero(active).tolist())
    sub.add_edges_from(zip(eu.tolist(), ev.tolist()))
    if not nx.is_connected(sub):
        raise SystemExit("unit graph is not connected")

    # CSR: for each node, its neighbor nodes and the edge ids.
    ends = np.concatenate([eu, ev])
    others = np.concatenate([ev, eu])
    eids = np.concatenate([np.arange(len(eu)), np.arange(len(eu))])
    order = np.lexsort((others, ends))
    ends, others, eids = ends[order], others[order], eids[order]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, ends + 1, 1)
    indptr = np.cumsum(indptr)

    county_code = pd.factorize(units["county"])[0]
    muni = units["municipality"].fillna("")
    muni_code = np.where(muni == "", -1, pd.factorize(muni)[0])

    pairs = pd.read_csv(PROCESSED / "split_pairs.csv", dtype={"unit_id": str})
    keys = pairs["kind"] + ":" + pairs["unit_id"]
    codes, uniq = pd.factorize(keys)
    unit_kind = [u.split(":")[0] for u in uniq]
    node_units: list[list[int]] = [[] for _ in range(n)]
    for c, v in zip(codes.tolist(), pairs["vtd"].tolist()):
        node_units[v].append(c)

    k = int(params["plan"]["districts"])
    ideal = pop.sum() / k
    tol = float(params["plan"]["population_tolerance"])
    return Problem(units=units, active=active, pop=pop, votes=votes, eu=eu, ev=ev, ew=ew,
                   indptr=indptr, adj_nodes=others, adj_edges=eids, county_code=county_code,
                   muni_code=muni_code, unit_kind=unit_kind, node_units=node_units, k=k,
                   ideal=ideal, lo=ideal * (1 - tol), hi=ideal * (1 + tol), params=params)


class Plan:
    """A plan plus everything needed to score a change quickly."""

    def __init__(self, prob: Problem, asg: np.ndarray, mode: str):
        self.p = prob
        self.mode = mode
        self.asg = asg.astype(np.int64).copy()
        s = prob.params["score"]
        self.kind_w = {"county": s["county_split_weight"], "municipality": s["municipality_split_weight"],
                       "township": s["township_split_weight"]}
        self.unit_w = np.array([self.kind_w[kd] for kd in prob.unit_kind])
        if mode == "weaver":
            w = prob.params["weaver"]
            self.tossup = float(w["tossup_margin"])
            self.nR, self.nT, self.nD = seat_targets(prob.k, (w["share_r_leaning"], w["share_tossup"], w["share_d_leaning"]))
            self.part_w = float(w["partisan_weight"])
        self.recompute()

    # ---- full recomputation (used at start and in tests) ----
    def recompute(self) -> None:
        p, a = self.p, self.asg
        act = p.active
        self.dpop = np.bincount(a[act], weights=p.pop[act], minlength=p.k)
        # Election data is only read in weaver mode. In hjrca41 mode the optimizer
        # never looks at votes (HJRCA 41 forbids it); results are scored afterwards.
        self.dvotes = None
        if self.mode == "weaver":
            self.dvotes = np.zeros((3, 2, p.k))
            for y in range(3):
                for party in range(2):
                    self.dvotes[y, party] = np.bincount(a[act], weights=p.votes[y, party][act], minlength=p.k)
        self.cut_m = float(p.ew[a[p.eu] != a[p.ev]].sum())
        self.ucount = [Counter() for _ in p.unit_kind]
        for v in np.flatnonzero(act):
            for u in p.node_units[v]:
                self.ucount[u][a[v]] += 1
        self.pieces = np.array([len(c) - 1 for c in self.ucount])

    # ---- score parts ----
    def margins(self, dvotes=None) -> np.ndarray:
        """Average two-party margin (D minus R, points) over the three races, per district."""
        dv = self.dvotes if dvotes is None else dvotes
        m = 100 * (dv[:, 0] - dv[:, 1]) / np.maximum(dv[:, 0] + dv[:, 1], 1)
        return m.mean(axis=0)

    def partisan_penalty(self, dvotes=None) -> float:
        if self.mode != "weaver":
            return 0.0
        m = np.sort(self.margins(dvotes))
        t, buf = self.tossup, 0.1   # small buffer so a seat sits strictly inside its band
        pen = np.maximum(0, m[: self.nR] + t + buf).sum()
        mid = m[self.nR: self.nR + self.nT]
        pen += np.maximum(0, np.abs(mid) - (t - buf)).sum()
        pen += np.maximum(0, t + buf - m[self.nR + self.nT:]).sum()
        return float(pen)

    def score_parts(self) -> dict:
        return {"border_km": self.cut_m / 1000, "split_penalty": float((self.pieces * self.unit_w).sum()),
                "partisan_penalty": self.partisan_penalty()}

    def score(self) -> float:
        s = self.score_parts()
        pw = self.part_w if self.mode == "weaver" else 0.0
        return s["border_km"] + s["split_penalty"] + pw * s["partisan_penalty"]

    # ---- incremental change: nodes `nodes` move to labels `new` ----
    def delta(self, nodes: np.ndarray, new: np.ndarray):
        p = self.p
        old = self.asg[nodes]
        ch = old != new
        nodes, old, new = nodes[ch], old[ch], new[ch]
        if len(nodes) == 0:
            return 0.0, None
        trial = self.asg.copy()
        trial[nodes] = new
        eids = np.unique(np.concatenate([p.adj_edges[p.indptr[v]:p.indptr[v + 1]] for v in nodes]))
        ue, ve, we = p.eu[eids], p.ev[eids], p.ew[eids]
        d_cut = float(we[trial[ue] != trial[ve]].sum() - we[self.asg[ue] != self.asg[ve]].sum())

        touched: dict[int, Counter] = {}
        for v, o, nw in zip(nodes.tolist(), old.tolist(), new.tolist()):
            for u in p.node_units[v]:
                c = touched.get(u)
                if c is None:
                    c = touched[u] = Counter(self.ucount[u])
                c[o] -= 1
                if c[o] == 0:
                    del c[o]
                c[nw] += 1
        d_split = sum(self.unit_w[u] * ((len(c) - 1) - self.pieces[u]) for u, c in touched.items())

        dpop = self.dpop.copy()
        np.add.at(dpop, old, -p.pop[nodes])
        np.add.at(dpop, new, p.pop[nodes])
        d_part, dvotes = 0.0, None
        if self.mode == "weaver":
            dvotes = self.dvotes.copy()
            for y in range(3):
                for party in range(2):
                    np.add.at(dvotes[y, party], old, -p.votes[y, party][nodes])
                    np.add.at(dvotes[y, party], new, p.votes[y, party][nodes])
            d_part = self.part_w * (self.partisan_penalty(dvotes) - self.partisan_penalty())
        d = d_cut / 1000 + d_split + d_part
        return d, (nodes, new, trial, d_cut, touched, dpop, dvotes)

    def apply(self, change) -> None:
        nodes, new, trial, d_cut, touched, dpop, dvotes = change
        self.asg = trial
        self.cut_m += d_cut
        for u, c in touched.items():
            self.ucount[u] = c
            self.pieces[u] = len(c) - 1
        self.dpop = dpop
        if dvotes is not None:
            self.dvotes = dvotes
