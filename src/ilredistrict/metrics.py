"""Score any plan (ours, or the enacted 2021 House map) the same way."""
from __future__ import annotations

import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .build import CRS, PROCESSED
from .fetch import RAW, load_sources
from .model import YEARS, Problem


def lean(m: float, tossup: float) -> str:
    return "R" if m < -tossup else ("D" if m > tossup else "toss-up")


def _district_table(df: pd.DataFrame, ideal: float, tossup: float) -> pd.DataFrame:
    """df: one row per district with pop, vap, hvap, bvap, wvap, d14..r22, geometry."""
    t = df.copy()
    t["deviation_pct"] = 100 * (t["pop"] - ideal) / ideal
    t["hispanic_vap_pct"] = 100 * t["hvap"] / t["vap"]
    t["black_vap_pct"] = 100 * t["bvap"] / t["vap"]
    t["minority_vap_pct"] = 100 * (t["vap"] - t["wvap"]) / t["vap"]
    for y in YEARS:
        t[f"margin_gov20{y}"] = 100 * (t[f"d{y}"] - t[f"r{y}"]) / (t[f"d{y}"] + t[f"r{y}"])
    t["margin_avg"] = t[[f"margin_gov20{y}" for y in YEARS]].mean(axis=1)
    t["lean"] = [lean(m, tossup) for m in t["margin_avg"]]
    area = t.geometry.area
    perim = t.geometry.length
    t["area_km2"] = area / 1e6
    t["perimeter_km"] = perim / 1e3
    t["polsby_popper"] = 4 * math.pi * area / perim**2
    return t


def _summary(t: gpd.GeoDataFrame, pieces: pd.DataFrame, tossup: float) -> dict:
    outline = t.geometry.union_all().length
    s = {
        "districts": int(len(t)),
        "population_max_over_pct": round(float(t["deviation_pct"].max()), 3),
        "population_max_under_pct": round(float(t["deviation_pct"].min()), 3),
        "population_range_pct": round(float(t["deviation_pct"].max() - t["deviation_pct"].min()), 3),
        "interior_border_km": round(float((t.geometry.length.sum() - outline) / 2 / 1000), 1),
        "polsby_popper_mean": round(float(t["polsby_popper"].mean()), 4),
        "polsby_popper_min": round(float(t["polsby_popper"].min()), 4),
    }
    for kind in ("county", "municipality", "township"):
        k = pieces[pieces.kind == kind].groupby("unit_id")["district"].nunique()
        s[f"{kind}_units"] = int(len(k))
        s[f"{kind}_units_split"] = int((k > 1).sum())
        s[f"{kind}_extra_pieces"] = int((k - 1).sum())
    for cat in ("R", "toss-up", "D"):
        s[f"seats_{cat}"] = int((t["lean"] == cat).sum())
    for y in YEARS:
        s[f"seats_won_D_gov20{y}"] = int((t[f"margin_gov20{y}"] > 0).sum())
    s["vra_screen_hispanic_majority_vap"] = int((t["hispanic_vap_pct"] > 50).sum())
    s["vra_screen_black_majority_vap"] = int((t["black_vap_pct"] > 50).sum())
    s["vra_screen_minority_majority_vap"] = int((t["minority_vap_pct"] > 50).sum())
    s["tossup_margin_pts"] = tossup
    return s


def evaluate_plan(p: Problem, asg: np.ndarray, labels: np.ndarray) -> tuple[gpd.GeoDataFrame, dict]:
    """asg: internal label per unit (-1 = water); labels[internal] = public district number."""
    tossup = float(p.params["weaver"]["tossup_margin"])
    u = p.units.loc[p.active].copy()
    u["district"] = labels[asg[p.active]]
    cols = ["pop", "vap", "hvap", "bvap", "wvap"] + [f"{x}{y}" for y in YEARS for x in "dr"]
    g = u[["district", "geometry"] + cols].dissolve(by="district", aggfunc="sum").reset_index()
    t = _district_table(g, p.ideal, tossup)
    pairs = pd.read_csv(PROCESSED / "split_pairs.csv", dtype={"unit_id": str})
    pairs["district"] = labels[asg[pairs["vtd"].to_numpy()]]
    return t, _summary(t, pairs, tossup)


def evaluate_enacted(p: Problem) -> tuple[gpd.GeoDataFrame, dict]:
    """The 2021 enacted House map (used for the 2022 election), scored identically.
    VTD-level votes are shared out to split VTDs by population."""
    tossup = float(p.params["weaver"]["tossup_margin"])
    ep = pd.read_csv(PROCESSED / "enacted_pairs.csv", dtype={"enacted": str})
    ep = ep[ep["enacted"] != "ZZZ"]
    vpop = p.pop[ep["vtd"].to_numpy()]
    share = np.where(vpop > 0, ep["pop"].to_numpy() / np.maximum(vpop, 1), 0.0)
    for j, y in enumerate(YEARS):
        for i, x in enumerate("dr"):
            ep[f"{x}{y}"] = p.votes[j, i][ep["vtd"].to_numpy()] * share
    cols = ["pop", "vap", "hvap", "bvap", "wvap"] + [f"{x}{y}" for y in YEARS for x in "dr"]
    agg = ep.groupby("enacted")[cols].sum()
    shp = gpd.read_file(RAW / load_sources()["enacted_sldl_shapes"]["file"]).to_crs(CRS)
    shp = shp[shp["SLDLST"] != "ZZZ"][["SLDLST", "geometry"]].rename(columns={"SLDLST": "enacted"})
    g = shp.merge(agg, left_on="enacted", right_index=True).rename(columns={"enacted": "district"})
    g["district"] = g["district"].astype(int)
    g = g.sort_values("district").reset_index(drop=True)
    t = _district_table(g, p.ideal, tossup)
    es = pd.read_csv(PROCESSED / "enacted_splits.csv", dtype={"unit_id": str, "enacted": str})
    es = es[es["enacted"] != "ZZZ"].rename(columns={"enacted": "district"})
    return t, _summary(t, es, tossup)


def senate_pairs(p: Problem, asg: np.ndarray) -> list[tuple[int, int]] | None:
    """Pair the 118 House districts into 59 Senate districts of two touching
    House districts (Ill. Const. art. IV sec. 2), preferring long shared
    borders. Uses a maximum-weight perfect matching; None if impossible."""
    a, b = asg[p.eu], asg[p.ev]
    cut = a != b
    w: dict[tuple[int, int], float] = {}
    for x, y, l in zip(a[cut].tolist(), b[cut].tolist(), p.ew[cut].tolist()):
        key = (min(x, y), max(x, y))
        w[key] = w.get(key, 0.0) + l
    G = nx.Graph()
    G.add_nodes_from(range(p.k))
    for (x, y), l in sorted(w.items()):
        G.add_edge(x, y, weight=1.0 + l / 1000.0)
    m = nx.max_weight_matching(G, maxcardinality=True)
    if len(m) * 2 != p.k:
        return None
    return sorted((min(x, y), max(x, y)) for x, y in m)


def public_numbers(p: Problem, asg: np.ndarray, pairs) -> np.ndarray:
    """Deterministic public numbering: Senate districts north to south by
    population-weighted centroid, House districts 2s-1 and 2s inside Senate s."""
    pts = p.units.geometry.representative_point()
    x, y = pts.x.to_numpy(), pts.y.to_numpy()
    act = p.active
    wt = np.bincount(asg[act], weights=p.pop[act], minlength=p.k)
    cx = np.bincount(asg[act], weights=(x * p.pop)[act], minlength=p.k) / wt
    cy = np.bincount(asg[act], weights=(y * p.pop)[act], minlength=p.k) / wt
    labels = np.zeros(p.k, np.int64)
    if pairs is None:
        order = sorted(range(p.k), key=lambda d: (-round(cy[d]), round(cx[d])))
        for i, d in enumerate(order):
            labels[d] = i + 1
        return labels
    pair_order = sorted(pairs, key=lambda pr: (-round((cy[pr[0]] + cy[pr[1]]) / 2), round((cx[pr[0]] + cx[pr[1]]) / 2)))
    for s, (d1, d2) in enumerate(pair_order, start=1):
        first, second = sorted((d1, d2), key=lambda d: (-round(cy[d]), round(cx[d])))
        labels[first], labels[second] = 2 * s - 1, 2 * s
    return labels
