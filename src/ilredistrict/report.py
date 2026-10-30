"""Write the outputs: assignment CSV, GeoJSON, map PNG, metrics JSON, report."""
from __future__ import annotations

import json
import platform
import time

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import __version__  # noqa: E402
from .fetch import ROOT, fetch, load_sources  # noqa: E402
from .metrics import evaluate_enacted, evaluate_plan, public_numbers, senate_pairs  # noqa: E402
from .model import load_params, load_problem, seat_targets  # noqa: E402
from .optimize import solve  # noqa: E402

RESULTS = ROOT / "results"
LEAN_COLORS = {"R": "#d7301f", "toss-up": "#b9a6d1", "D": "#2b6cb0"}
TABLE_COLS = ["district", "pop", "deviation_pct", "polsby_popper", "perimeter_km", "area_km2",
              "hispanic_vap_pct", "black_vap_pct", "minority_vap_pct", "margin_gov2014",
              "margin_gov2018", "margin_gov2022", "margin_avg", "lean"]


def _colors(t: gpd.GeoDataFrame) -> list:
    G = nx.Graph()
    G.add_nodes_from(range(len(t)))
    sidx = t.sindex
    for i, geom in enumerate(t.geometry):
        for j in sidx.query(geom, predicate="touches"):
            if j > i:
                G.add_edge(i, int(j))
    col = nx.greedy_color(G, strategy="largest_first")
    pal = plt.get_cmap("Set3").colors
    return [pal[col[i] % len(pal)] for i in range(len(t))]


def draw_map(t: gpd.GeoDataFrame, title: str, path, zoom) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    t.plot(ax=axes[0], color=_colors(t), edgecolor="black", linewidth=0.3)
    axes[0].set_title("House districts")
    t.plot(ax=axes[1], color=[LEAN_COLORS[x] for x in t["lean"]], edgecolor="white", linewidth=0.2)
    axes[1].set_title("Lean: avg two-party Governor margin 2014/2018/2022")
    x0, y0, x1, y1 = zoom
    t.plot(ax=axes[2], color=[LEAN_COLORS[x] for x in t["lean"]], edgecolor="white", linewidth=0.5)
    for _, r in t.iterrows():
        pt = r.geometry.representative_point()
        if x0 <= pt.x <= x1 and y0 <= pt.y <= y1:
            axes[2].annotate(str(r["district"]), (pt.x, pt.y), fontsize=6, ha="center", va="center")
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y0, y1)
    axes[2].set_title("Cook and DuPage counties (district numbers)")
    for ax in axes:
        ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LEAN_COLORS.values()]
    fig.legend(handles, ["R-leaning", "toss-up", "D-leaning"], loc="lower center", ncol=3)
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_plan_outputs(p, out, t, summary, extra: dict, asg=None, labels=None, senate=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tab = t[TABLE_COLS].copy()
    if senate is not None:
        tab.insert(1, "senate_district", [senate[d] for d in tab["district"]])
    tab.round(4).to_csv(out / "districts.csv", index=False)
    gj = t[TABLE_COLS + ["geometry"]].copy()
    if senate is not None:
        gj["senate_district"] = [senate[d] for d in gj["district"]]
    gj.round(4).to_crs("EPSG:4326").to_file(out / "districts.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    if asg is not None:
        house = np.where(p.active, labels[np.maximum(asg, 0)], 0)
        a = pd.DataFrame({"vtd_geoid20": p.units["GEOID20"], "house_district": house})
        if senate is not None:
            a["senate_district"] = [senate.get(h, 0) for h in house]
        a.loc[~p.active, ["house_district"] + (["senate_district"] if senate is not None else [])] = 0
        a.to_csv(out / "assignment_vtd.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump({"summary": summary, **extra}, f, indent=2, sort_keys=True)


def run(modes=("hjrca41", "weaver"), params_path=None, steps=None) -> None:
    params = load_params(params_path)
    if steps is not None:
        params["search"]["steps"] = int(steps)
    p = load_problem(params)
    RESULTS.mkdir(exist_ok=True)

    fetch(record=False)  # re-check every input file's SHA-256 before using derived data
    zoom = tuple(p.units[p.units["county"].isin(["031", "043"]) & p.active].total_bounds)
    t0 = time.time()
    te, se = evaluate_enacted(p)
    write_plan_outputs(p, RESULTS / "enacted", te, se, {"plan": "Enacted 2021 Illinois House map (Census 2022 SLDL)"})
    draw_map(te, "Enacted Illinois House map (2021), scored by this program", RESULTS / "enacted" / "map.png", zoom)
    print(f"enacted map scored in {time.time() - t0:.0f}s")

    for mode in modes:
        plan, stats = solve(p, mode)
        pairs = senate_pairs(p, plan.asg)
        labels = public_numbers(p, plan.asg, pairs)
        senate = None
        if pairs is not None:
            senate = {}
            for s_ in range(1, len(pairs) + 1):
                senate[2 * s_ - 1] = s_
                senate[2 * s_] = s_
        t, s = evaluate_plan(p, plan.asg, labels)
        s["senate_nesting"] = "59 Senate districts, each two touching House districts" if pairs else "no perfect pairing found"
        extra = {
            "mode": mode,
            "seed": params["plan"]["seed"],
            "params": params,
            "search": stats,
            "score_parts": plan.score_parts(),
            "program_version": __version__,
            "python": platform.python_version(),
            "data_sha256": {k: v["sha256"] for k, v in load_sources().items()},
        }
        if mode == "weaver":
            w = params["weaver"]
            extra["weaver_target_seats"] = dict(zip(("R", "toss-up", "D"), seat_targets(
                p.k, (w["share_r_leaning"], w["share_tossup"], w["share_d_leaning"]))))
        out = RESULTS / mode
        write_plan_outputs(p, out, t, s, extra, plan.asg, labels, senate)
        title = {"hjrca41": "HJRCA 41 mode (no election or incumbent data used)",
                 "weaver": "Weaver mode (partisan balance target from 2014/2018/2022 Governor races)"}[mode]
        draw_map(t, f"Illinois House, {title}, seed {params['plan']['seed']}", out / "map.png", zoom)
        print(f"[{mode}] done: {json.dumps(s)}")
