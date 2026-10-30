"""Turn the raw public files into the program's working units.

Units are 2020 Census VTDs (voting tabulation districts, roughly precincts).
Every district is a set of whole VTDs. Block-level census data is used to get
exact population, voting-age population, and exact county / township /
municipality membership, so split counts are exact at block resolution.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .fetch import RAW, ROOT, load_sources

PROCESSED = ROOT / "data" / "processed"
CRS = "EPSG:5070"  # NAD83 / CONUS Albers, equal-area, meters

# Column positions in the 2020 P.L. 94-171 legacy geographic header.
GEO_COLS = {7: "logrecno", 2: "sumlev", 9: "geocode", 14: "county", 17: "cousub",
            18: "cousubcc", 29: "place", 30: "placecc", 77: "vtd", 87: "name"}


def _read_pl(zf: zipfile.ZipFile, member: str, cols: dict[int, str]) -> pd.DataFrame:
    with zf.open(member) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), sep="|", header=None,
                           usecols=list(cols), names=None, dtype=str).rename(columns=cols)


def load_blocks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Block records (summary level 750) with population and VAP, plus county names."""
    src = load_sources()["census_pl"]
    with zipfile.ZipFile(RAW / src["file"]) as zf:
        geo = _read_pl(zf, "ilgeo2020.pl", GEO_COLS)
        # Segment 1: col 4 LOGRECNO, col 5 P0010001 (total population)
        s1 = _read_pl(zf, "il000012020.pl", {4: "logrecno", 5: "pop"})
        # Segment 2: P0030001 at col 5..75, then P4: col 76 VAP total, 77 Hispanic VAP,
        # 81 non-Hispanic Black alone VAP, 80 non-Hispanic White alone VAP
        s2 = _read_pl(zf, "il000022020.pl", {4: "logrecno", 76: "vap", 77: "hvap", 80: "wvap", 81: "bvap"})
    counties = geo[geo.sumlev == "050"][["county", "name"]].copy()
    blocks = geo[geo.sumlev == "750"].drop(columns=["sumlev", "name"])
    blocks = blocks.merge(s1, on="logrecno").merge(s2, on="logrecno")
    for c in ["pop", "vap", "hvap", "wvap", "bvap"]:
        blocks[c] = blocks[c].astype(np.int64)
    blocks["vtd_geoid"] = "17" + blocks["county"] + blocks["vtd"]
    blocks["block_geoid"] = blocks["geocode"]
    return blocks.reset_index(drop=True), counties


def _norm_county(name: str) -> str:
    return name.upper().replace(" COUNTY", "").replace(".", "").replace(" ", "")


def load_elections(vtd_geoids: pd.Series, vtd_county: pd.Series) -> pd.DataFrame:
    """Two-party Governor votes per VTD for 2014 (estimated), 2018 and 2022."""
    src = load_sources()["dra_elections"]
    with zipfile.ZipFile(RAW / src["file"]) as zf, zf.open(src["member"]) as f:
        e = pd.read_csv(f, dtype={"GEOID20": str})
    e = e.set_index("GEOID20")
    out = pd.DataFrame(index=vtd_geoids.values)
    missing = set(out.index) - set(e.index)
    if missing:
        raise SystemExit(f"{len(missing)} VTDs missing from election file, e.g. {sorted(missing)[:3]}")
    for yy in ("18", "22"):
        out[f"d{yy}"] = e.loc[out.index, f"E_{yy}_GOV_Dem"].astype(float).values
        out[f"r{yy}"] = e.loc[out.index, f"E_{yy}_GOV_Rep"].astype(float).values
    out["county"] = vtd_county.values
    return out


def estimate_2014(el: pd.DataFrame, county_names: pd.DataFrame) -> pd.DataFrame:
    """2014 Governor on 2020 VTDs is not published. Estimate it by scaling each
    VTD's 2018 Democratic and Republican votes so that every county's totals
    equal the official 2014 county results (Quinn D, Rauner R). This keeps the
    real 2014 county-by-county result and the 2018 within-county pattern."""
    src = load_sources()["openelections_2014"]
    oe = pd.read_csv(RAW / src["file"])
    g = oe[oe.office == "Governor"].pivot_table(index="county", columns="party", values="votes", aggfunc="sum")
    g.index = [_norm_county(c) for c in g.index]
    names = county_names.assign(key=county_names["name"].map(_norm_county)).set_index("key")["county"]
    g["fips"] = names.reindex(g.index).values
    if g["fips"].isna().any():
        raise SystemExit(f"unmatched 2014 counties: {list(g.index[g['fips'].isna()])}")
    g = g.set_index("fips")
    c18 = el.groupby("county")[["d18", "r18"]].sum()
    scale_d = (g["DEM"] / c18["d18"]).reindex(el["county"]).values
    scale_r = (g["REP"] / c18["r18"]).reindex(el["county"]).values
    el["d14"] = el["d18"] * scale_d
    el["r14"] = el["r18"] * scale_r
    return el


def build() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("reading census blocks (P.L. 94-171)...")
    blocks, county_names = load_blocks()

    print("reading VTD shapes...")
    vtd = gpd.read_file(RAW / load_sources()["tiger_vtd"]["file"])[["GEOID20", "NAME20", "COUNTYFP20", "geometry"]]
    vtd = vtd.sort_values("GEOID20").reset_index(drop=True).to_crs(CRS)
    vtd["geometry"] = vtd.geometry.make_valid()
    idx = pd.Series(np.arange(len(vtd)), index=vtd["GEOID20"])
    if not set(blocks.vtd_geoid).issubset(idx.index):
        raise SystemExit("some blocks reference VTDs not in the shapefile")
    blocks["unit"] = idx.reindex(blocks.vtd_geoid).values

    agg = blocks.groupby("unit")[["pop", "vap", "hvap", "bvap", "wvap"]].sum()
    for c in agg.columns:
        vtd[c] = agg[c].reindex(vtd.index, fill_value=0).astype(np.int64).values
    vtd["county"] = vtd["COUNTYFP20"]

    print("reading Governor results...")
    el = load_elections(vtd["GEOID20"], vtd["county"])
    el = estimate_2014(el, county_names)
    for c in ["d14", "r14", "d18", "r18", "d22", "r22"]:
        vtd[c] = el[c].values

    # Exact membership pairs (unit kind, unit id, VTD, population), from blocks.
    blocks["township"] = np.where(blocks.cousubcc.str.startswith("T"), blocks.county + blocks.cousub, "")
    blocks["municipality"] = np.where(blocks.placecc.str.startswith("C"), blocks.place, "")
    pairs = []
    for kind, col in (("county", "county"), ("township", "township"), ("municipality", "municipality")):
        b = blocks[(blocks[col] != "") & (blocks["pop"] > 0)]
        p = b.groupby([col, "unit"], as_index=False)["pop"].sum().rename(columns={col: "unit_id", "unit": "vtd"})
        p.insert(0, "kind", kind)
        pairs.append(p)
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_csv(PROCESSED / "split_pairs.csv", index=False)

    # Majority county subdivision / place per VTD (for the tree-drawing bias only).
    for col in ("township", "municipality"):
        m = (blocks[blocks[col] != ""].groupby(["unit", col])["pop"].sum().reset_index()
             .sort_values(["unit", "pop", col], ascending=[True, False, True]).drop_duplicates("unit"))
        vtd[col] = m.set_index("unit")[col].reindex(vtd.index).fillna("").values

    # Enacted 2021 House map (2022 election): block -> district, aggregated per VTD.
    s = load_sources()["enacted_sldl_bef"]
    with zipfile.ZipFile(RAW / s["file"]) as zf, zf.open(s["member"]) as f:
        bef = pd.read_csv(f, dtype=str)
    blocks = blocks.merge(bef.rename(columns={"GEOID": "block_geoid", "SLDLST": "enacted"}), on="block_geoid", how="left")
    if blocks["enacted"].isna().any():
        raise SystemExit("blocks missing from enacted House block equivalency file")
    enacted = blocks.groupby(["unit", "enacted"], as_index=False)[["pop", "vap", "hvap", "bvap", "wvap"]].sum()
    enacted = enacted.rename(columns={"unit": "vtd"})
    enacted.to_csv(PROCESSED / "enacted_pairs.csv", index=False)
    # Enacted map's county / township / municipality pieces, same block-level rule as ours.
    esplits = []
    for kind, col in (("county", "county"), ("township", "township"), ("municipality", "municipality")):
        b = blocks[(blocks[col] != "") & (blocks["pop"] > 0)]
        p = b.groupby([col, "enacted"], as_index=False)["pop"].sum().rename(columns={col: "unit_id"})
        p.insert(0, "kind", kind)
        esplits.append(p)
    pd.concat(esplits, ignore_index=True).to_csv(PROCESSED / "enacted_splits.csv", index=False)

    print("building adjacency graph (rook)...")
    graph = adjacency(vtd)
    vtd.to_file(PROCESSED / "units.gpkg", driver="GPKG")
    with open(PROCESSED / "graph.json", "w") as f:
        json.dump(nx.node_link_data(graph, edges="edges"), f)
    print(f"built {len(vtd)} VTDs, {graph.number_of_edges()} adjacencies, population {int(vtd['pop'].sum()):,}")


def adjacency(vtd: gpd.GeoDataFrame) -> nx.Graph:
    """Rook adjacency (shared border of positive length), with the shared
    border length in meters on each edge and the outer-state border length on
    each node. Disconnected pieces (islands) are linked to their nearest VTD."""
    geoms = vtd.geometry.values
    tree = gpd.GeoSeries(geoms).sindex
    left, right = tree.query(geoms, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    g = nx.Graph()
    g.add_nodes_from(range(len(vtd)))
    shared = np.zeros(len(vtd))
    for i, j in zip(left.tolist(), right.tolist()):
        length = geoms[i].boundary.intersection(geoms[j].boundary).length
        if length > 1.0:  # meters; corner touches are not adjacency
            g.add_edge(i, j, shared_perim=float(length))
            shared[i] += length
            shared[j] += length
    perim = vtd.geometry.length.values
    area = vtd.geometry.area.values
    for n in g.nodes:
        g.nodes[n]["perimeter"] = float(perim[n])
        g.nodes[n]["boundary_perim"] = float(max(perim[n] - shared[n], 0.0))
        g.nodes[n]["area"] = float(area[n])
    # Link islands deterministically to the nearest node of the largest component.
    comps = sorted(nx.connected_components(g), key=lambda c: (-len(c), min(c)))
    main = comps[0]
    cent = vtd.geometry.representative_point()
    main_idx = np.array(sorted(main))
    for comp in comps[1:]:
        best = None
        for n in sorted(comp):
            d = cent.iloc[main_idx].distance(cent.iloc[n]).values
            k = int(np.argmin(d))
            if best is None or d[k] < best[0]:
                best = (float(d[k]), n, int(main_idx[k]))
        g.add_edge(best[1], best[2], shared_perim=0.0, island_link=True)
        main_idx = np.array(sorted(set(main_idx.tolist()) | comp))
    return g
