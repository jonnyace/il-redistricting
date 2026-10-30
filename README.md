# il-redistricting

An open source, deterministic computer program that draws Illinois House (118) and
Senate (59) districts from public data. Same code + same data files + same seed
= the same map, on anyone's computer. No AI runs when the map is drawn.

It was written in response to State Rep. Travis Weaver's proposal
([post](https://x.com/RepTWeaver/status/2116188250444345643)) that Illinois maps be drawn by a
program instead of by politicians. AI can help write the
program, but the program itself must be open source, deterministic and reproducible, and a law
should name the algorithm, the data vintage and the random seed, not "whose AI".


## Results

See [`results/REPORT.md`](results/REPORT.md) for the full comparison. Each folder in `results/`
has a map image, a district table (CSV), district shapes (GeoJSON) and `metrics.json`;
the two drawn plans also have `assignment_vtd.csv` (every 2020 voting precinct and its House and
Senate district), which is the file you would load into any other mapping tool.

Headline numbers from the published run (seed 20261030, 1,000,000 steps per mode):

| | Enacted 2021 | HJRCA 41 mode | Weaver mode |
|---|---|---|---|
| Population range (max minus min) | 0.48% | 1.00% | 0.98% |
| Counties split (of 102) | 62 | 36 | 39 |
| Cities/villages split | 301 | 144 | 142 |
| Townships split | 244 | 79 | 86 |
| Interior district border | 8,504 km | 6,547 km | 6,686 km |
| Polsby-Popper mean / worst | 0.26 / 0.07 | 0.44 / 0.20 | 0.43 / 0.14 |
| R-leaning / toss-up / D-leaning seats | 39 / 16 / 63 | 44 / 20 / 54 | **35 / 38 / 45** (target hit exactly) |
| Hispanic-majority VAP districts (screen) | 11 | 8 | 6 |
| Black-majority VAP districts (screen) | 9 | 8 | 9 |

Both drawn maps are contiguous, nest two House districts in each Senate district, and split far
fewer counties, cities and townships than the enacted map. Both also have **fewer
Hispanic-majority districts** than the enacted map, which is a real Voting Rights Act concern
(see Limitations).

## Two modes, because the two proposals conflict

| | `hjrca41` mode | `weaver` mode |
|---|---|---|
| Based on | [HJRCA 41](https://ilga.gov/ftp/legislation/98/HJRCA/09800HC0041.htm) (98th General Assembly, 2014), a proposed constitutional amendment under which the State Board of Elections would designate a public computer program to draw the maps | Rep. Weaver's 3 points |
| Uses election results? | **No.** The optimizer never reads vote data (a test proves it: scrambling the votes gives the identical map). Partisan numbers are computed afterward, only for the report. | **Yes.** It targets the statewide partisan balance of the last 3 Governor races (2014, 2018, 2022) |
| Partisan target | none | 30% R-leaning / 32% toss-up / 38% D-leaning, which is 35 / 38 / 45 of 118 House seats |

HJRCA 41's text says the program shall not consider incumbents' addresses, political
affiliation, previous election results, or demographics beyond what the law requires. Weaver's
point 1 requires previous election results. You cannot satisfy both, so the program does both and
shows the difference.

"Toss-up" means the district's average two-party margin across the three Governor races is within
5 points. R-leaning is more than 5 points Republican; D-leaning more than 5 points Democratic.

## What the program does (the "algorithm" a law would name)

1. **Data.** 2020 Census precincts (VTDs; 10,082 on land, the 2 Lake Michigan water VTDs are left out) and blocks. Population, voting-age
   population by race and ethnicity from the 2020 PL 94-171 file. Governor results by precinct.
2. **Graph.** Each precinct is a node; precincts that share a border are linked, weighted by the
   length of the shared border in km (equal-area projection EPSG:5070). Lake Michigan water
   precincts are excluded.
3. **First map.** Recursive splitting by random spanning trees: cut the state into two pieces with
   the right population for the number of districts each will hold, then repeat. Edges that cross
   a county or city line are made "expensive" so the cuts prefer to follow those lines.
4. **Improve it (simulated annealing).** One million steps. Each step either recombines two
   neighboring districts (merge, draw a random spanning tree, cut it at a balanced edge) or moves
   one precinct to a neighboring district. Changes that break contiguity or the population band
   are refused. Better scores are always kept; worse scores are sometimes kept early on (the
   "temperature" falls from 40 to 0.05) so the search does not get stuck. The best map seen is the
   output.
5. **Score (lower is better)** =
   total km of district borders inside Illinois
   \+ 10 per extra piece of a split county
   \+ 10 per extra piece of a split city/village
   \+ 5 per extra piece of a split township
   \+ (weaver mode only) 200 per point that the districts miss the 35 / 38 / 45 target.
6. **Hard rules.** Every district is contiguous and within ±0.5% of the ideal population
   (12,812,508 / 118 = 108,581), so the range is at most 1%.
7. **Senate.** Each Senate district is two adjacent House districts, paired to maximize the
   shared border. Numbered north to south.
8. **Randomness.** All of it comes from one NumPy `default_rng(seed)`. The seed is in
   `config/params.toml` (20261030 for the published run). Ties are broken by that generator and
   by fixed sort order, never by clock time or file order.

All weights are in [`config/params.toml`](config/params.toml). Changing them changes the map, which
is exactly why a law should fix them in writing.

## Reproduce it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m ilredistrict all     # fetch, check hashes, build, run both modes, write report
PYTHONPATH=src python -m pytest               # tests, including the same-seed-same-map test
```

`fetch` downloads every input from its public source and refuses to continue if a file's SHA-256
differs from the one recorded in [`config/sources.toml`](config/sources.toml). `run` checks the
hashes again before drawing. Useful options:

```bash
PYTHONPATH=src python -m ilredistrict run --mode hjrca41      # one mode
PYTHONPATH=src python -m ilredistrict run --steps 5000        # quick test run (different map)
```

The published run took about 17 minutes of compute per mode on an 8 core Linux box (Weaver mode 1,022 s; HJRCA 41 mode reported 1,358 s, of which roughly 230 s was the machine being paused mid-run; the program uses one core),
Python 3.13.5, with the exact package versions in `requirements.txt`. Results are bit-for-bit
reproducible with the same versions. Different versions of NumPy, SciPy or GEOS could change
floating point details and therefore the map; a law should pin those too, or require a container
image.

## Data sources

| Data | Source | Vintage | License |
|---|---|---|---|
| Population, VAP by race/ethnicity | Census 2020 PL 94-171 (Illinois) | 2020 Census | public domain |
| Precinct (VTD) and block shapes | Census TIGER/Line 2020 | 2020 | public domain |
| Enacted House map, for comparison | Census 2022 SLDL block equivalency + TIGER 2022 SLDL shapes | 2021 plan | public domain |
| Governor 2018 and 2022 by precinct | Dave's Redistricting `vtd_data`, pinned commit `22cb7f7` | certified results on 2020 VTDs | CC BY-SA 4.0 |
| Governor 2014 by county | OpenElections Illinois, pinned commit `1fa096f` | certified 2014 | see project |

Exact URLs and hashes: [`config/sources.toml`](config/sources.toml).

## Limitations (read these)

* **2014 is estimated by precinct.** No public 2014 Governor file on 2020 precincts was found. The
  program scales each precinct's 2018 Democratic and Republican votes by its county's 2014/2018
  ratio. County totals match 2014 exactly; precinct detail is an estimate.
* **Precincts are the building blocks.** Districts are made of whole 2020 VTDs, so population
  balance is ±0.5% rather than exact, and some city and township splits are forced by precinct
  lines that cross them. Block level drawing would do better on both.
* **The race and VRA numbers are a screen, not an analysis.** "Black" means non-Hispanic Black
  alone; "minority" means everyone except non-Hispanic white alone. Counting majority-minority
  districts does not show Voting Rights Act compliance; that needs racially polarized voting
  analysis, a review of the Gingles factors, and lawyers. The program does not draw VRA districts
  on purpose. In both modes it draws fewer Hispanic-majority districts than the enacted map, which
  in a real process would very likely have to be fixed (see below).
* **Splits are counted on populated territory**, using 2020 Census block codes for counties,
  places (cities/villages) and county subdivisions (townships).
* **Compactness** is Polsby-Popper on the drawn shapes with the Lake Michigan shoreline as drawn by
  TIGER, which lowers the score of lakefront districts in every map equally.
* **Enacted map votes** in precincts it splits are divided by population share, an approximation.
* One seed and one run per mode. Other seeds give other, similar maps; a law should say which seed
  and how it is chosen (for example, drawn publicly from a lottery after the data is frozen).


## Layout

```
config/          sources.toml (URLs + hashes), params.toml (seed, weights, steps)
src/ilredistrict fetch.py, build.py, model.py, optimize.py, metrics.py, report.py, writeup.py, cli.py
tests/           determinism, legality, score and seat-target tests
results/         REPORT.md, enacted/, hjrca41/, weaver/
```

How it was made: the code was written with an AI coding assistant at Jonny Wallace's direction,
then run as ordinary Python. The AI is not part of the program.
