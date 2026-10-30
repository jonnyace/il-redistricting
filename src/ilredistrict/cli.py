"""Command line: python -m ilredistrict {fetch,build,run,all}."""
from __future__ import annotations

import argparse


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ilredistrict", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the public inputs and check their SHA-256 hashes")
    f.add_argument("--record", action="store_true", help="write hashes for files that have none yet")
    sub.add_parser("build", help="turn raw inputs into data/processed")
    r = sub.add_parser("run", help="draw plans and write results/")
    r.add_argument("--mode", choices=["hjrca41", "weaver", "both"], default="both")
    r.add_argument("--params", default=None, help="path to a params TOML (default config/params.toml)")
    r.add_argument("--steps", type=int, default=None, help="override search.steps (for quick tests)")
    sub.add_parser("all", help="fetch + build + run both modes + write report")
    sub.add_parser("report", help="rewrite results/REPORT.md from results/*/metrics.json")
    a = ap.parse_args(argv)

    if a.cmd in ("fetch", "all"):
        from .fetch import fetch
        fetch(record=getattr(a, "record", False))
    if a.cmd in ("build", "all"):
        from .build import build
        build()
    if a.cmd in ("run", "all"):
        from .report import run
        modes = ("hjrca41", "weaver") if getattr(a, "mode", "both") == "both" else (a.mode,)
        run(modes, getattr(a, "params", None), getattr(a, "steps", None))
    if a.cmd in ("report", "all", "run"):
        from .writeup import write_report
        write_report()
