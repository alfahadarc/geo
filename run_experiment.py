#!/usr/bin/env python
"""Command-line runner for the fragmentation pipeline.

Examples
--------
# Single county, ZCTA level only, quick settings -- start here.
python run_experiment.py --fips 18097 --levels zcta --replicates 20 \
    --p-values 0.05:0.50:0.05

# Marion County, full multi-scalar sweep, both reachability scopes.
python run_experiment.py --fips 18097 --levels block bg tract zcta \
    --pmedian 5 --scopes county zone --replicates 100

# A four-archetype comparison, results written per county so it is restartable.
python run_experiment.py --fips 18097 13121 37119 53033 \
    --levels bg tract zcta --replicates 50 --outdir results/archetypes

# Just report the network, no simulation -- useful for sanity-checking a county.
python run_experiment.py --fips 18097 --describe-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tigerfrag import pipeline
from tigerfrag.config import DEFAULT_YEAR, EXAMPLE_COUNTIES, MTFCC_SETS


def parse_p_values(spec: str | None):
    if not spec:
        return None
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        return np.round(np.arange(lo, hi + step / 2, step), 4)
    return np.round(np.array([float(x) for x in spec.split(",")]), 4)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="TIGER/Line road-network fragmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Known county FIPS:\n  "
        + "\n  ".join(f"{k}  {v}" for k, v in EXAMPLE_COUNTIES.items()),
    )
    ap.add_argument("--fips", nargs="+", required=True,
                    help="One or more 5-digit county FIPS codes.")
    ap.add_argument("--levels", nargs="*", default=["bg", "tract", "zcta"],
                    choices=["block", "bg", "tract", "zcta"],
                    help="Census aggregation levels to analyse.")
    ap.add_argument("--pmedian", nargs="*", type=int, default=[5],
                    help="Also build k-median network partitions for these k. 0 to skip.")
    ap.add_argument("--scopes", nargs="+", default=["county"],
                    choices=["county", "zone"],
                    help="Reachability scope. 'county' is physically correct; "
                         "'zone' confines paths inside each zone.")
    ap.add_argument("--protocols", nargs="+", default=["random"],
                    choices=["random", "fitness"])
    ap.add_argument("--units", nargs="+", default=["edge"], choices=["edge", "road"],
                    help="Delete individual edges, or entire named roads.")
    ap.add_argument("--mtfcc", default="drive", choices=sorted(MTFCC_SETS),
                    help="Which road classes form the network.")
    ap.add_argument("--p-values", default=None,
                    help="'lo:hi:step' or comma list. Default 0.01:0.50:0.01.")
    ap.add_argument("--replicates", type=int, default=50)
    ap.add_argument("--detail-p", type=float, nargs="+", default=[0.20],
                    help="Deletion levels at which per-zone detail is kept.")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--cache-dir", default="./tiger_cache")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--describe-only", action="store_true",
                    help="Build and report the network, then stop.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.describe_only:
        rows = []
        for fips in args.fips:
            net = pipeline.build_county_network(
                fips, args.mtfcc, args.year, args.cache_dir)
            rows.append({"fips": str(fips).zfill(5), **net.summary(), **net.meta})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return 0

    pmedian_k = tuple(k for k in args.pmedian if k > 0)
    summary = pipeline.run_many(
        args.fips,
        outdir=args.outdir,
        levels=tuple(args.levels),
        pmedian_k=pmedian_k,
        mtfcc_set=args.mtfcc,
        scopes=tuple(args.scopes),
        protocols=tuple(args.protocols),
        units=tuple(args.units),
        p_values=parse_p_values(args.p_values),
        n_replicates=args.replicates,
        detail_p=tuple(args.detail_p),
        year=args.year,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    if summary.empty:
        print("No counties completed successfully.", file=sys.stderr)
        return 1

    cols = [c for c in ["fips", "level", "scope", "unit", "n_zones", "median_frag",
                        "rank_at_50pct", "pc_second_peak", "pc_giant_half"]
            if c in summary.columns]
    print("\n" + summary[cols].to_string(index=False))

    manifest = Path(args.outdir) / "run_manifest.json"
    manifest.write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"\nResults + manifest in {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
