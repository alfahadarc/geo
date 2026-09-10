"""End-to-end orchestration: county FIPS in, tidy result tables out."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import cluster_sizes, curve_summary, estimate_pc, fit_powerlaw, rank_curves
from .config import DEFAULT_YEAR
from .data import load_county_boundary, load_layer
from .network import build_network
from .zones import assign_by_pmedian, assign_by_polygons, load_zone_polygons
from .simulate import sweep

log = logging.getLogger(__name__)


def build_county_network(
    fips5: str,
    mtfcc_set: str = "drive",
    year: int = DEFAULT_YEAR,
    cache_dir="./tiger_cache",
    layer: str = "edges",
):
    """Download and build one county's road graph."""
    fips5 = str(fips5).zfill(5)
    raw = load_layer(layer, fips5, year=year, cache_dir=cache_dir)
    net = build_network(raw, mtfcc_set=mtfcc_set)
    log.info("%s network: %s", fips5, net.summary())
    return net


def run_county(
    fips5: str,
    levels=("bg", "tract", "zcta"),
    pmedian_k=(5,),
    mtfcc_set: str = "drive",
    scopes=("county", "zone"),
    protocols=("random",),
    units=("edge",),
    p_values=None,
    n_replicates: int = 50,
    detail_p=(0.20,),
    year: int = DEFAULT_YEAR,
    cache_dir="./tiger_cache",
    outdir: str | Path | None = None,
    seed: int = 0,
):
    """Full sweep for one county across every level x scope x protocol x unit.

    Returns (global_df, zone_df, summary_df). If `outdir` is given, all three
    are written as CSV so a long multi-county run is restartable.
    """
    fips5 = str(fips5).zfill(5)
    p_values = np.round(np.arange(0.01, 0.51, 0.01), 2) if p_values is None else p_values

    net = build_county_network(fips5, mtfcc_set, year, cache_dir)
    boundary = load_county_boundary(fips5, year, cache_dir)

    zas = {}
    for lvl in levels:
        polys = load_zone_polygons(lvl, fips5, boundary, year, cache_dir)
        zas[lvl] = assign_by_polygons(net, polys, lvl)
    for k in pmedian_k:
        zas[f"pmedian{k}"] = assign_by_pmedian(net, k=k, seed=seed)

    globs, zones = [], []
    for lvl, za in zas.items():
        for scope in scopes:
            for protocol in protocols:
                for unit in units:
                    log.info("Sweep: %s | %s | %s | %s | %s", fips5, lvl, scope, protocol, unit)
                    g, z = sweep(
                        net, za, p_values=p_values, n_replicates=n_replicates,
                        scope=scope, protocol=protocol, unit=unit, seed=seed,
                        zone_detail_at=detail_p, label=fips5,
                    )
                    globs.append(g)
                    zones.append(z)

    global_df = pd.concat(globs, ignore_index=True)
    zone_df = pd.concat([z for z in zones if len(z)], ignore_index=True)
    global_df["fips"] = fips5
    zone_df["fips"] = fips5

    curves = rank_curves(zone_df, p=detail_p[0])
    summary = curve_summary(curves).merge(
        estimate_pc(global_df), on=["level", "scope", "protocol", "unit"], how="left"
    )
    summary["fips"] = fips5
    summary = summary.assign(**net.summary())

    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        global_df.to_csv(outdir / f"global_{fips5}.csv", index=False)
        zone_df.to_csv(outdir / f"zones_{fips5}.csv", index=False)
        curves.to_csv(outdir / f"curves_{fips5}.csv", index=False)
        summary.to_csv(outdir / f"summary_{fips5}.csv", index=False)
        log.info("Wrote results for %s to %s", fips5, outdir)

    return global_df, zone_df, summary


def run_many(fips_list, outdir="./results", skip_existing=True, **kwargs):
    """Loop run_county over a list of FIPS codes, one CSV set each.

    Failures are logged and skipped rather than killing the batch -- with 50
    counties you do not want run 37 to lose you the previous 36.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fips in fips_list:
        fips = str(fips).zfill(5)
        target = outdir / f"summary_{fips}.csv"
        if skip_existing and target.exists():
            log.info("Skipping %s (already done)", fips)
            summaries.append(pd.read_csv(target, dtype={"fips": str}))
            continue
        try:
            _, _, s = run_county(fips, outdir=outdir, **kwargs)
            summaries.append(s)
        except Exception as exc:  # noqa: BLE001
            log.error("County %s failed: %s", fips, exc)
    if not summaries:
        return pd.DataFrame()
    allsum = pd.concat(summaries, ignore_index=True)
    allsum.to_csv(outdir / "summary_all_counties.csv", index=False)
    return allsum


def critical_exponent(net, pc: float, n_replicates: int = 20, seed: int = 0) -> dict:
    """Fit the cluster-size exponent tau at the estimated threshold.

    For 2D percolation the universal value is tau = 187/91 ~= 2.055. Getting
    within ~0.1 of that is a genuine confirmation of the paper's universality
    claim; getting 1.4 or 3.0 means something upstream is wrong (usually the
    network is not actually planar because of the geometric-noding fallback,
    or you are not near p_c).
    """
    sizes = cluster_sizes(net, pc, n_replicates=n_replicates, seed=seed)
    fit = fit_powerlaw(sizes)
    fit["p"] = pc
    fit["tau_2d_reference"] = 187 / 91
    return fit
