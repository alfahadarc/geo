#!/usr/bin/env python
"""Sorted per-zone marooned-rate curve with a power-law fit (TransCAD-style).

Reproduces the hand analysis: for each zone take marooned count and road count,
form  score = marooned * 1000 / road_count,  sort LOW -> HIGH, plot score against
rank on linear axes (the curve rises slowly then shoots up, so it looks like an
exponential), and fit a power law  score = a * rank**b  to it.

Input : results/zones_<fips>.csv   (per-zone detail written at --detail-p, default p=0.20)
Usage : python plot_powerlaw.py --fips 18097 --level tract
        python plot_powerlaw.py --fips 18097 18001 18003 --level tract
        python plot_powerlaw.py --all --level tract          # every zones_*.csv in resultsdir
        python plot_powerlaw.py --all --level tract --overlay # + one combined figure

With more than one FIPS a summary table of the fit parameters is written to
results/plots/powerlaw_summary_<level><suffix>.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def discover_fips(resultsdir: Path) -> list[str]:
    """Every <fips> for which resultsdir/zones_<fips>.csv exists, sorted."""
    out = []
    for f in sorted(resultsdir.glob("zones_*.csv")):
        m = re.fullmatch(r"zones_(.+)", f.stem)
        if m:
            out.append(m.group(1))
    return out


def process_one(fips: str, args):
    """Build the sorted curve + power-law fit for one FIPS.

    Writes the per-FIPS PNG and CSV. Returns a dict of fit stats (plus the x/y
    arrays for an optional overlay), or None if this FIPS could not be processed.
    """
    src = Path(args.resultsdir) / f"zones_{fips}.csv"
    if not src.exists():
        print(f"[{fips}] SKIP -- {src} not found")
        return None
    raw = pd.read_csv(src)
    df = raw[raw["level"] == args.level].copy()
    if df.empty:
        print(f"[{fips}] SKIP -- no rows with level == {args.level!r} "
              f"(levels present: {sorted(raw['level'].unique())})")
        return None
    if "scope" in df.columns:
        if args.scope not in set(df["scope"]):
            print(f"[{fips}] SKIP -- level {args.level!r} has scopes "
                  f"{sorted(df['scope'].unique())}, not --scope {args.scope!r}")
            return None
        df = df[df["scope"] == args.scope].copy()

    p = df["p"].iloc[0]

    # --- the professor's score ----------------------------------------------
    # His metric is MAROONED alone (survived deletion but cut off from the zone
    # anchor), not deleted+marooned. Use frag_marooned = n_marooned / n_total.
    # Older result files only have frag_effective (deleted+marooned lumped) --
    # fall back to it with a warning so the script still runs.
    if "frag_marooned" in df.columns:
        rate_col = "frag_marooned"
    else:
        rate_col = "frag_effective"
        print(f"[{fips}] WARNING: frag_marooned not in this CSV -- falling back "
              "to frag_effective (deleted+marooned). Re-run run_experiment.py to "
              "get the split-out marooned rate.")

    df["marooned"] = df[rate_col] * df["n_edges"]
    df["score"] = df["marooned"] * 1000.0 / df["n_edges"]        # == rate_col * 1000

    # --- sort LOW -> HIGH, then rank --------------------------------------
    df = df.sort_values("score", ascending=True).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["norm_rank"] = df["rank"] / len(df)

    xcol = "rank" if args.xaxis == "rank" else "norm_rank"
    x = df[xcol].to_numpy()
    y = df["score"].to_numpy()

    # --- power-law fit:  y = a * x**b   (linear fit of log y on log x) ------
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        print(f"[{fips}] SKIP -- fewer than 2 zones with positive score")
        return None
    b, log_a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    a = np.exp(log_a)
    yhat = a * x ** b
    ss_res = np.sum((y[m] - yhat[m]) ** 2)
    ss_tot = np.sum((y[m] - y[m].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    # --- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, "o", ms=4, alpha=0.6, label="zones (sorted low → high)")
    xs = np.linspace(x[m].min(), x.max(), 400)
    ax.plot(xs, a * xs ** b, "r-", lw=2,
            label=rf"power-law fit: score = {a:.3g}$\cdot$rank$^{{{b:.2f}}}$  ($R^2$={r2:.3f})")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("Rank of zone (low → high marooned rate)"
                  if args.xaxis == "rank" else
                  "Normalized rank of zone (0 → 1)")
    ax.set_ylabel(f"Marooned roads per 1000  ({rate_col} $\\times$ 1000)")
    ax.set_title(f"FIPS {fips}  ·  {args.level}  ·  scope={args.scope}  ·  "
                 f"p={p}  ·  n={len(df)} zones")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()

    suffix = "" if args.scope == "county" else f"_{args.scope}"
    out_png = Path(args.outdir) / f"powerlaw_{fips}_{args.level}{suffix}.png"
    out_csv = Path(args.outdir) / f"powerlaw_{fips}_{args.level}{suffix}.csv"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    keep = [c for c in ["zone_id", "n_edges", "frag_nominal", "frag_marooned",
                        "frag_effective", "marooned", "score", "rank", "norm_rank"]
            if c in df.columns]
    df[keep].to_csv(out_csv, index=False)

    print(f"[{fips}] n={len(df):4d}  score = {a:.4g} * {xcol}**{b:.4f}  "
          f"R^2={r2:.4f}  -> {out_png.name}")

    return {"fips": fips, "n_zones": len(df), "p": p, "rate_col": rate_col,
            "a": a, "b": b, "r2": r2, "x": x, "y": y, "xcol": xcol}


def overlay_plot(results: list[dict], args):
    """One combined figure: every FIPS' power-law fit line (normalized rank)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(sorted(results, key=lambda d: d["b"])):
        x, y = r["x"], r["y"]
        m = (x > 0) & (y > 0)
        # rescale rank to (0, 1] so curves of different N are comparable
        xn = x / x.max()
        color = cmap(i / max(len(results) - 1, 1))
        ax.plot(xn[m], y[m], ".", ms=3, alpha=0.25, color=color)
        xs = np.linspace(xn[m].min(), 1.0, 200)
        a, b = r["a"] * (x.max() ** r["b"]), r["b"]   # a in normalized-rank units
        ax.plot(xs, a * xs ** b, "-", lw=1.5, color=color,
                label=f"{r['fips']} (b={b:.2f}, R²={r['r2']:.2f})")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("Normalized rank of zone (0 → 1)")
    ax.set_ylabel("Marooned roads per 1000")
    ax.set_title(f"Power-law fits · {args.level} · scope={args.scope} · "
                 f"{len(results)} FIPS")
    ax.grid(True, ls=":", alpha=0.4)
    if len(results) <= 20:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    suffix = "" if args.scope == "county" else f"_{args.scope}"
    out_png = Path(args.outdir) / f"powerlaw_overlay_{args.level}{suffix}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fips", nargs="+", default=["18097"],
                    help="One or more county FIPS codes (space separated).")
    ap.add_argument("--all", action="store_true",
                    help="Process every results/zones_*.csv (overrides --fips).")
    ap.add_argument("--level", default="tract",
                    help="Which zoning level to use (tract, zcta, bg, block, pmedian5, ...).")
    ap.add_argument("--scope", default="county",
                    help="Reachability scope to keep (county or zone). A run made "
                         "with --scopes county zone puts both in the CSV; without "
                         "this filter every zone would be counted twice.")
    ap.add_argument("--resultsdir", default="results")
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--xaxis", choices=["rank", "normrank"], default="rank",
                    help="rank = 1..N ; normrank = rank rescaled to (0, 1].")
    ap.add_argument("--logy", action="store_true",
                    help="Use a log y-axis (a power law then looks like a straight line).")
    ap.add_argument("--overlay", action="store_true",
                    help="Also write one combined figure with every FIPS' fit line.")
    args = ap.parse_args(argv)

    if args.all:
        fips_list = discover_fips(Path(args.resultsdir))
        if not fips_list:
            raise SystemExit(f"No zones_*.csv found in {args.resultsdir}")
    else:
        fips_list = args.fips

    results = []
    for fips in fips_list:
        try:
            r = process_one(fips, args)
        except Exception as exc:                       # keep the batch going
            print(f"[{fips}] ERROR -- {type(exc).__name__}: {exc}")
            r = None
        if r is not None:
            results.append(r)

    if not results:
        raise SystemExit("Nothing processed.")

    suffix = "" if args.scope == "county" else f"_{args.scope}"

    if len(results) > 1:
        summary = pd.DataFrame([{k: r[k] for k in
                                 ("fips", "n_zones", "p", "rate_col", "a", "b", "r2")}
                                for r in results])
        out_sum = Path(args.outdir) / f"powerlaw_summary_{args.level}{suffix}.csv"
        summary.to_csv(out_sum, index=False)
        print(f"\n{len(results)}/{len(fips_list)} FIPS processed")
        print(f"wrote {out_sum}")
        print(summary.to_string(index=False,
                                formatters={"a": "{:.4g}".format,
                                            "b": "{:.4f}".format,
                                            "r2": "{:.4f}".format}))

    if args.overlay:
        overlay_plot(results, args)


if __name__ == "__main__":
    main()


# plot_powerlaw.py --fips 18097 --level tract
# plot_powerlaw.py --all --level tract --overlay
