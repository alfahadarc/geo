"""Smoke test on a synthetic lattice -- runs with no network access.

Builds a grid-plus-cul-de-sacs graph that mimics a gridiron core with a
dendritic fringe, then checks the marooning logic, the percolation threshold
estimator, and the exponent fit behave the way percolation theory says they
should. Run this before you burn an hour downloading TIGER data.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tigerfrag.analysis import cluster_sizes, curve_summary, estimate_pc, fit_powerlaw, rank_curves
from tigerfrag.network import build_network
from tigerfrag.simulate import build_view, run_trial, select_deleted, sweep
from tigerfrag.zones import assign_by_pmedian, assign_by_polygons


def make_lattice(n=40, spacing=100.0, cul_de_sac_frac=0.25, seed=1):
    """n x n grid of streets, with a fraction of edges pruned into dead ends."""
    rng = np.random.default_rng(seed)
    rows = []
    tnid = lambda i, j: i * (n + 1) + j  # noqa: E731

    for i in range(n + 1):
        for j in range(n + 1):
            if j < n:
                rows.append((tnid(i, j), tnid(i, j + 1),
                             LineString([(j * spacing, i * spacing),
                                         ((j + 1) * spacing, i * spacing)])))
            if i < n:
                rows.append((tnid(i, j), tnid(i + 1, j),
                             LineString([(j * spacing, i * spacing),
                                         (j * spacing, (i + 1) * spacing)])))

    gdf = gpd.GeoDataFrame(
        {"TNIDF": [r[0] for r in rows],
         "TNIDT": [r[1] for r in rows],
         "ROADFLG": "Y",
         "MTFCC": "S1400",
         "LINEARID": [f"L{r[0] // 7}" for r in rows],
         "geometry": [r[2] for r in rows]},
        crs="EPSG:5070",
    )
    # Prune some edges up front to create dead ends / lower redundancy.
    keep = rng.random(len(gdf)) > cul_de_sac_frac
    return gdf.loc[keep].reset_index(drop=True)


def make_zone_grid(n_cells, extent, crs="EPSG:5070"):
    step = extent / n_cells
    polys, ids = [], []
    for i in range(n_cells):
        for j in range(n_cells):
            polys.append(box(j * step, i * step, (j + 1) * step, (i + 1) * step))
            ids.append(f"Z{i:02d}{j:02d}")
    return gpd.GeoDataFrame({"GEOID": ids, "geometry": polys}, crs=crs)


def main():
    ok = True
    gdf = make_lattice(n=40)
    net = build_network(gdf, mtfcc_set="drive", projected_crs="EPSG:5070")
    s = net.summary()
    print(f"[net]  {s['n_nodes']} nodes, {s['n_edges']} edges, "
          f"mean degree {s['mean_degree']:.2f}, {s['total_km']:.1f} km")
    assert 2.0 < s["mean_degree"] < 4.1, "planar degree should sit near 3"

    extent = 40 * 100.0
    zas = {
        "coarse": assign_by_polygons(net, make_zone_grid(4, extent), "coarse"),
        "medium": assign_by_polygons(net, make_zone_grid(8, extent), "medium"),
        "fine": assign_by_polygons(net, make_zone_grid(16, extent), "fine"),
        "pmedian5": assign_by_pmedian(net, k=5, seed=0),
    }
    for k, za in zas.items():
        c = za.edge_counts()
        print(f"[zones] {k:9s} {za.n_zones:4d} zones, "
              f"{c.sum()}/{net.n_edges} edges assigned, median {int(np.median(c))} edges/zone")
        assert c.sum() <= net.n_edges

    # --- p=0 sanity: nothing deleted, nothing marooned ------------------------
    view = build_view(net, zas["medium"], "county")
    pz, gl = run_trial(view, np.zeros(net.n_edges, bool), net.length)
    assert pz["n_deleted"].sum() == 0 and pz["n_marooned"].sum() == 0, "p=0 must be clean"
    assert abs(gl["giant_frac"] - 1.0) < 1e-9, "baseline must be one component"
    print("[check] p=0 gives zero fragmentation and a single component  OK")

    # --- p=1 sanity: everything deleted ---------------------------------------
    pz, _ = run_trial(view, np.ones(net.n_edges, bool), net.length)
    assert pz["n_deleted"].sum() == pz["n_total"].sum(), "every counted edge must be deleted"
    print("[check] p=1 deletes every counted edge  OK")

    # --- scope matters --------------------------------------------------------
    rng = np.random.default_rng(0)
    deleted = select_deleted(net, 0.20, rng, "random", "edge")
    for scope in ("county", "zone"):
        v = build_view(net, zas["medium"], scope)
        pz, gl = run_trial(v, deleted, net.length)
        tot = pz["n_total"].sum()
        eff = (pz["n_deleted"].sum() + pz["n_marooned"].sum()) / tot
        print(f"[scope] {scope:7s} p=0.20 -> nominal {pz['n_deleted'].sum()/tot:.3f}, "
              f"effective {eff:.3f}, giant {gl['giant_frac']:.3f}")

    # --- deletion units -------------------------------------------------------
    for unit in ("edge", "road"):
        d = select_deleted(net, 0.20, np.random.default_rng(3), "random", unit)
        v = build_view(net, zas["medium"], "county")
        pz, gl = run_trial(v, d, net.length)
        tot = pz["n_total"].sum()
        eff = (pz["n_deleted"].sum() + pz["n_marooned"].sum()) / tot
        print(f"[unit]  {unit:5s} p=0.20 -> deleted {d.mean():.3f}, "
              f"effective {eff:.3f}, giant {gl['giant_frac']:.3f}")

    # --- fitness protocol runs ------------------------------------------------
    d = select_deleted(net, 0.20, np.random.default_rng(4), "fitness", "edge")
    assert abs(d.mean() - 0.20) < 0.01
    print("[check] fitness-weighted protocol hits the target deletion rate  OK")

    # --- full sweep across levels --------------------------------------------
    ps = np.round(np.arange(0.05, 0.71, 0.05), 2)
    globs, zones = [], []
    for name, za in zas.items():
        g, z = sweep(net, za, p_values=ps, n_replicates=6, scope="county",
                     seed=0, zone_detail_at=(0.20,), label="synthetic")
        globs.append(g)
        zones.append(z)
    global_df = pd.concat(globs, ignore_index=True)
    zone_df = pd.concat(zones, ignore_index=True)
    print(f"[sweep] {len(global_df)} global rows, {len(zone_df)} zone rows")

    # Giant component must decrease monotonically in p (up to noise).
    gm = global_df.groupby("p")["giant_frac"].mean()
    assert gm.iloc[0] > gm.iloc[-1], "giant component should shrink as p rises"
    print("[check] giant component decreases with p  OK")

    pc = estimate_pc(global_df)
    print("\n[p_c estimates]")
    print(pc[["level", "pc_second_peak", "max_second_frac", "pc_giant_half"]].to_string(index=False))

    curves = rank_curves(zone_df, p=0.20)
    summ = curve_summary(curves)
    print("\n[Table 1 equivalent]")
    print(summ.to_string(index=False))

    # Curve collapse: median fragmentation should be close across levels.
    spread = summ["median_frag"].max() - summ["median_frag"].min()
    print(f"\n[check] median fragmentation spread across levels: {spread:.4f}")

    # --- cluster-size power law near threshold --------------------------------
    p_star = float(pc["pc_second_peak"].iloc[0])
    sizes = cluster_sizes(net, p_star, n_replicates=8, seed=0)
    fit = fit_powerlaw(sizes)
    print(f"\n[powerlaw] at p={p_star:.2f}: alpha={fit['alpha']:.3f}, "
          f"xmin={fit['xmin']:.0f}, n_tail={fit['n_tail']}, KS={fit['ks']:.4f}")
    print(f"           2D percolation reference tau = {187/91:.4f}")

    print("\nAll smoke tests passed." if ok else "\nFAILURES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
