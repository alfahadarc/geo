"""Multi-scalar spatial aggregation: assign road edges to zones, pick anchors.

This is the Python equivalent of TransCAD's "segment the map by zone, use the
zone centroid as the median". Two families of partition are supported:

  * Census geography  -- block / bg / tract / zcta polygons from TIGER.
  * Network p-median  -- k medians solved on the network itself (the Figure 5
                         "five zones optimised for minimum distance travelled"
                         case), no polygons involved.

Both produce the same `ZoneAssignment` object, so the simulation code doesn't
care which one you used and you can sweep across all of them in one run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import DEFAULT_YEAR, GEOID_CANDIDATES
from .data import load_layer
from .network import RoadNetwork

log = logging.getLogger(__name__)

CENSUS_LEVELS = ("block", "bg", "tract", "zcta")


@dataclass
class ZoneAssignment:
    level: str
    zone_ids: np.ndarray        # (Z,) label per zone (GEOID string or int)
    zone_of_edge: np.ndarray    # (M,) int32 zone index per edge, -1 = unassigned
    anchor_node: np.ndarray     # (Z,) int32 node index acting as that zone's median
    geometry: gpd.GeoDataFrame | None = None

    @property
    def n_zones(self) -> int:
        return len(self.zone_ids)

    def edge_counts(self) -> np.ndarray:
        z = self.zone_of_edge
        return np.bincount(z[z >= 0], minlength=self.n_zones)


def _geoid_column(gdf: gpd.GeoDataFrame) -> str:
    for c in GEOID_CANDIDATES:
        if c in gdf.columns:
            return c
    raise KeyError(f"No GEOID column found. Have: {list(gdf.columns)}")


def load_zone_polygons(
    level: str,
    fips5: str,
    county_boundary: gpd.GeoDataFrame,
    year: int = DEFAULT_YEAR,
    cache_dir="./tiger_cache",
) -> gpd.GeoDataFrame:
    """Load one aggregation level and clip it to the county.

    ZCTAs are a national file and do NOT nest inside counties -- a ZCTA can
    straddle a county line. We select ZCTAs that intersect the county rather
    than filtering on a FIPS field, because no such field exists.
    """
    fips5 = str(fips5).zfill(5)
    bounds = tuple(county_boundary.to_crs("EPSG:4269").total_bounds)

    if level == "zcta":
        gdf = load_layer("zcta", year=year, cache_dir=cache_dir, bbox=bounds)
        gdf = gpd.sjoin(
            gdf, county_boundary.to_crs(gdf.crs)[["geometry"]],
            how="inner", predicate="intersects",
        ).drop(columns=["index_right"]).drop_duplicates(subset=_geoid_column(gdf))
    elif level in ("block", "bg", "tract"):
        gdf = load_layer(level, fips5, year=year, cache_dir=cache_dir)
        if "COUNTYFP" in gdf.columns and "STATEFP" in gdf.columns:
            gdf = gdf[(gdf["STATEFP"] == fips5[:2]) & (gdf["COUNTYFP"] == fips5[2:])]
        elif "COUNTYFP20" in gdf.columns:
            gdf = gdf[(gdf["STATEFP20"] == fips5[:2]) & (gdf["COUNTYFP20"] == fips5[2:])]
    else:
        raise ValueError(f"Unknown census level {level!r}. Use one of {CENSUS_LEVELS}")

    gdf = gdf.reset_index(drop=True)
    log.info("Level %s: %d zones", level, len(gdf))
    return gdf


def assign_by_polygons(
    net: RoadNetwork,
    polygons: gpd.GeoDataFrame,
    level: str,
    drop_empty: bool = True,
    min_edges: int = 5,
) -> ZoneAssignment:
    """Assign each edge to the zone containing its midpoint, then pick anchors.

    Midpoint assignment (rather than intersection area) means every edge lands
    in exactly one zone, so `sum(edges per zone) == total edges` and your
    fragmentation denominators are clean. Edges that straddle a zone boundary
    are assigned to whichever zone owns their middle -- an arbitrary but
    consistent rule, and the same one TransCAD's overlay effectively applies.

    We join with `intersects`, not `within`, and then keep the first match.
    This is not a stylistic choice: census blocks are *delineated along road
    centerlines*, so a large share of edge midpoints land exactly on a block
    or block-group boundary. `within` returns False for a point on a polygon
    edge, which would silently discard those edges from every denominator and
    quietly deflate your fragmentation rates at the finer aggregation levels.
    """
    geoid = _geoid_column(polygons)
    polys = polygons.to_crs(net.crs)[[geoid, "geometry"]].reset_index(drop=True)

    mids = gpd.GeoDataFrame(
        geometry=net.edges.geometry.interpolate(0.5, normalized=True), crs=net.crs
    )
    joined = gpd.sjoin(mids, polys, how="left", predicate="intersects")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(mids.index)

    zone_of_edge = joined["index_right"].to_numpy()
    zone_of_edge = np.where(pd.isna(zone_of_edge), -1, zone_of_edge).astype(np.int32)

    unassigned = int((zone_of_edge < 0).sum())
    if unassigned:
        log.info("%d edges (%.1f%%) fell outside all %s polygons",
                 unassigned, 100 * unassigned / net.n_edges, level)

    zone_ids = polys[geoid].to_numpy()
    centroids = np.c_[polys.geometry.centroid.x, polys.geometry.centroid.y]
    anchors = _anchors_from_centroids(net, zone_of_edge, len(polys), centroids)

    za = ZoneAssignment(level, zone_ids, zone_of_edge, anchors, polys)
    if drop_empty:
        za = _drop_small_zones(za, min_edges)
    return za


def _anchors_from_centroids(net, zone_of_edge, n_zones, centroids) -> np.ndarray:
    """For each zone, the node incident to that zone's edges nearest its centroid.

    Restricting candidates to *incident* nodes matters: the geometric centroid
    of an L-shaped or water-split ZCTA can land on a node that belongs to a
    different zone entirely, which would silently make the zone look connected
    when it isn't.
    """
    anchors = np.full(n_zones, -1, dtype=np.int32)
    order = np.argsort(zone_of_edge, kind="stable")
    sorted_zones = zone_of_edge[order]
    starts = np.searchsorted(sorted_zones, np.arange(n_zones), "left")
    ends = np.searchsorted(sorted_zones, np.arange(n_zones), "right")

    for z in range(n_zones):
        eidx = order[starts[z]:ends[z]]
        if eidx.size == 0:
            continue
        cand = np.unique(np.concatenate([net.src[eidx], net.dst[eidx]]))
        d = np.hypot(*(net.node_xy[cand] - centroids[z]).T)
        anchors[z] = cand[int(d.argmin())]
    return anchors


def _drop_small_zones(za: ZoneAssignment, min_edges: int) -> ZoneAssignment:
    counts = za.edge_counts()
    keep = (counts >= min_edges) & (za.anchor_node >= 0)
    if keep.all():
        return za
    log.info("Dropping %d %s zones with <%d edges", int((~keep).sum()), za.level, min_edges)
    remap = np.full(za.n_zones, -1, dtype=np.int32)
    remap[keep] = np.arange(keep.sum(), dtype=np.int32)
    new_zoe = np.where(za.zone_of_edge >= 0, remap[za.zone_of_edge], -1).astype(np.int32)
    geom = za.geometry.loc[keep].reset_index(drop=True) if za.geometry is not None else None
    return ZoneAssignment(za.level, za.zone_ids[keep], new_zoe, za.anchor_node[keep], geom)


def assign_by_pmedian(
    net: RoadNetwork, k: int = 5, seed: int = 0, n_iter: int = 60
) -> ZoneAssignment:
    """Partition the network into k p-median service zones (Figure 5 of the paper).

    Lloyd's algorithm on node coordinates, with each centre snapped back to a
    real network node. This is a Euclidean approximation to the true network
    p-median. It is fast, deterministic under a fixed seed, and more than good
    enough for a fragmentation partition -- but say so in your write-up rather
    than calling it an exact p-median solution. If you need the real thing,
    `spopt.locate.PMedian` will solve it against a network distance matrix.
    """
    rng = np.random.default_rng(seed)
    mids = np.c_[
        net.edges.geometry.interpolate(0.5, normalized=True).x,
        net.edges.geometry.interpolate(0.5, normalized=True).y,
    ]
    centres = mids[rng.choice(len(mids), size=k, replace=False)]

    for _ in range(n_iter):
        lab = cKDTree(centres).query(mids)[1]
        new = np.array([
            mids[lab == j].mean(axis=0) if np.any(lab == j) else centres[j]
            for j in range(k)
        ])
        if np.allclose(new, centres):
            break
        centres = new

    zone_of_edge = cKDTree(centres).query(mids)[1].astype(np.int32)
    anchors = _anchors_from_centroids(net, zone_of_edge, k, centres)
    return ZoneAssignment(
        f"pmedian{k}", np.arange(k), zone_of_edge, anchors, None
    )


def build_all_levels(
    net: RoadNetwork,
    fips5: str,
    county_boundary: gpd.GeoDataFrame,
    levels=("block", "bg", "tract", "zcta"),
    pmedian_k=(5,),
    year: int = DEFAULT_YEAR,
    cache_dir="./tiger_cache",
) -> dict[str, ZoneAssignment]:
    """Build every aggregation level for one county in one pass."""
    out: dict[str, ZoneAssignment] = {}
    for lvl in levels:
        polys = load_zone_polygons(lvl, fips5, county_boundary, year, cache_dir)
        out[lvl] = assign_by_polygons(net, polys, lvl)
    for k in pmedian_k:
        out[f"pmedian{k}"] = assign_by_pmedian(net, k=k)
    return out
