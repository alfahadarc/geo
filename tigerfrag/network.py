"""Build a road graph from TIGER/Line EDGES.

The key move: TIGER's EDGES layer carries TNIDF and TNIDT -- the Census's own
"from node" and "to node" identifiers. Two edges meet at an intersection if and
only if they share a TNID. That means you get exact topology for free and never
have to snap coordinates with a tolerance, which is where most hand-rolled
street-network pipelines quietly go wrong (a 0.5 m tolerance welds two roads
that pass on a grade separation; a 0.05 m tolerance leaves a T-intersection
unjoined, and either way your connectivity results are garbage).

A geometric fallback is included for the ROADS layer, which has no node IDs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import DEFAULT_PROJECTED_CRS, MTFCC_SETS

log = logging.getLogger(__name__)


@dataclass
class RoadNetwork:
    """Array-based road network. Deliberately not a NetworkX object.

    Every simulation replicate is just a boolean mask over `src`/`dst` plus one
    call to scipy's connected_components, which is 50-100x faster than
    rebuilding a NetworkX graph per trial. With 1,000 bootstrap replicates x 50
    deletion levels x 50 counties, that difference is the whole project.
    """

    src: np.ndarray            # int32 (M,) node index of edge start
    dst: np.ndarray            # int32 (M,) node index of edge end
    length: np.ndarray         # float64 (M,) metres
    n_nodes: int
    node_xy: np.ndarray        # float64 (N, 2) projected coordinates
    edges: gpd.GeoDataFrame    # M rows, aligned with src/dst, carries TLID etc.
    crs: str = DEFAULT_PROJECTED_CRS
    meta: dict = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def degree(self) -> np.ndarray:
        return np.bincount(
            np.concatenate([self.src, self.dst]), minlength=self.n_nodes
        )

    def summary(self) -> dict:
        deg = self.degree()
        nz = deg[deg > 0]
        return {
            "n_nodes": int((deg > 0).sum()),
            "n_edges": int(self.n_edges),
            "mean_degree": float(nz.mean()) if nz.size else 0.0,
            "total_km": float(self.length.sum() / 1000.0),
            **{f"deg_{k}_frac": float((nz == k).mean()) for k in (1, 2, 3, 4, 5)},
        }


def _road_mask(gdf: gpd.GeoDataFrame, mtfcc_set: str) -> pd.Series:
    keep = pd.Series(True, index=gdf.index)
    if "ROADFLG" in gdf.columns:
        keep &= gdf["ROADFLG"].astype(str).str.upper().eq("Y")
    codes = MTFCC_SETS.get(mtfcc_set, MTFCC_SETS["drive"])
    if codes is not None and "MTFCC" in gdf.columns:
        keep &= gdf["MTFCC"].isin(codes)
    return keep


def _node_index_from_tnid(gdf: gpd.GeoDataFrame):
    """Factorise TNIDF/TNIDT into a dense 0..N-1 node index."""
    f = pd.to_numeric(gdf["TNIDF"], errors="coerce")
    t = pd.to_numeric(gdf["TNIDT"], errors="coerce")
    ok = f.notna() & t.notna()
    if not ok.all():
        log.warning("Dropping %d edges with missing TNIDF/TNIDT", int((~ok).sum()))
        gdf = gdf.loc[ok].copy()
        f, t = f.loc[ok], t.loc[ok]
    codes, _ = pd.factorize(np.concatenate([f.values, t.values]))
    m = len(gdf)
    return gdf, codes[:m].astype(np.int32), codes[m:].astype(np.int32), int(codes.max()) + 1


def _node_index_from_geometry(gdf: gpd.GeoDataFrame, tol: float = 1.0):
    """Fallback noding for layers without TNIDs: snap endpoints to a `tol`-metre grid.

    Only use this on the ROADS layer. It cannot distinguish an intersection from
    an overpass, so it will over-connect grade-separated crossings.
    """
    coords = np.array(
        [(g.coords[0], g.coords[-1]) for g in gdf.geometry], dtype=float
    )  # (M, 2, 2)
    snapped = np.round(coords / tol).astype(np.int64)
    flat = snapped.reshape(-1, 2)
    _, codes = np.unique(flat, axis=0, return_inverse=True)
    codes = codes.reshape(-1, 2).astype(np.int32)
    return gdf, codes[:, 0], codes[:, 1], int(codes.max()) + 1


def build_network(
    edges_gdf: gpd.GeoDataFrame,
    mtfcc_set: str = "drive",
    projected_crs: str = DEFAULT_PROJECTED_CRS,
    largest_component_only: bool = True,
    drop_self_loops: bool = True,
) -> RoadNetwork:
    """Turn a raw TIGER edges/roads GeoDataFrame into a RoadNetwork."""
    gdf = edges_gdf.loc[_road_mask(edges_gdf, mtfcc_set)].copy()
    if gdf.empty:
        raise ValueError(f"No edges survived the {mtfcc_set!r} MTFCC filter.")
    gdf = gdf.to_crs(projected_crs).reset_index(drop=True)

    has_tnid = {"TNIDF", "TNIDT"}.issubset(gdf.columns)
    if has_tnid:
        gdf, src, dst, n_nodes = _node_index_from_tnid(gdf)
    else:
        log.warning("No TNIDF/TNIDT columns -- falling back to geometric noding.")
        gdf, src, dst, n_nodes = _node_index_from_geometry(gdf)
    gdf = gdf.reset_index(drop=True)

    if drop_self_loops:
        keep = src != dst
        if not keep.all():
            log.info("Dropping %d self-loop edges", int((~keep).sum()))
            gdf, src, dst = gdf.loc[keep].reset_index(drop=True), src[keep], dst[keep]

    length = gdf.geometry.length.to_numpy(dtype=float)

    # Node coordinates: average the endpoint coordinates that map to each node.
    ends = np.array([(g.coords[0], g.coords[-1]) for g in gdf.geometry], dtype=float)
    node_xy = np.zeros((n_nodes, 2))
    counts = np.zeros(n_nodes)
    for side, idx in ((0, src), (1, dst)):
        np.add.at(node_xy, idx, ends[:, side, :])
        np.add.at(counts, idx, 1)
    node_xy /= np.maximum(counts, 1)[:, None]

    net = RoadNetwork(
        src=src, dst=dst, length=length, n_nodes=n_nodes,
        node_xy=node_xy, edges=gdf, crs=projected_crs,
        meta={"mtfcc_set": mtfcc_set, "topology": "tnid" if has_tnid else "geometric"},
    )

    if largest_component_only:
        net = restrict_to_largest_component(net)
    return net


def restrict_to_largest_component(net: RoadNetwork) -> RoadNetwork:
    """Keep only the giant component of the *undamaged* network.

    Real county extracts always contain a few stranded fragments (a frontage
    road clipped at the county line, a private drive). Leaving them in means
    they are counted as "fragmented" at p=0, which biases every curve upward.
    """
    labels = _components(net.src, net.dst, net.n_nodes, np.ones(net.n_edges, bool))
    deg = net.degree()
    sizes = np.bincount(labels[deg > 0])
    giant = int(sizes.argmax())
    keep = labels[net.src] == giant
    if keep.all():
        return net
    log.info(
        "Baseline: keeping giant component, dropping %d of %d edges (%.2f%%)",
        int((~keep).sum()), net.n_edges, 100 * (~keep).mean(),
    )
    sub = net.edges.loc[keep].reset_index(drop=True)
    old_src, old_dst = net.src[keep], net.dst[keep]
    codes, uniq = pd.factorize(np.concatenate([old_src, old_dst]))
    m = keep.sum()
    return RoadNetwork(
        src=codes[:m].astype(np.int32),
        dst=codes[m:].astype(np.int32),
        length=net.length[keep],
        n_nodes=len(uniq),
        node_xy=net.node_xy[np.asarray(uniq)],
        edges=sub,
        crs=net.crs,
        meta={**net.meta, "restricted_to_giant": True},
    )


def _components(src, dst, n_nodes, keep_mask) -> np.ndarray:
    """Connected-component labels for the subgraph induced by `keep_mask`."""
    s, d = src[keep_mask], dst[keep_mask]
    g = coo_matrix(
        (np.ones(len(s), dtype=np.int8), (s, d)), shape=(n_nodes, n_nodes)
    ).tocsr()
    _, labels = connected_components(g, directed=False)
    return labels
