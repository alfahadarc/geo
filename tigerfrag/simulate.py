"""The simulation core: delete links, find what got marooned, sweep over p.

Definitions used throughout (state these explicitly in your write-up, because
the paper is vague about them and they change the answer by a lot):

  deleted    an edge removed by the deletion protocol.
  marooned   an edge that survived deletion but can no longer reach its zone's
             median/anchor node through *any* surviving path.
  nominal fragmentation    deleted / total edges in the zone.
  effective fragmentation  (deleted + marooned) / total edges in the zone.

The 20% -> 83% result in the source paper is the gap between those last two.

Two reachability scopes are supported and they are NOT interchangeable:

  scope="county"  paths may leave the zone and come back. This is physically
                  correct -- an ambulance does not stop at a ZCTA boundary.
  scope="zone"    paths are confined to edges inside the zone. This is what a
                  naive per-zone subgraph analysis does, and it produces
                  dramatically more marooning.

Run both. If you can only reproduce the paper's headline number under
scope="zone", that is a finding worth reporting, not a bug to hide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .network import RoadNetwork
from .zones import ZoneAssignment

log = logging.getLogger(__name__)

# Rough structural-fitness proxy by road class, for the fitness-weighted
# protocol. Higher = more likely to survive. These are a modelling assumption,
# not data -- swap in bridge condition, elevation, or betweenness if you have it.
DEFAULT_FITNESS_BY_MTFCC = {
    "S1100": 0.95, "S1200": 0.85, "S1400": 0.55, "S1500": 0.25,
    "S1630": 0.70, "S1640": 0.40, "S1730": 0.30, "S1740": 0.30,
    "S1780": 0.25,
}


@dataclass
class MaroonView:
    """Precomputed graph indexing for one (network, zones, scope) combination.

    Built once, reused for every replicate. This is what makes 1,000 bootstrap
    replicates cheap.
    """

    src: np.ndarray
    dst: np.ndarray
    n_nodes: int
    anchor_idx: np.ndarray      # (Z,) node index in this view's numbering
    zone_of_edge: np.ndarray    # (M,) -1 for edges not counted in any zone
    n_zones: int
    scope: str
    counted: np.ndarray         # (M,) bool: edge belongs to a scored zone


def build_view(net: RoadNetwork, za: ZoneAssignment, scope: str = "county") -> MaroonView:
    counted = za.zone_of_edge >= 0
    if scope == "county":
        return MaroonView(
            src=net.src, dst=net.dst, n_nodes=net.n_nodes,
            anchor_idx=za.anchor_node.astype(np.int64),
            zone_of_edge=za.zone_of_edge, n_zones=za.n_zones,
            scope=scope, counted=counted,
        )
    if scope == "zone":
        # Compound node = (zone, node). Same physical intersection in two zones
        # becomes two separate vertices, so paths cannot leak across boundaries.
        z = np.where(counted, za.zone_of_edge, 0).astype(np.int64)
        key_s = z * net.n_nodes + net.src
        key_d = z * net.n_nodes + net.dst
        key_a = np.arange(za.n_zones, dtype=np.int64) * net.n_nodes + za.anchor_node
        all_keys = np.concatenate([key_s, key_d, key_a])
        uniq, inv = np.unique(all_keys, return_inverse=True)
        m = net.n_edges
        return MaroonView(
            src=inv[:m].astype(np.int32), dst=inv[m:2 * m].astype(np.int32),
            n_nodes=len(uniq), anchor_idx=inv[2 * m:].astype(np.int64),
            zone_of_edge=za.zone_of_edge, n_zones=za.n_zones,
            scope=scope, counted=counted,
        )
    raise ValueError("scope must be 'county' or 'zone'")


# ---------------------------------------------------------------------------
# Deletion protocols
# ---------------------------------------------------------------------------

def fitness_from_mtfcc(net: RoadNetwork, table=None) -> np.ndarray:
    table = table or DEFAULT_FITNESS_BY_MTFCC
    if "MTFCC" not in net.edges.columns:
        return np.full(net.n_edges, 0.5)
    return net.edges["MTFCC"].map(table).fillna(0.5).to_numpy(dtype=float)


def _weighted_sample_without_replacement(weights, k, rng) -> np.ndarray:
    """Efraimidis-Spirakis: exact weighted sampling without replacement, O(M).

    Far faster than rng.choice(..., p=..., replace=False), which is O(k*M).
    Indices come back sorted by key, i.e. in draw order -- `unit="road"` walks
    the result cumulatively, so an unordered partition would silently break the
    weighting there.
    """
    w = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
    n = len(w)
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, n)
    keys = rng.exponential(size=n) / w
    if k == n:
        return np.argsort(keys)
    cand = np.argpartition(keys, k - 1)[:k]
    return cand[np.argsort(keys[cand])]


def select_deleted(
    net: RoadNetwork,
    p: float,
    rng: np.random.Generator,
    protocol: str = "random",
    unit: str = "edge",
    fitness: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean mask (True = deleted) removing ~p of edges.

    unit="edge"  removes individual TIGER edges (block faces). Standard bond
                 percolation. On a planar street grid this needs p near 0.4-0.5
                 before the giant component shatters.
    unit="road"  removes entire named roads (all edges sharing a LINEARID or
                 FULLNAME). Much more destructive at the same p, and closer to
                 what "remove 20% of main streets" means in plain English. If
                 you are trying to reproduce a 20% -> 83% result, try this first.
    """
    m = net.n_edges
    target = int(round(p * m))
    deleted = np.zeros(m, dtype=bool)
    if target <= 0:
        return deleted

    if protocol == "fitness":
        w = 1.0 - (fitness if fitness is not None else fitness_from_mtfcc(net))
    elif protocol == "random":
        w = None
    else:
        raise ValueError("protocol must be 'random' or 'fitness'")

    if unit == "edge":
        if w is None:
            deleted[rng.choice(m, size=target, replace=False)] = True
        else:
            deleted[_weighted_sample_without_replacement(w, target, rng)] = True
        return deleted

    if unit == "road":
        col = "LINEARID" if "LINEARID" in net.edges.columns else "FULLNAME"
        if col not in net.edges.columns:
            raise KeyError("unit='road' needs a LINEARID or FULLNAME column.")
        codes, _ = pd.factorize(net.edges[col].fillna("__unnamed__"))
        n_groups = codes.max() + 1
        sizes = np.bincount(codes, minlength=n_groups)
        if w is None:
            order = rng.permutation(n_groups)
        else:
            gw = np.bincount(codes, weights=w, minlength=n_groups) / np.maximum(sizes, 1)
            order = _weighted_sample_without_replacement(gw, n_groups, rng)
        take = np.cumsum(sizes[order]) <= target
        if not take.any():
            take[0] = True
        chosen = np.zeros(n_groups, dtype=bool)
        chosen[order[take]] = True
        return chosen[codes]

    raise ValueError("unit must be 'edge' or 'road'")


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------

def run_trial(view: MaroonView, deleted: np.ndarray, length: np.ndarray | None = None):
    """One deletion replicate. Returns (per-zone arrays, global scalars)."""
    keep = ~deleted
    s, d = view.src[keep], view.dst[keep]
    g = coo_matrix(
        (np.ones(len(s), dtype=np.int8), (s, d)), shape=(view.n_nodes, view.n_nodes)
    ).tocsr()
    _, labels = connected_components(g, directed=False)

    anchor_lab = labels[view.anchor_idx]                       # (Z,)
    z = view.zone_of_edge
    valid = view.counted

    # A surviving edge is reachable iff its endpoint shares a component label
    # with its own zone's anchor. Both endpoints are equivalent here: the edge
    # itself survives, so they are in the same component by construction.
    reachable = np.zeros(len(z), dtype=bool)
    sel = valid & keep
    reachable[sel] = labels[view.src[sel]] == anchor_lab[z[sel]]
    marooned = sel & ~reachable

    zi = np.where(valid, z, 0)
    n_total = np.bincount(zi[valid], minlength=view.n_zones)
    n_deleted = np.bincount(zi[valid & deleted], minlength=view.n_zones)
    n_marooned = np.bincount(zi[marooned], minlength=view.n_zones)

    per_zone = {"n_total": n_total, "n_deleted": n_deleted, "n_marooned": n_marooned}

    if length is not None:
        per_zone["km_total"] = np.bincount(zi[valid], weights=length[valid],
                                           minlength=view.n_zones) / 1000.0
        lost = valid & (deleted | marooned)
        per_zone["km_lost"] = np.bincount(zi[lost], weights=length[lost],
                                          minlength=view.n_zones) / 1000.0

    # Global percolation diagnostics on the surviving graph.
    #
    # Only meaningful under scope="county". Under scope="zone" the graph is
    # made of compound (zone, node) vertices, so it is disconnected by
    # construction and its "giant component" is an artefact of the partition,
    # not a percolation observable. Emitting NaN there is deliberate: a
    # plausible-looking 0.019 would otherwise end up in somebody's p_c table.
    if view.scope != "county":
        glob = {"giant_frac": np.nan, "second_frac": np.nan, "n_components": np.nan}
        return per_zone, glob

    deg = np.bincount(np.concatenate([s, d]), minlength=view.n_nodes)
    comp_sizes = np.bincount(labels[deg > 0])
    comp_sizes = np.sort(comp_sizes)[::-1]
    n_live = int((deg > 0).sum())
    glob = {
        "giant_frac": float(comp_sizes[0] / n_live) if n_live else 0.0,
        # The second-largest cluster peaks at p_c. This is the standard
        # finite-size estimator of the percolation threshold -- much more
        # defensible than eyeballing where the giant component "looks" like
        # it drops.
        "second_frac": float(comp_sizes[1] / n_live) if len(comp_sizes) > 1 else 0.0,
        "n_components": int(len(comp_sizes)),
    }
    return per_zone, glob


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def sweep(
    net: RoadNetwork,
    za: ZoneAssignment,
    p_values=np.round(np.arange(0.01, 0.51, 0.01), 2),
    n_replicates: int = 100,
    scope: str = "county",
    protocol: str = "random",
    unit: str = "edge",
    seed: int = 0,
    zone_detail_at=(0.20,),
    label: str = "",
):
    """Run the full deletion sweep. Returns (global_df, zone_df).

    global_df   one row per (p, replicate): giant component, thresholds,
                county-wide nominal and effective fragmentation.
    zone_df     one row per (zone, p) averaged over replicates, but only at the
                p values in `zone_detail_at`. Recording every zone at every p
                for census blocks would be tens of millions of rows for no
                analytical gain -- the rank-ordered curve figures only need one
                deletion level. Three per-zone rates are reported so the deleted
                and marooned parts can be looked at separately:
                  frag_nominal    mean over replicates of n_deleted / n_total
                  frag_marooned   mean over replicates of n_marooned / n_total
                  frag_effective  mean over replicates of
                                  (n_deleted + n_marooned) / n_total
    """
    view = build_view(net, za, scope)
    fitness = fitness_from_mtfcc(net) if protocol == "fitness" else None
    detail = set(np.round(zone_detail_at, 4))

    glob_rows, zone_rows = [], []
    n_total_ref = None

    for p in p_values:
        acc = None
        for r in range(n_replicates):
            rng = np.random.default_rng((seed, int(round(p * 1000)), r))
            deleted = select_deleted(net, float(p), rng, protocol, unit, fitness)
            pz, gl = run_trial(view, deleted, net.length)

            if n_total_ref is None:
                n_total_ref = pz["n_total"]
            denom = np.maximum(pz["n_total"], 1)

            glob_rows.append({
                "level": za.level, "scope": scope, "protocol": protocol,
                "unit": unit, "p": float(p), "replicate": r,
                "frag_nominal": float(pz["n_deleted"].sum() / max(pz["n_total"].sum(), 1)),
                "frag_effective": float(
                    (pz["n_deleted"].sum() + pz["n_marooned"].sum())
                    / max(pz["n_total"].sum(), 1)
                ),
                "marooned_frac": float(pz["n_marooned"].sum() / max(pz["n_total"].sum(), 1)),
                # Mean over zones -- the paper's per-zone figures average this way,
                # which is not the same as the county-wide ratio above when zones
                # differ in size. Report both.
                "zone_mean_frag_effective": float(
                    ((pz["n_deleted"] + pz["n_marooned"]) / denom).mean()
                ),
                **gl,
                "label": label,
            })

            if round(float(p), 4) in detail:
                contrib = {
                    "frag_nominal": pz["n_deleted"] / denom,
                    "frag_marooned": pz["n_marooned"] / denom,
                    "frag_effective": (pz["n_deleted"] + pz["n_marooned"]) / denom,
                }
                if acc is None:
                    acc = contrib
                else:
                    acc = {k: acc[k] + contrib[k] for k in acc}

        if acc is not None:
            zone_rows.append(pd.DataFrame({
                "level": za.level, "scope": scope, "protocol": protocol,
                "unit": unit, "p": float(p),
                "zone_id": za.zone_ids.astype(str),
                "n_edges": n_total_ref,
                "frag_nominal": acc["frag_nominal"] / n_replicates,
                "frag_marooned": acc["frag_marooned"] / n_replicates,
                "frag_effective": acc["frag_effective"] / n_replicates,
                "label": label,
            }))

    zone_df = pd.concat(zone_rows, ignore_index=True) if zone_rows else pd.DataFrame()
    return pd.DataFrame(glob_rows), zone_df
