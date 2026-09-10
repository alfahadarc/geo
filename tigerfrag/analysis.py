"""Analysis and figures: curve collapse, percolation threshold, power-law fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .network import RoadNetwork
from .simulate import select_deleted


# ---------------------------------------------------------------------------
# Figure 7 equivalent: rank-ordered fragmentation curves, one per zone level
# ---------------------------------------------------------------------------

def rank_curves(zone_df: pd.DataFrame, p: float = 0.20) -> pd.DataFrame:
    """Rank-order zones by fragmentation within each level and normalise rank.

    Normalising rank to [0, 1] is what lets a level with 50 ZCTAs and a level
    with 50,000 blocks be plotted on the same axis. If the curves land on top of
    each other, the fragmentation process is statistically indistinguishable
    across aggregation scales -- that is the scale-invariance claim, and it is
    the one result in the paper you can test cleanly.
    """
    sub = zone_df[np.isclose(zone_df["p"], p)].copy()
    out = []
    for keys, g in sub.groupby(["level", "scope", "protocol", "unit"], sort=False):
        g = g.sort_values("frag_effective").reset_index(drop=True)
        n = len(g)
        g["norm_rank"] = (np.arange(n) + 0.5) / n
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else sub


def curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Table 1 equivalent: median fragmentation, rank at 50%, crude tail slope."""
    rows = []
    for keys, g in curves.groupby(["level", "scope", "protocol", "unit"], sort=False):
        g = g.sort_values("norm_rank")
        f = g["frag_effective"].to_numpy()
        x = g["norm_rank"].to_numpy()
        above = np.where(f >= 0.5)[0]
        rank50 = float(x[above[0]]) if above.size else np.nan

        tail = x > 0.6
        slope = np.nan
        if tail.sum() > 5:
            xx, yy = 1 - x[tail], 1 - f[tail]
            ok = (xx > 0) & (yy > 0)
            if ok.sum() > 5:
                slope = float(np.polyfit(np.log(xx[ok]), np.log(yy[ok]), 1)[0])

        rows.append({
            "level": keys[0], "scope": keys[1], "protocol": keys[2], "unit": keys[3],
            "n_zones": len(g),
            "median_frag": float(np.median(f)),
            "rank_at_50pct": rank50,
            "tail_slope_ols": slope,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Percolation threshold
# ---------------------------------------------------------------------------

def estimate_pc(global_df: pd.DataFrame) -> pd.DataFrame:
    """p_c from the peak of the second-largest cluster, plus a giant-drop check.

    Two estimators, because they should agree and it is worth knowing if they
    do not:
      pc_second_peak  p where the second-largest cluster is biggest. This is the
                      textbook finite-size estimator.
      pc_giant_half   p where the giant component first falls below half the
                      network. Cruder, but it is what people usually eyeball.
    """
    rows = []
    keys = ["level", "scope", "protocol", "unit"]
    df = global_df[global_df["scope"] == "county"] if "scope" in global_df else global_df
    for k, g in df.groupby(keys, sort=False):
        m = g.groupby("p")[["second_frac", "giant_frac"]].mean().sort_index()
        if m["second_frac"].isna().all():
            continue
        below = m.index[m["giant_frac"] < 0.5]
        rows.append({
            **dict(zip(keys, k)),
            "pc_second_peak": float(m["second_frac"].idxmax()),
            "max_second_frac": float(m["second_frac"].max()),
            "pc_giant_half": float(below[0]) if len(below) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cluster-size distribution and power-law fitting
# ---------------------------------------------------------------------------

def cluster_sizes(
    net: RoadNetwork, p: float, n_replicates: int = 20, seed: int = 0,
    protocol: str = "random", unit: str = "edge",
) -> np.ndarray:
    """Pooled component sizes (in nodes) after deleting a fraction p of links.

    This is the quantity that should follow n_s ~ s^-tau at p_c, with tau close
    to 187/91 = 2.0549 for 2D percolation. Testing that is a much stronger
    replication of the paper's core claim than fitting a slope to a rank curve.
    """
    sizes = []
    for r in range(n_replicates):
        rng = np.random.default_rng((seed, int(round(p * 1000)), r, 7))
        deleted = select_deleted(net, p, rng, protocol, unit)
        keep = ~deleted
        s, d = net.src[keep], net.dst[keep]
        g = coo_matrix(
            (np.ones(len(s), np.int8), (s, d)), shape=(net.n_nodes, net.n_nodes)
        ).tocsr()
        _, labels = connected_components(g, directed=False)
        deg = np.bincount(np.concatenate([s, d]), minlength=net.n_nodes)
        cs = np.bincount(labels[deg > 0])
        sizes.append(cs[cs > 0])
    return np.concatenate(sizes)


def fit_powerlaw(data: np.ndarray, xmin_candidates=None) -> dict:
    """Clauset-Shalizi-Newman MLE fit with KS-selected x_min.

    Do NOT fit a straight line to a log-log histogram and report the slope --
    that is the mistake the source paper's own Table 1 admits to. This uses
    maximum likelihood over the tail and picks x_min by minimising the KS
    distance, which is the standard the paper cites (reference [17]) but only
    partially follows.
    """
    x = np.sort(np.asarray(data, dtype=float))
    x = x[x > 0]
    if x.size < 20:
        return {"alpha": np.nan, "xmin": np.nan, "n_tail": 0, "ks": np.nan}

    if xmin_candidates is None:
        xmin_candidates = np.unique(x)[:-10]
        if xmin_candidates.size > 200:
            idx = np.linspace(0, xmin_candidates.size - 1, 200).astype(int)
            xmin_candidates = xmin_candidates[idx]

    best = {"alpha": np.nan, "xmin": np.nan, "n_tail": 0, "ks": np.inf}
    for xmin in xmin_candidates:
        tail = x[x >= xmin]
        n = tail.size
        if n < 20:
            continue
        alpha = 1.0 + n / np.sum(np.log(tail / (xmin - 0.5)))
        emp = np.arange(n, dtype=float) / n
        theo = 1.0 - (tail / (xmin - 0.5)) ** (1.0 - alpha)
        ks = float(np.max(np.abs(emp - theo)))
        if ks < best["ks"]:
            best = {"alpha": float(alpha), "xmin": float(xmin), "n_tail": int(n), "ks": ks}
    return best


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_rank_curves(curves: pd.DataFrame, ax=None, title=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7.5, 5))[1]
    for (lvl, scope), g in curves.groupby(["level", "scope"], sort=False):
        g = g.sort_values("norm_rank")
        ax.plot(g["norm_rank"], g["frag_effective"], lw=1.6,
                label=f"{lvl} ({scope})")
    ax.set_xlabel("Normalized rank of spatial unit")
    ax.set_ylabel("Effective fragmentation")
    ax.set_title(title or "Cross-scalar fragmentation curves")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    return ax


def plot_percolation(global_df: pd.DataFrame, ax=None, title=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7.5, 5))[1]
    m = global_df.groupby("p")[["giant_frac", "second_frac", "frag_effective"]].mean()
    ax.plot(m.index, m["giant_frac"], lw=2, label="Giant component fraction")
    ax.plot(m.index, m["frag_effective"], lw=2, label="Effective fragmentation")
    ax.plot(m.index, m["second_frac"] / max(m["second_frac"].max(), 1e-9),
            lw=1.2, ls="--", label="2nd cluster (scaled) -- peak marks $p_c$")
    ax.axvline(float(m["second_frac"].idxmax()), color="0.4", ls=":", lw=1)
    ax.set_xlabel("Fraction of links deleted, $p$")
    ax.set_ylabel("Fraction")
    ax.set_title(title or "Percolation transition")
    ax.legend(fontsize=8)
    return ax


def plot_cluster_distribution(sizes: np.ndarray, fit: dict | None = None, ax=None):
    """Log-log complementary CDF -- the right way to look at a heavy tail.

    A CCDF does not need binning, so it cannot be distorted by bin choice the
    way a log-log histogram can.
    """
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 5))[1]
    x = np.sort(sizes)
    ccdf = 1.0 - np.arange(len(x)) / len(x)
    ax.loglog(x, ccdf, ".", ms=3, alpha=0.5, label="Empirical CCDF")
    if fit and np.isfinite(fit.get("alpha", np.nan)):
        xs = x[x >= fit["xmin"]]
        c = (fit["n_tail"] / len(x)) * (xs / fit["xmin"]) ** (1 - fit["alpha"])
        ax.loglog(xs, c, "r-", lw=1.5,
                  label=rf"MLE fit $\alpha$={fit['alpha']:.2f}, $x_{{min}}$={fit['xmin']:.0f}")
    ax.set_xlabel("Cluster size $s$ (nodes)")
    ax.set_ylabel("$P(S \\geq s)$")
    ax.legend(fontsize=8)
    return ax
