# tigerfrag — road-network fragmentation in Python

Open-source replication of the TransCAD workflow: take a county, partition it
into zones, delete a fraction of street links, and measure how much of the
network gets *marooned* — cut off from its zone's median node even though it
was never deleted.

Built to scale from one county to fifty without changing anything but a list of
FIPS codes.

---

## 1. Where the data comes from

Everything is free and downloads over plain HTTPS. No API key.

| What you need | TIGER/Line layer | URL pattern | Scope |
|---|---|---|---|
| Road network **with topology** | `EDGES` | `.../TIGER2025/EDGES/tl_2025_18097_edges.zip` | county |
| Road network (cartographic) | `ROADS` | `.../TIGER2025/ROADS/tl_2025_18097_roads.zip` | county |
| Census blocks | `TABBLOCK20` | `.../TIGER2025/TABBLOCK20/tl_2025_18_tabblock20.zip` | state |
| Block groups | `BG` | `.../TIGER2025/BG/tl_2025_18_bg.zip` | state |
| Census tracts | `TRACT` | `.../TIGER2025/TRACT/tl_2025_18_tract.zip` | state |
| ZCTAs | `ZCTA520` | `.../TIGER2025/ZCTA520/tl_2025_us_zcta520.zip` | **national, ~500 MB** |
| County boundaries | `COUNTY` | `.../TIGER2025/COUNTY/tl_2025_us_county.zip` | national |

Base: `https://www2.census.gov/geo/tiger/TIGER2025/`

`2025` is the current vintage (released 23 September 2025). The Bureau ships a
new one each September — change `DEFAULT_YEAR` in `tigerfrag/config.py`.

Marion County, Indiana is FIPS **18097** (state `18` + county `097`).

### Use EDGES, not ROADS

This is the single most important choice in the pipeline.

The `EDGES` layer carries **`TNIDF`** and **`TNIDT`** — the Census Bureau's own
"from node" and "to node" identifiers. Two street segments meet at an
intersection if and only if they share a TNID. You get exact, authoritative
topology as a table join, with zero geometry involved.

The alternative — which is what most tutorials do — is to load `ROADS` and infer
intersections by snapping endpoints within some tolerance. That approach is
where hand-rolled street-network projects quietly fail:

- Too loose a tolerance welds together roads that cross on an **overpass**,
  inventing connections that do not exist. Your network then looks far more
  resilient than it is.
- Too tight a tolerance leaves genuine T-intersections unjoined, so your
  baseline network arrives pre-fragmented and every curve is wrong from p = 0.

Since fragmentation analysis is *entirely* about which things connect to which,
a topology error does not add noise — it changes the answer. `EDGES` sidesteps
it. A geometric fallback is included for `ROADS`, with a warning.

`EDGES` also carries `MTFCC` (road class), `ROADFLG`, `LINEARID`/`FULLNAME`
(so you can delete whole named streets), and `ZIPL`/`ZIPR` (ZIP code on each
side, if you want ZIP zones without the 500 MB ZCTA download).

### Optional extras

- **Hospitals / trauma centres**, for the accessibility index `A(p)` in
  Equation 10 — HIFLD Open Data publishes a national hospitals point layer.
  Not wired in; the hooks are in `analysis.py`.
- **OSMnx** is a fine alternative source with cleaner drivable-road tagging,
  and gives you a graph directly. TIGER is the better choice here because the
  paper's zone geographies are Census geographies, so keeping roads and zones
  in one coordinate frame and one vintage avoids alignment headaches.
- **`pygris`** wraps these same downloads in a `tigris`-style API if you prefer
  that to the direct URLs.

---

## 2. Install

```bash
pip install geopandas shapely pyproj numpy pandas scipy matplotlib
```

Tested against geopandas 1.1, shapely 2.1, numpy 2.4, scipy 1.17. NetworkX is
deliberately **not** required — see §4.

```bash
python tests/test_synthetic.py     # ~20 s, no network access needed
```

Run that first. It builds a synthetic lattice and checks the whole chain, so if
something is broken you find out before downloading anything.

---

## 3. Quickstart

```bash
# Sanity-check a county's network without simulating anything
python run_experiment.py --fips 18097 --describe-only

# Small run: ZCTA level, coarse p grid, 20 replicates
python run_experiment.py --fips 18097 --levels zcta \
    --p-values 0.05:0.50:0.05 --replicates 20

# Full multi-scalar sweep for Marion County
python run_experiment.py --fips 18097 --levels block bg tract zcta \
    --pmedian 5 --scopes county zone --replicates 100

# Four topological archetypes, restartable
python run_experiment.py --fips 18097 13121 37119 53033 \
    --levels bg tract zcta --replicates 50 --outdir results/archetypes
```

Or from Python:

```python
from tigerfrag import pipeline

global_df, zone_df, summary = pipeline.run_county(
    "18097", levels=("bg", "tract", "zcta"), pmedian_k=(5,),
    scopes=("county", "zone"), n_replicates=100, outdir="results",
)
```

---

## 4. Pipeline stages

```
FIPS code
   │
   ├─ data.py       download + unzip + cache TIGER layers (cached by URL,
   │                so county #2 onward reuses the state and national files)
   │
   ├─ network.py    filter ROADFLG='Y' and MTFCC → reproject to EPSG:5070 →
   │                factorise TNIDF/TNIDT into a dense node index →
   │                keep the giant component → RoadNetwork(src, dst, length, xy)
   │
   ├─ zones.py      load block / bg / tract / zcta polygons, clip to county →
   │                assign each edge to a zone by its midpoint →
   │                pick each zone's anchor = incident node nearest its centroid
   │                (or assign_by_pmedian for k network medians, Figure 5)
   │
   ├─ simulate.py   for each p, for each replicate:
   │                  sample edges to delete →
   │                  connected components on the surviving graph →
   │                  edge is MAROONED if it survived but its component
   │                  differs from its zone anchor's component →
   │                  aggregate per zone with bincount
   │
   └─ analysis.py   rank-ordered curve collapse (Figure 7) · p_c from the
                    second-largest-cluster peak · cluster-size distribution ·
                    Clauset MLE power-law fit · plots
```

`RoadNetwork` holds plain NumPy arrays rather than a NetworkX graph, and each
replicate is one boolean mask plus one `scipy.sparse.csgraph.connected_components`
call. Measured: **~7 ms per replicate** on a 64k-edge, 1,600-zone network. A
full 50-value × 100-replicate sweep is under a minute per level; a whole county
across four levels and two scopes is a few minutes. Fifty counties runs
overnight. Rebuilding a NetworkX graph each replicate would be 50–100× slower
and would put the same run in the multi-day range.

### Output

- `global_<fips>.csv` — one row per (level, scope, protocol, unit, p, replicate):
  nominal and effective fragmentation, giant-component fraction, second-cluster
  fraction, component count.
- `zones_<fips>.csv` — per-zone fragmentation at the `--detail-p` levels only.
  Recording every block at every p would be tens of millions of rows for no
  analytical gain; the rank-curve figures need one deletion level.
- `curves_<fips>.csv` — rank-ordered, normalised curves ready to plot.
- `summary_<fips>.csv` — the Table 1 / Table 2 equivalent.

---

## 5. Four decisions that change your results

These are the knobs that matter. Every one of them is a defensible choice, and
every one of them moves the headline number, so state which you picked.

**1 · Which roads count (`--mtfcc`).** `drive` keeps everything a car can use
including service drives; `main` keeps arterials and city streets; `arterial`
keeps only highways. A sparse arterial network fragments far earlier than a
dense one at the same p. Your professor's phrase "from main street remove 20%"
points at `main`.

**2 · What a "20% deletion" removes (`--units`).** `edge` removes individual
block faces — textbook bond percolation. `road` removes entire named streets
(everything sharing a `LINEARID`). Removing 20% of the *edges* and removing 20%
of the *streets* are very different experiments, and the second is far more
destructive.

**3 · Where paths may run (`--scopes`).** `county` lets a route leave a zone and
come back, which is what an ambulance actually does. `zone` confines routing to
edges inside the zone. `zone` produces dramatically more marooning — in the
synthetic test, 0.27 vs 0.36 effective fragmentation at the same p = 0.20.

**4 · Where the median sits.** Zone polygon centroid (the TransCAD default your
professor described) versus a k-median solved on the network. Candidate nodes
are restricted to nodes actually incident to the zone's own edges — otherwise
the centroid of an L-shaped or river-split ZCTA can snap to a node in a
different zone and make the zone look connected when it isn't.

---

## 6. On reproducing the paper's 20% → 83%

Expect this to be the interesting part of your write-up rather than a formality.

The paper reports that deleting 20% of Marion County's street segments maroons
83% of them. Its own Table 2 reports a percolation threshold of
p_c ≈ 0.248 for dense-gridiron cities, with Marion County named as the gridiron
exemplar. Those two numbers are in tension: at p = 0.20 you are *below* the
stated threshold, so the giant component should still be largely intact and
most surviving segments should still reach their median. Standard 2D bond
percolation puts the threshold for a planar street lattice around p ≈ 0.4–0.5;
real road networks sit somewhat lower because their mean degree is near 3.

So if you run the physically correct configuration — full drivable network,
per-edge deletion, county-wide reachability — you will most likely get effective
fragmentation in the 25–35% range at p = 0.20, not 83%. That is not your
pipeline failing. It is the expected result.

The paper's 83% is reachable, but it needs specific choices: a sparser network
(`--mtfcc main` or `arterial`), and/or whole-street deletion (`--units road`),
and/or zone-confined routing (`--scopes zone`). The paper also mentions 17,947
street segments for Marion County, which is far fewer than a full TIGER edge
extract — another sign the analysis ran on a filtered arterial subset.

The pipeline exposes all of those as flags precisely so you can test which
combination reproduces the number. Running the grid and reporting *which
assumptions the headline result depends on* is a stronger replication than
matching it by accident, and it is the kind of thing that reads well in a
methods section.

Two related checks worth including:

- **`estimate_pc`** locates p_c from the peak of the second-largest cluster,
  which is the standard finite-size estimator, rather than by eyeballing where
  the giant component "looks like" it drops.
- **`fit_powerlaw`** does a Clauset–Shalizi–Newman maximum-likelihood fit with
  KS-selected `x_min`. The paper's own Table 1 concedes its exponents are
  ordinary least squares on binned log-log data, which is the exact practice
  Clauset et al. (the paper's reference [17]) was written to discourage. For 2D
  percolation the universal cluster-size exponent is τ = 187/91 ≈ 2.055; if your
  fit lands near that at p_c, you have replicated the universality claim
  properly. If it lands at 1.4 or 3.0, check your topology before you check
  your theory.

---

## 7. Extending it

- **More counties** — add FIPS codes to `--fips`. State and national files are
  cached, so counties in the same state cost one road download each.
- **Accessibility index A(p)** — add hospital points, snap to nearest node, and
  run `scipy.sparse.csgraph.dijkstra` from those nodes on the surviving graph.
  The per-replicate structure in `simulate.run_trial` is where that goes.
- **Targeted attack** — the paper only does random and fitness-weighted
  deletion. Deleting by descending betweenness is the classic contrast, and it
  is the case where a road network's bounded degree distribution stops
  protecting it. `--protocols` is where to add it.
- **Real fitness values** — `DEFAULT_FITNESS_BY_MTFCC` is a placeholder by road
  class. The National Bridge Inventory gives real condition ratings; elevation
  from a DEM gives flood vulnerability.
