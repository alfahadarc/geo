"""Static configuration: TIGER/Line URL templates, MTFCC road classes, CRS."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TIGER/Line vintage. 2025 was released 2025-09-23 and is the current vintage.
# Bump this when the Census Bureau publishes a new one (usually late September).
# ---------------------------------------------------------------------------
DEFAULT_YEAR = 2025

_BASE = "https://www2.census.gov/geo/tiger/TIGER{year}"

# {year}, {fips5} = 5-digit state+county FIPS, {state} = 2-digit state FIPS
LAYER_URLS = {
    # --- network layers (county-based) ---
    # EDGES ("All Lines") is the topological layer: it carries TNIDF / TNIDT,
    # the Census's own from/to node IDs. Use it -- it gives you exact network
    # topology with zero geometric snapping error.
    "edges": _BASE + "/EDGES/tl_{year}_{fips5}_edges.zip",
    # ROADS is the cartographic layer (LINEARID, FULLNAME, MTFCC) with no node
    # IDs. Only useful if you want named-road grouping or a lighter download.
    "roads": _BASE + "/ROADS/tl_{year}_{fips5}_roads.zip",
    # --- zone layers ---
    "block": _BASE + "/TABBLOCK20/tl_{year}_{state}_tabblock20.zip",   # state
    "bg": _BASE + "/BG/tl_{year}_{state}_bg.zip",                      # state
    "tract": _BASE + "/TRACT/tl_{year}_{state}_tract.zip",             # state
    "zcta": _BASE + "/ZCTA520/tl_{year}_us_zcta520.zip",               # NATIONAL (~500 MB)
    "county": _BASE + "/COUNTY/tl_{year}_us_county.zip",               # national
}

# Which layers are national (no state/county substitution needed)
NATIONAL_LAYERS = {"zcta", "county"}
STATE_LAYERS = {"block", "bg", "tract"}
COUNTY_LAYERS = {"edges", "roads"}

# ---------------------------------------------------------------------------
# MAF/TIGER Feature Class Codes.
# Which codes you keep is the single biggest lever on your results -- a network
# of arterials fragments far faster than a full network including every alley.
# ---------------------------------------------------------------------------
MTFCC_SETS = {
    # Everything a car can drive on. Closest to an OSMnx "drive" network.
    "drive": {"S1100", "S1200", "S1400", "S1500", "S1630", "S1640"},
    # Arterials + city streets only. This is the "main street" network; it is
    # sparser and more tree-like, so it fragments much earlier.
    "main": {"S1100", "S1200", "S1400"},
    # Highways and major arterials only. Very sparse; expect early collapse.
    "arterial": {"S1100", "S1200"},
    # No filtering at all (includes alleys, private drives, parking aisles).
    "all": None,
}

MTFCC_LABELS = {
    "S1100": "Primary road (interstate/limited access)",
    "S1200": "Secondary road (US/state highway)",
    "S1400": "Local neighborhood road, rural road, city street",
    "S1500": "Vehicular trail (4WD)",
    "S1630": "Ramp",
    "S1640": "Service drive",
    "S1710": "Walkway/pedestrian trail",
    "S1720": "Stairway",
    "S1730": "Alley",
    "S1740": "Private road for service vehicles",
    "S1750": "Internal US Census Bureau use",
    "S1780": "Parking lot road",
    "S1820": "Bike path or trail",
    "S1830": "Bridle path",
}

# ---------------------------------------------------------------------------
# CRS. TIGER ships in EPSG:4269 (NAD83 geographic). Reproject before measuring
# any length or distance. EPSG:5070 (Albers Equal Area CONUS) is a safe default
# for the lower 48; use a UTM zone if you want lower local distortion.
# ---------------------------------------------------------------------------
TIGER_CRS = "EPSG:4269"
DEFAULT_PROJECTED_CRS = "EPSG:5070"

# Zone-layer GEOID column names, in order of preference.
GEOID_CANDIDATES = ("GEOID20", "GEOID10", "GEOID")

# A few FIPS codes for convenience / smoke tests.
EXAMPLE_COUNTIES = {
    "18097": "Marion County, IN (Indianapolis) -- dense gridiron",
    "17031": "Cook County, IL (Chicago) -- dense gridiron",
    "13121": "Fulton County, GA (Atlanta) -- radial-concentric",
    "37119": "Mecklenburg County, NC (Charlotte) -- dendritic sprawl",
    "53033": "King County, WA (Seattle) -- terrain-constrained",
    "42003": "Allegheny County, PA (Pittsburgh) -- terrain-constrained",
}
