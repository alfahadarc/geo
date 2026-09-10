"""Download + cache TIGER/Line shapefiles.

Everything is cached on disk by URL, so re-running an experiment or adding a
second county never re-downloads a file you already have. The national ZCTA
file is large (~500 MB) but you download it exactly once for all 50 counties.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd

from .config import (
    COUNTY_LAYERS,
    DEFAULT_YEAR,
    LAYER_URLS,
    NATIONAL_LAYERS,
    STATE_LAYERS,
)

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path("./tiger_cache")


def build_url(layer: str, fips5: str | None = None, year: int = DEFAULT_YEAR) -> str:
    """Resolve a TIGER download URL for a layer, given a 5-digit county FIPS."""
    if layer not in LAYER_URLS:
        raise KeyError(f"Unknown layer {layer!r}. Options: {sorted(LAYER_URLS)}")
    tmpl = LAYER_URLS[layer]
    if layer in NATIONAL_LAYERS:
        return tmpl.format(year=year)
    if fips5 is None:
        raise ValueError(f"Layer {layer!r} needs a county FIPS code.")
    fips5 = str(fips5).zfill(5)
    if layer in STATE_LAYERS:
        return tmpl.format(year=year, state=fips5[:2])
    if layer in COUNTY_LAYERS:
        return tmpl.format(year=year, fips5=fips5)
    raise KeyError(layer)


def fetch(url: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Download `url` into the cache (if absent), unzip it, return the .shp path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    name = url.rsplit("/", 1)[-1]
    zip_path = cache_dir / name
    extract_dir = cache_dir / name.replace(".zip", "")

    if not zip_path.exists():
        log.info("Downloading %s", url)
        tmp = zip_path.with_suffix(".partial")
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.rename(zip_path)
    else:
        log.info("Cache hit: %s", zip_path.name)

    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    shps = sorted(extract_dir.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No .shp found inside {zip_path}")
    return shps[0]


def load_layer(
    layer: str,
    fips5: str | None = None,
    year: int = DEFAULT_YEAR,
    cache_dir: Path = DEFAULT_CACHE,
    columns: list[str] | None = None,
    bbox=None,
) -> gpd.GeoDataFrame:
    """Fetch (if needed) and read a TIGER layer as a GeoDataFrame.

    `columns` and `bbox` are pushed down to the reader, which matters a lot for
    the national ZCTA file -- passing the county bbox turns a 33k-row read into
    a handful of rows.
    """
    shp = fetch(build_url(layer, fips5, year), cache_dir)
    kwargs = {}
    if columns is not None:
        kwargs["columns"] = columns
    if bbox is not None:
        kwargs["bbox"] = bbox
    gdf = gpd.read_file(shp, **kwargs)
    log.info("Read %s: %d features", layer, len(gdf))
    return gdf


def load_county_boundary(
    fips5: str, year: int = DEFAULT_YEAR, cache_dir: Path = DEFAULT_CACHE
) -> gpd.GeoDataFrame:
    """Single-row GeoDataFrame for one county."""
    fips5 = str(fips5).zfill(5)
    counties = load_layer("county", year=year, cache_dir=cache_dir)
    sel = counties[counties["GEOID"] == fips5]
    if sel.empty:
        raise ValueError(f"County FIPS {fips5} not found in the {year} county file.")
    return sel.reset_index(drop=True)
