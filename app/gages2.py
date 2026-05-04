"""v14.5a: lazy loader for the curated GAGES-II static-attribute table.

`scripts/build_gages2_attrs.py` distills Falcone 2011's 9067-gauge basin
characterization into one ~480 KB JSON keyed by USGS site ID. ~68% of our
1893 actives have a row; the rest get an empty dict and the corresponding
LightGBM feature cells stay NaN (which both the pooled-LGBM and the v14.4
stacker handle natively as a separate split).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

_ATTRS_PATH = Path(__file__).resolve().parents[1] / "data" / "gages2_attrs.json"
_cache: Optional[Dict[str, Dict[str, float]]] = None


# The full column set we ship. Anything outside this list is ignored — keep
# in sync with `scripts/build_gages2_attrs.py` so we don't accidentally
# embed columns the stacker / pooled LGBM aren't trained on.
GAGES2_KEYS = (
    "FORESTNLCD06", "DEVNLCD06", "WOODYWETNLCD06", "EMERGWETNLCD06",
    "HGA_PCT", "HGB_PCT", "HGC_PCT", "HGD_PCT",
    "AWCAVE", "PERMAVE",
    "ELEV_MEAN_M_BASIN", "SLOPE_PCT",
    "BFI_AVE", "TOPWET", "RUNAVE7100",
    "PPTAVG_BASIN", "T_AVG_BASIN", "SNOW_PCT_PRECIP",
)


def _load() -> Dict[str, Dict[str, float]]:
    global _cache
    if _cache is not None:
        return _cache
    if not _ATTRS_PATH.exists():
        _cache = {}
        return _cache
    try:
        _cache = json.loads(_ATTRS_PATH.read_text())
    except Exception:
        _cache = {}
    return _cache


def attrs_for(station_id: str) -> Dict[str, float]:
    """Return the GAGES-II static attribute row for `station_id` or `{}`."""
    return _load().get(str(station_id), {}) or {}


def enrich_station_attrs(st: dict) -> dict:
    """Merge GAGES-II attrs into a station dict in place; return same dict.

    Existing keys win — we never overwrite metadata like name/lat/lon.
    """
    sid = str(st.get("id") or st.get("site_no") or "")
    if not sid:
        return st
    g2 = attrs_for(sid)
    for k, v in g2.items():
        if k not in st:
            st[k] = v
    return st


def coverage_summary() -> dict:
    """Quick observability hook for the build script."""
    a = _load()
    return {"stations_with_gages2": len(a), "keys": list(GAGES2_KEYS)}
