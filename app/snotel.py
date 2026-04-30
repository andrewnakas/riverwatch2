"""NRCS SNOTEL snow water equivalent fetcher.

Maps each USGS gauge to its nearest SNOTEL site (within ~50 km), then exposes
two helpers:

  - `fetch_swe_history(triplet, start, end)` — daily SWE in inches
  - `nearest_site(lat, lon)` — cached nearest-station lookup

SNOTEL data lives at the NRCS Air & Water Data Base (AWDB) REST API,
``https://wcc.sc.egov.usda.gov/awdbRestApi``. The full SNTL station list is
fetched once and cached on disk; data calls hit the per-station data
endpoint and merge into the existing per-gauge JSON record.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def _no_fetch() -> bool:
    """True when `RW2_NO_FETCH=1` is set — go cache-only and never hit the
    NRCS API. CI uses this to keep deploys deterministic on whatever data
    each shard's restored cache already has."""
    return os.environ.get("RW2_NO_FETCH") == "1"

ROOT = Path(__file__).resolve().parents[1] / "data" / "cache"
SITES_DIR = ROOT / "snotel_sites"
SITES_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR = ROOT / "snotel_records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

API = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
NEAREST_KM = 50.0  # only attach SNOTEL when there's a site this close


def _http_json(url: str, timeout: int = 60) -> object:
    req = Request(url, headers={"User-Agent": "riverwatch2/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _stations_path() -> Path:
    return SITES_DIR / "sntl_stations.json"


def _load_all_sntl() -> list[dict]:
    """Cache the SNTL station catalogue (refreshed weekly)."""
    p = _stations_path()
    if p.exists() and (time.time() - p.stat().st_mtime) < 7 * 86400:
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    url = (
        f"{API}/stations?networkCds=SNTL"
        "&returnForecastPointMetadata=false"
        "&returnReservoirMetadata=false"
        "&returnStationElements=false"
    )
    try:
        all_sites = _http_json(url)
    except Exception:
        return json.loads(p.read_text()) if p.exists() else []
    sntl = [
        s for s in (all_sites if isinstance(all_sites, list) else [])
        if s.get("networkCode") == "SNTL"
        and s.get("latitude") is not None
        and s.get("longitude") is not None
    ]
    if sntl:
        p.write_text(json.dumps(sntl, separators=(",", ":")))
    return sntl


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


_NEAREST_INDEX: Optional[Path] = SITES_DIR / "nearest_by_gauge.json"


def nearest_site(gauge_id: str, lat: float, lon: float) -> Optional[dict]:
    """Return the closest SNOTEL site dict (with `stationTriplet`, `distance_km`)
    or None if nothing within `NEAREST_KM`. Cached per-gauge."""
    cache: dict = {}
    if _NEAREST_INDEX and _NEAREST_INDEX.exists():
        try:
            cache = json.loads(_NEAREST_INDEX.read_text())
        except Exception:
            cache = {}
    if gauge_id in cache:
        v = cache[gauge_id]
        if v is None or (isinstance(v, dict) and "stationTriplet" in v):
            return v
    if _no_fetch():
        return None
    sites = _load_all_sntl()
    best: Optional[dict] = None
    best_d = float("inf")
    for s in sites:
        d = _haversine_km(lat, lon, s["latitude"], s["longitude"])
        if d < best_d:
            best_d = d
            best = s
    if best is None or best_d > NEAREST_KM:
        cache[gauge_id] = None
    else:
        cache[gauge_id] = {
            "stationTriplet": best["stationTriplet"],
            "name": best.get("name"),
            "elevation_ft": best.get("elevation"),
            "distance_km": round(best_d, 2),
        }
    if _NEAREST_INDEX:
        _NEAREST_INDEX.write_text(json.dumps(cache, separators=(",", ":")))
    return cache[gauge_id]


def _record_path(triplet: str) -> Path:
    safe = triplet.replace(":", "_")
    return RECORDS_DIR / f"{safe}.json"


def fetch_swe_history(triplet: str, start: date, end: date, *, max_age_hours: int = 24) -> pd.DataFrame:
    """Daily SWE (inches) for a SNOTEL site, with persistent incremental caching."""
    rp = _record_path(triplet)
    rec: dict = {}
    if rp.exists():
        try:
            rec = json.loads(rp.read_text())
        except Exception:
            rec = {}
    have: dict = rec.get("rows", {})
    last_known = rec.get("last_known")
    first_known = rec.get("first_known") or (min(have.keys()) if have else None)
    fetched_at = rec.get("fetched_at", 0)
    age = time.time() - float(fetched_at) if fetched_at else float("inf")

    needs_backward = bool(first_known) and first_known > start.isoformat()
    needs_forward = (not last_known) or (last_known < end.isoformat())

    if not needs_backward and last_known and last_known >= end.isoformat() and age < max_age_hours * 3600:
        return _slice(have, start, end)
    if _no_fetch():
        return _slice(have, start, end)

    fwd_from = start
    if last_known:
        try:
            ld = date.fromisoformat(last_known)
            if ld >= start:
                fwd_from = ld - timedelta(days=2)
        except ValueError:
            pass

    fetched_any = False
    if needs_backward:
        bwd_from = start
        bwd_to = (date.fromisoformat(first_known) - timedelta(days=1)) if first_known else end
        if bwd_from <= bwd_to:
            fetched_any = _fetch_and_merge(triplet, have, bwd_from, bwd_to) or fetched_any
    if needs_forward and fwd_from <= end:
        fetched_any = _fetch_and_merge(triplet, have, fwd_from, end) or fetched_any

    if fetched_any and have:
        rec = {
            "stationTriplet": triplet,
            "rows": have,
            "first_known": min(have.keys()),
            "last_known": max(have.keys()),
            "fetched_at": time.time(),
        }
        rp.write_text(json.dumps(rec, separators=(",", ":")))
    return _slice(have, start, end)


def _fetch_and_merge(triplet: str, have: dict, start: date, end: date) -> bool:
    """Pull WTEQ (SWE) and SNWD (snow depth) in one request; merge into `have`."""
    params = {
        "stationTriplets": triplet,
        "elements": "WTEQ,SNWD",
        "duration": "DAILY",
        "beginDate": start.isoformat(),
        "endDate": end.isoformat(),
    }
    url = f"{API}/data?" + urlencode(params)
    try:
        payload = _http_json(url)
    except Exception:
        return False
    if not isinstance(payload, list) or not payload:
        return False
    added = False
    for station_payload in payload:
        for elt in station_payload.get("data", []):
            element = elt.get("stationElement", {}).get("elementCode")
            if element not in ("WTEQ", "SNWD"):
                continue
            key = "swe_in" if element == "WTEQ" else "snow_depth_in"
            for row in elt.get("values", []) or []:
                d = row.get("date")
                v = row.get("value")
                if d is None or v is None:
                    continue
                # AWDB returns ISO date "YYYY-MM-DD" or with time appended
                d = d[:10]
                cell = have.setdefault(d, {})
                cell[key] = float(v)
                added = True
    return added


def _slice(have: dict, start: date, end: date) -> pd.DataFrame:
    if not have:
        return pd.DataFrame(columns=["date", "swe_in", "snow_depth_in"])
    s, e = start.isoformat(), end.isoformat()
    rows = []
    for d_iso, vals in have.items():
        if not (s <= d_iso <= e):
            continue
        rows.append({
            "date": d_iso,
            "swe_in": vals.get("swe_in") if isinstance(vals, dict) else None,
            "snow_depth_in": vals.get("snow_depth_in") if isinstance(vals, dict) else None,
        })
    if not rows:
        return pd.DataFrame(columns=["date", "swe_in", "snow_depth_in"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)
