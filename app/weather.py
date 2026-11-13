"""Open-Meteo weather fetchers (historical + short-range forecast).

Open-Meteo is free, no key, returns daily aggregations.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

NO_FETCH = os.environ.get("RW2_NO_FETCH") == "1"


# PREREG §V16-1 (H-STALE). Scheduled builds run RW2_NO_FETCH=1 so the cold
# multi-decade backfills are never repeated -- but that also froze the weather
# FORECAST and the recent weather history at whatever the cache last held.
# RW2_WX_REFRESH=1 allows, under NO_FETCH, exactly the cheap forward deltas:
# today's forecast and the days of history since the record's last row. It is
# the same product the members were fit on, so only its AGE changes. A fetch
# that fails falls back to the cache exactly as before; after
# _REFRESH_MAX_FAIL_STREAK consecutive failures (rate limit, outage) the
# process stops trying, and a per-process call budget bounds the worst case.
WX_REFRESH = os.environ.get("RW2_WX_REFRESH") == "1"
_REFRESH_MAX_FAIL_STREAK = 5
_refresh_budget = int(os.environ.get("RW2_WX_REFRESH_BUDGET", "2000"))
_refresh_fail_streak = 0
_REFRESH_TIMEOUT_S = 20
#: History re-fetch overlap: the archive's newest days arrive late and null.
_REFRESH_HIST_OVERLAP_D = 7


def refresh_allowed() -> bool:
    return (NO_FETCH and WX_REFRESH and _refresh_budget > 0
            and _refresh_fail_streak < _REFRESH_MAX_FAIL_STREAK)


def refresh_json(url: str) -> dict | None:
    """One budgeted fetch under NO_FETCH. None on failure (caller falls back)."""
    global _refresh_budget, _refresh_fail_streak
    if not refresh_allowed():
        return None
    _refresh_budget -= 1
    try:
        payload = _http_json(url, timeout=_REFRESH_TIMEOUT_S)
    except Exception:
        _refresh_fail_streak += 1
        return None
    _refresh_fail_streak = 0
    return payload


CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo"
if not CACHE_DIR.is_symlink():  # dangling symlink (unmounted volume) breaks mkdir
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo_records"
if not RECORDS_DIR.is_symlink():  # dangling symlink (unmounted volume) breaks mkdir
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "shortwave_radiation_sum",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",
    # v11 covariates: soil columns drive baseflow recession; snow_depth tracks pack.
    "soil_moisture_0_to_10cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_temperature_0_to_7cm_mean",
    "snow_depth_max",
]


def _cache_path(lat: float, lon: float, start: date, end: date, kind: str) -> Path:
    key = f"{kind}_{lat:.3f}_{lon:.3f}_{start.isoformat()}_{end.isoformat()}.json"
    return CACHE_DIR / key


def _http_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "riverwatch2/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


SCHEMA_VERSION = "v11"  # bumped when DAILY_VARS expands so we re-fetch missing cols


def _record_path(lat: float, lon: float) -> Path:
    return RECORDS_DIR / f"hist_{lat:.3f}_{lon:.3f}.json"


def fetch_history(lat: float, lon: float, start: date, end: date, *, max_age_hours: int = 24) -> pd.DataFrame:
    """Persistent incremental Open-Meteo daily archive cache.

    Per (lat, lon) we keep a record of all daily rows we've ever fetched and
    only request [last_known + 1, end] on subsequent calls. Returns rows in
    [start, end] from the merged record.
    """
    rp = _record_path(lat, lon)
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
    rec_schema = rec.get("schema_version")
    schema_stale = rec_schema != SCHEMA_VERSION  # added vars in v11

    if NO_FETCH:
        # §V16-1: forward delta only (never the backward backfill).
        if refresh_allowed() and last_known and last_known < end.isoformat():
            try:
                frm = max(start, date.fromisoformat(last_known) - timedelta(days=_REFRESH_HIST_OVERLAP_D))
            except ValueError:
                frm = None
            if frm is not None and _refresh_hist(lat, lon, have, frm, end):
                rec.update({"lat": lat, "lon": lon, "rows": have,
                            "first_known": min(have.keys()), "last_known": max(have.keys()),
                            "fetched_at": time.time()})
                rec.setdefault("schema_version", rec_schema)
                try:
                    rp.write_text(json.dumps(rec, separators=(",", ":")))
                except Exception:
                    pass
        return _slice_hist_record(have, start, end)

    needs_backward = bool(first_known) and first_known > start.isoformat()
    needs_forward = (not last_known) or (last_known < end.isoformat())

    if (
        not schema_stale
        and not needs_backward
        and last_known
        and last_known >= end.isoformat()
        and age < max_age_hours * 3600
    ):
        return _slice_hist_record(have, start, end)

    fwd_from = start
    if last_known:
        try:
            ld = date.fromisoformat(last_known)
            if ld >= start:
                fwd_from = ld - timedelta(days=2)  # re-fetch last 2 days in case provisional values were revised
        except ValueError:
            pass

    fetched_any = False
    if needs_backward:
        bwd_from = start
        bwd_to = date.fromisoformat(first_known) - timedelta(days=1) if first_known else end
        if bwd_from <= bwd_to:
            fetched_any = _fetch_and_merge_hist(lat, lon, have, bwd_from, bwd_to) or fetched_any

    if needs_forward and fwd_from <= end:
        fetched_any = _fetch_and_merge_hist(lat, lon, have, fwd_from, end) or fetched_any

    # NOTE: schema-stale backfill of historical openmeteo rows is intentionally
    # disabled here so v11.1 ships fast on existing caches. New v11 columns
    # populate naturally as each daily delta arrives going forward.

    if fetched_any and have:
        rec = {
            "lat": lat,
            "lon": lon,
            "rows": have,
            "first_known": min(have.keys()),
            "last_known": max(have.keys()),
            "fetched_at": time.time(),
            "schema_version": SCHEMA_VERSION,
        }
        rp.write_text(json.dumps(rec, separators=(",", ":")))
    elif schema_stale and rec:
        # No rows changed but we want to mark the cache as upgraded so we don't
        # re-attempt forever. Only do this if we actually attempted a fetch.
        rec["schema_version"] = SCHEMA_VERSION
        rp.write_text(json.dumps(rec, separators=(",", ":")))

    return _slice_hist_record(have, start, end)


def _hist_url(lat: float, lon: float, start: date, end: date) -> str:
    return ARCHIVE_URL + "?" + urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    })


def _refresh_hist(lat: float, lon: float, have: dict, start: date, end: date) -> bool:
    """§V16-1 forward delta under NO_FETCH. A day whose precipitation is still
    null is NOT merged: it would advance `last_known` past a day the archive
    has not published yet, and the day would then never be asked for again."""
    payload = refresh_json(_hist_url(lat, lon, start, end))
    if payload is None:
        return False
    df_new = _to_df(payload)
    if df_new.empty or "precipitation_sum" not in df_new:
        return False
    df_new = df_new[df_new["precipitation_sum"].notna()]
    return _merge_hist(df_new, have)


def _fetch_and_merge_hist(lat: float, lon: float, have: dict, start: date, end: date) -> bool:
    """Fetch Open-Meteo archive [start, end] and merge into `have`. Returns True if anything was added."""
    url = _hist_url(lat, lon, start, end)
    try:
        payload = _http_json(url)
    except Exception:
        return False
    return _merge_hist(_to_df(payload), have)


def _merge_hist(df_new: pd.DataFrame, have: dict) -> bool:
    added = False
    for _, r in df_new.iterrows():
        row = {k: (None if pd.isna(r[k]) else float(r[k]) if k != "date" else r[k].isoformat())
               for k in df_new.columns}
        k = row["date"]
        if k not in have:
            added = True
        have[k] = {col: row[col] for col in df_new.columns if col != "date"}
    return added or not df_new.empty


def _slice_hist_record(have: dict, start: date, end: date) -> pd.DataFrame:
    if not have:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    s, e = start.isoformat(), end.isoformat()
    rows = []
    for d_iso, vals in have.items():
        if not (s <= d_iso <= e):
            continue
        row = {"date": d_iso}
        for v in DAILY_VARS:
            row[v] = vals.get(v) if isinstance(vals, dict) else None
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_forecast(lat: float, lon: float, days: int = 14, *, max_age_hours: int = 3) -> pd.DataFrame:
    today = date.today()
    end = today + timedelta(days=days)
    cache = _cache_path(lat, lon, today, end, "fcst")
    if NO_FETCH:
        # Use today's exact-match forecast cache if it exists; else pick the
        # most recent fcst file for this lat/lon; else give up with empty df.
        if cache.exists():
            return _to_df(json.loads(cache.read_text()))
        payload = refresh_json(_fcst_url(lat, lon, days))  # §V16-1
        if payload is not None and (payload.get("daily") or {}).get("time"):
            cache.write_text(json.dumps(payload))
            return _to_df(payload)
        candidates = sorted(
            CACHE_DIR.glob(f"fcst_{lat:.3f}_{lon:.3f}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return _to_df(json.loads(candidates[0].read_text()))
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        payload = json.loads(cache.read_text())
    else:
        payload = _http_json(_fcst_url(lat, lon, days))
        cache.write_text(json.dumps(payload))
    return _to_df(payload)


def _fcst_url(lat: float, lon: float, days: int) -> str:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_VARS),
        "forecast_days": min(days, 16),
        "timezone": "UTC",
    }
    # v17.1: forecast model selection. ECMWF forcing beat the prior
    # default decisively on the honest harness (full-scale NSE 0.540 vs
    # 0.510, CRPS -7%); serving defaults to Open-Meteo's "best match"
    # unless RW2_FCST_MODEL pins a model (production: ecmwf_ifs025).
    model = os.environ.get("RW2_FCST_MODEL", "").strip()
    if model:
        params["models"] = model
    return FORECAST_URL + "?" + urlencode(params)


# v17.1: decoder-forcing vars for the ECMWF ensemble-mean serving path. Must
# stay identical to scripts/fetch_ens_forcings.py OUT_VARS — the eval-time
# forcing that measured +0.030 NSE full-scale over GFS on the honest harness
# (benchmarks/EXPERIMENTS.md row 4). Same units as the archive/corpus: temps
# °C, precipitation_sum mm/day, shortwave_radiation_sum MJ/m²/day.
ENS_DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
]

_ENS_MEMBER_RE = re.compile(r"_member\d+$")


def fetch_forecast_ecmwf_ens(lat: float, lon: float, days: int = 14, *, max_age_hours: int = 3) -> pd.DataFrame:
    """ECMWF IFS ENS (0.25°) ensemble-MEAN daily forecast via Open-Meteo's
    Ensemble API.

    Returns a frame of ["date"] + ENS_DAILY_VARS where each var is the mean
    across all ensemble members (control + member01..member50, 51 total). The
    Ensemble API serves these five vars as native *daily* aggregates per
    member — daily max/min are computed per member first, then averaged, which
    reproduces exactly how the eval-time ens-mean forcing archive was built.
    IMPORTANT: this is the IFS ENS member mean, NOT HRES deterministic — the
    +0.030 NSE basis was measured on the ensemble mean.
    """
    today = date.today()
    end = today + timedelta(days=days)
    cache = _cache_path(lat, lon, today, end, "ensfcst")
    if NO_FETCH:
        # Mirror fetch_forecast: today's exact-match cache, else the most
        # recent ensfcst file for this lat/lon, else an empty frame.
        if cache.exists():
            return _ens_mean_df(json.loads(cache.read_text()))
        payload = refresh_json(_ens_url(lat, lon, days))  # §V16-1
        if payload is not None and (payload.get("daily") or {}).get("time"):
            cache.write_text(json.dumps(payload))
            return _ens_mean_df(payload)
        candidates = sorted(
            CACHE_DIR.glob(f"ensfcst_{lat:.3f}_{lon:.3f}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return _ens_mean_df(json.loads(candidates[0].read_text()))
        return pd.DataFrame(columns=["date"] + ENS_DAILY_VARS)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        payload = json.loads(cache.read_text())
    else:
        payload = _http_json(_ens_url(lat, lon, days))
        cache.write_text(json.dumps(payload))
    return _ens_mean_df(payload)


def _ens_url(lat: float, lon: float, days: int) -> str:
    return ENSEMBLE_URL + "?" + urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(ENS_DAILY_VARS),
        "models": "ecmwf_ifs025",
        "forecast_days": min(days, 15),  # IFS ENS horizon cap on this API
        "timezone": "UTC",
    })


def _ens_mean_df(payload: dict) -> pd.DataFrame:
    """Collapse the Ensemble API's per-member daily columns (`<var>` for the
    control run plus `<var>_memberNN`) into the per-day ensemble mean."""
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame(columns=["date"] + ENS_DAILY_VARS)
    raw = pd.DataFrame(daily)
    out = pd.DataFrame({"date": pd.to_datetime(raw["time"]).dt.date})
    for var in ENS_DAILY_VARS:
        cols = [c for c in raw.columns if _ENS_MEMBER_RE.sub("", c) == var]
        if cols:
            out[var] = raw[cols].astype(float).mean(axis=1, skipna=True)
        else:
            out[var] = float("nan")
    return out


def _to_df(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
