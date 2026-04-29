"""Open-Meteo weather fetchers (historical + short-range forecast).

Open-Meteo is free, no key, returns daily aggregations.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo_records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
]


def _cache_path(lat: float, lon: float, start: date, end: date, kind: str) -> Path:
    key = f"{kind}_{lat:.3f}_{lon:.3f}_{start.isoformat()}_{end.isoformat()}.json"
    return CACHE_DIR / key


def _http_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "riverwatch2/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
    fetched_at = rec.get("fetched_at", 0)
    age = time.time() - float(fetched_at) if fetched_at else float("inf")

    fetch_from = start
    if last_known:
        try:
            ld = date.fromisoformat(last_known)
            if ld >= start:
                fetch_from = ld - timedelta(days=2)  # re-fetch last 2 days in case provisional values were revised
        except ValueError:
            pass

    if last_known and last_known >= end.isoformat() and age < max_age_hours * 3600:
        return _slice_hist_record(have, start, end)

    if fetch_from <= end:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": fetch_from.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "timezone": "UTC",
        }
        url = ARCHIVE_URL + "?" + urlencode(params)
        try:
            payload = _http_json(url)
        except Exception:
            return _slice_hist_record(have, start, end)
        df_new = _to_df(payload)
        for _, r in df_new.iterrows():
            row = {k: (None if pd.isna(r[k]) else float(r[k]) if k != "date" else r[k].isoformat())
                   for k in df_new.columns}
            have[row["date"]] = {k: row[k] for k in df_new.columns if k != "date"}
        if have:
            new_last = max(have.keys())
            rec = {
                "lat": lat,
                "lon": lon,
                "rows": have,
                "last_known": new_last,
                "fetched_at": time.time(),
            }
            rp.write_text(json.dumps(rec, separators=(",", ":")))

    return _slice_hist_record(have, start, end)


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
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        payload = json.loads(cache.read_text())
    else:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "daily": ",".join(DAILY_VARS),
            "forecast_days": min(days, 16),
            "timezone": "UTC",
        }
        url = FORECAST_URL + "?" + urlencode(params)
        payload = _http_json(url)
        cache.write_text(json.dumps(payload))
    return _to_df(payload)


def _to_df(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
