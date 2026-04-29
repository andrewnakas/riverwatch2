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


def fetch_history(lat: float, lon: float, start: date, end: date, *, max_age_hours: int = 24) -> pd.DataFrame:
    cache = _cache_path(lat, lon, start, end, "hist")
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        payload = json.loads(cache.read_text())
    else:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "timezone": "UTC",
        }
        url = ARCHIVE_URL + "?" + urlencode(params)
        payload = _http_json(url)
        cache.write_text(json.dumps(payload))
    return _to_df(payload)


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
