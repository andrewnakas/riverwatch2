"""USGS NWIS daily-discharge fetcher with on-disk caching."""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "usgs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def _cache_path(site_no: str, start: date, end: date, kind: str = "dv") -> Path:
    return CACHE_DIR / f"{kind}_{site_no}_{start.isoformat()}_{end.isoformat()}.json"


def _http_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "riverwatch2/0.1", "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_daily_discharge(site_no: str, start: date, end: date, *, max_age_hours: int = 6) -> pd.DataFrame:
    """Daily discharge (parameter 00060) as a tidy DataFrame: date, q_cfs.

    Cached on disk. Cache is honored if newer than max_age_hours.
    """
    cache = _cache_path(site_no, start, end, "dv")
    if cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < max_age_hours * 3600:
            return _parse_dv(json.loads(cache.read_text()))

    params = {
        "format": "json",
        "sites": site_no,
        "startDT": start.isoformat(),
        "endDT": end.isoformat(),
        "parameterCd": "00060",
        "siteStatus": "all",
    }
    url = DV_URL + "?" + urlencode(params)
    payload = _http_json(url)
    cache.write_text(json.dumps(payload))
    return _parse_dv(payload)


def _parse_dv(payload: dict) -> pd.DataFrame:
    rows = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        for v in ts.get("values", []):
            for entry in v.get("value", []):
                try:
                    val = float(entry["value"])
                except (TypeError, ValueError):
                    continue
                if val < 0:
                    continue
                d = entry["dateTime"][:10]
                rows.append({"date": d, "q_cfs": val})
    if not rows:
        return pd.DataFrame(columns=["date", "q_cfs"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def fetch_recent_instantaneous(site_no: str, hours: int = 168) -> pd.DataFrame:
    """Instantaneous (15-min) discharge for the last `hours`. Used to display 'now'."""
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    params = {
        "format": "json",
        "sites": site_no,
        "startDT": start.strftime("%Y-%m-%dT%H:%MZ"),
        "endDT": end.strftime("%Y-%m-%dT%H:%MZ"),
        "parameterCd": "00060",
    }
    url = IV_URL + "?" + urlencode(params)
    try:
        payload = _http_json(url, timeout=30)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "q_cfs"])
    rows = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        for v in ts.get("values", []):
            for entry in v.get("value", []):
                try:
                    val = float(entry["value"])
                except (TypeError, ValueError):
                    continue
                if val < 0:
                    continue
                rows.append({"timestamp": entry["dateTime"], "q_cfs": val})
    if not rows:
        return pd.DataFrame(columns=["timestamp", "q_cfs"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
