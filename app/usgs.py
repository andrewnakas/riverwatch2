"""USGS NWIS daily-discharge fetcher with persistent incremental caching.

Per-site records are stored as compact JSON keyed only by site number, and
grow over time. On each call we read the existing record and request only
[last_known_date+1, end] from USGS; results merge back in. This means an
hourly rebuild of 470 stations only fetches a single new day per station,
not the full 5-year window.
"""
from __future__ import annotations

import json
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "usgs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "usgs_records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def _record_path(site_no: str) -> Path:
    return RECORDS_DIR / f"{site_no}.json"


def _http_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "riverwatch2/0.1", "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_record(site_no: str) -> dict:
    """Load the persistent per-site record. Returns {} if none exists."""
    p = _record_path(site_no)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_record(site_no: str, rec: dict) -> None:
    _record_path(site_no).write_text(json.dumps(rec, separators=(",", ":")))


def fetch_daily_discharge(site_no: str, start: date, end: date, *, max_age_hours: int = 6) -> pd.DataFrame:
    """Daily discharge (parameter 00060) as a tidy DataFrame: date, q_cfs.

    Persistent incremental cache: on each call we extend the per-site record
    with whatever is missing between [last_known + 1, end] (or the full
    [start, end] window on first run), then return the requested window from
    the merged record.
    """
    rec = _load_record(site_no)
    have: dict[str, float] = rec.get("rows", {})
    last_known = rec.get("last_known")  # ISO string of latest date we hold
    first_known = rec.get("first_known") or (min(have.keys()) if have else None)

    needs_backward = bool(first_known) and first_known > start.isoformat()
    needs_forward = (not last_known) or (last_known < end.isoformat())

    fetched_at = rec.get("fetched_at", 0)
    age = time.time() - float(fetched_at) if fetched_at else float("inf")
    if not needs_backward and last_known and last_known >= end.isoformat() and age < max_age_hours * 3600:
        return _slice_record(have, start, end)

    fwd_from = start
    if last_known:
        try:
            ld = date.fromisoformat(last_known)
            if ld >= start:
                fwd_from = ld
        except ValueError:
            pass

    fetched_any = False
    if needs_backward:
        bwd_from = start
        bwd_to = (date.fromisoformat(first_known) - timedelta(days=1)) if first_known else end
        if bwd_from <= bwd_to:
            fetched_any = _fetch_and_merge(site_no, have, bwd_from, bwd_to) or fetched_any

    if needs_forward and fwd_from <= end:
        fetched_any = _fetch_and_merge(site_no, have, fwd_from, end) or fetched_any

    if fetched_any and have:
        rec = {
            "site_no": site_no,
            "rows": have,
            "first_known": min(have.keys()),
            "last_known": max(have.keys()),
            "fetched_at": time.time(),
        }
        _save_record(site_no, rec)

    return _slice_record(have, start, end)


def _fetch_and_merge(site_no: str, have: dict[str, float], start: date, end: date) -> bool:
    """Fetch [start, end] from USGS and merge into `have`. Returns True if anything was added."""
    params = {
        "format": "json",
        "sites": site_no,
        "startDT": start.isoformat(),
        "endDT": end.isoformat(),
        "parameterCd": "00060",
        "siteStatus": "all",
    }
    url = DV_URL + "?" + urlencode(params)
    time.sleep(0.05 + random.random() * 0.10)
    try:
        payload = _http_json(url)
    except Exception:
        return False
    df_new = _parse_dv(payload)
    added = False
    for _, r in df_new.iterrows():
        k = r["date"].isoformat()
        if k not in have:
            added = True
        have[k] = float(r["q_cfs"])
    return added or not df_new.empty


def _slice_record(have: dict[str, float], start: date, end: date) -> pd.DataFrame:
    """Return rows in [start, end] from the merged per-site record."""
    if not have:
        return pd.DataFrame(columns=["date", "q_cfs"])
    s, e = start.isoformat(), end.isoformat()
    rows = [{"date": k, "q_cfs": v} for k, v in have.items() if s <= k <= e]
    if not rows:
        return pd.DataFrame(columns=["date", "q_cfs"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


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
