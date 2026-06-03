"""USGS NWIS daily-discharge fetcher with persistent incremental caching.

Per-site records are stored as compact JSON keyed only by site number, and
grow over time. On each call we read the existing record and request only
[last_known_date+1, end] from USGS; results merge back in. This means an
hourly rebuild of 470 stations only fetches a single new day per station,
not the full 5-year window.
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

NO_FETCH = os.environ.get("RW2_NO_FETCH") == "1"
# v15.8: even when NO_FETCH is set (so we don't repeat the cold 100yr
# backfill on every build), allow a small forward-only refresh of the
# last N days of dv so the deployed chart's tail tracks reality. Without
# this, the cached "last_known" frozen at bootstrap drifts further from
# present every day and the historical chart appears stale (in some
# cases 10+ days behind USGS's published dv).
NO_FETCH_RECENT_DAYS = int(os.environ.get("RW2_NO_FETCH_RECENT_DAYS", "30"))

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

    if NO_FETCH:
        # v15.8: previously this was a hard short-circuit. That meant the
        # cache's last_known froze at bootstrap, even though `end` always
        # asks for today. Allow a small forward-only fetch over the
        # recent window (default 30 days) so the chart's tail follows
        # USGS dv updates. Deep history (>30 days back) still serves
        # entirely from cache — we never repeat the cold backfill here.
        today = date.today()
        recent_cutoff = today - timedelta(days=NO_FETCH_RECENT_DAYS)
        # Only fetch if (a) last_known is older than the requested end,
        # (b) the requested window overlaps the recent cutoff, and (c)
        # we haven't refreshed within the last `max_age_hours`.
        fetched_at = rec.get("fetched_at", 0)
        age = time.time() - float(fetched_at) if fetched_at else float("inf")
        last_known_date = None
        if last_known:
            try:
                last_known_date = date.fromisoformat(last_known)
            except ValueError:
                last_known_date = None
        wants_forward = (
            end >= recent_cutoff
            and (last_known_date is None or last_known_date < end)
            and age >= max_age_hours * 3600
        )
        if wants_forward:
            fwd_from = max(recent_cutoff, last_known_date or recent_cutoff)
            if fwd_from <= end:
                if _fetch_and_merge(site_no, have, fwd_from, end) and have:
                    rec = {
                        "site_no": site_no,
                        "rows": have,
                        "first_known": min(have.keys()),
                        "last_known": max(have.keys()),
                        "fetched_at": time.time(),
                    }
                    _save_record(site_no, rec)
        return _slice_record(have, start, end)

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
    """Fetch [start, end] from USGS and merge into `have`. Returns True if any
    new rows were added.

    v15.0-retry: at high concurrency (16-32 shards × 600 stations/shard
    cold-fetching 100yr windows) the USGS dv endpoint rate-limits and
    occasionally 503s. We chunk cold windows into 10-year slices and retry
    each chunk with exponential backoff before giving up. This converts
    one transient blip from "this gauge is permanently failed" into
    "this chunk re-fetches on the next run."
    """
    span_days = (end - start).days
    # Chunk anything bigger than ~12 years so a single 503 doesn't cost
    # us 100 years. The dv endpoint streams the full range in one
    # response, so smaller chunks also keep peak memory & timeout pressure
    # down.
    CHUNK_YEARS = 10
    if span_days > CHUNK_YEARS * 366:
        cur = start
        any_added = False
        while cur <= end:
            chunk_end = min(end, cur + timedelta(days=CHUNK_YEARS * 366 - 1))
            any_added = _fetch_chunk_with_retry(site_no, have, cur, chunk_end) or any_added
            cur = chunk_end + timedelta(days=1)
        return any_added
    return _fetch_chunk_with_retry(site_no, have, start, end)


def _fetch_chunk_with_retry(
    site_no: str, have: dict[str, float], start: date, end: date,
    *, retries: int = 4,
) -> bool:
    """One [start, end] fetch with exponential backoff. Returns True iff new
    rows were merged into `have`. A failed fetch (after all retries) returns
    False but does NOT poison `have` — the next run can try again, since the
    cache record's last_known/first_known are only updated on success."""
    params = {
        "format": "json",
        "sites": site_no,
        "startDT": start.isoformat(),
        "endDT": end.isoformat(),
        "parameterCd": "00060",
        "siteStatus": "all",
    }
    url = DV_URL + "?" + urlencode(params)
    last_exc: Exception | None = None
    for attempt in range(retries):
        # Initial jitter before every attempt (including the first) so 600+
        # shards don't synchronize their ramp.
        time.sleep(0.05 + random.random() * 0.20 + attempt * (1.0 + random.random() * 1.0) * (2 ** attempt))
        try:
            payload = _http_json(url)
            break
        except Exception as exc:
            last_exc = exc
            continue
    else:
        # All retries exhausted — record the failure path so build logs are
        # actionable, but don't raise.
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


def flag_suspect_jumps(
    df: pd.DataFrame,
    *,
    jump_factor: float = 50.0,
    min_cfs: float = 10.0,
) -> "pd.Series":
    """AUDIT (Phase 5): flag physically-implausible single-day spikes that are
    almost certainly gauge malfunction / data-entry errors rather than real
    hydrology.

    A day is suspect when it is BOTH >= `jump_factor`x the previous day AND
    >= `jump_factor`x the next day (an isolated spike that immediately
    collapses) — a real flood ramps and recedes over multiple days, so it
    won't satisfy both. `min_cfs` floors the comparison so noise on
    near-zero ephemeral streams (0.01 -> 1.0 cfs) doesn't trip the flag.

    Returns a boolean Series aligned to `df.index` (True == suspect). This is a
    FLAG, not a filter: we never silently drop a value, because deleting a real
    flood peak is worse than keeping a rare bad point. Callers decide what to do
    (e.g. record a count in forecast notes for observability).
    """
    n = len(df)
    flags = pd.Series([False] * n, index=df.index)
    if n < 3:
        return flags
    q = df["q_cfs"].astype(float).to_numpy()
    for i in range(1, n - 1):
        cur = q[i]
        prev = max(q[i - 1], min_cfs)
        nxt = max(q[i + 1], min_cfs)
        if cur >= jump_factor * prev and cur >= jump_factor * nxt:
            flags.iloc[i] = True
    return flags


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
