"""NOAA National Water Model (NWM) forecast fetcher.

Hits the NOAA National Water Prediction Service (NWPS) REST API at
``api.water.noaa.gov/nwps/v1`` to pull NWM medium-range deterministic
streamflow forecasts for a given NHDPlus COMID. Falls back to short-range
+ long-range mean to cover the full 14-day horizon.

Returns a daily-mean CFS series aligned to UTC dates. The crosswalk from
USGS site number → COMID is built offline by `scripts/build_nwm_crosswalk.py`
and shipped in `data/nwm_crosswalk.json`.

NWM is a forecast-only source (no analysis/history blending here); it slots
in as a 6th ensemble member alongside persistence, runoff_ridge, chronos_bolt,
ttm, and timesfm. Stations without a COMID — or COMIDs that NWPS does not
serve forecasts for (the API only forecasts reaches with associated NWS
forecast points, ~10k of the 2.7M NWM reaches) — return None and the blend
gracefully drops the member.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CROSSWALK_PATH = ROOT / "data" / "nwm_crosswalk.json"
CACHE_DIR = ROOT / "data" / "cache" / "nwm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API = "https://api.water.noaa.gov/nwps/v1"
USER_AGENT = "riverwatch2/0.1 (treesixtyweather@gmail.com)"
NWM_MISSING = -9999  # NWPS sentinel for missing flow values
CACHE_TTL_SEC = 6 * 3600  # NWM medium_range cycles every 6h, so this is conservative


_CROSSWALK_CACHE: Optional[dict] = None


def _no_fetch() -> bool:
    """NWM fetching is gated separately because it's fast (~100ms/gauge) and
    forecast-only — there's no historical archive to backfill from cache.
    The site-wide RW2_NO_FETCH=1 turns off USGS / OpenMeteo / SNOTEL fetches
    but RW2_NWM_ALLOW_FETCH=1 keeps the NWM live-forecast pull on."""
    if os.environ.get("RW2_NWM_ALLOW_FETCH") == "1":
        return False
    return os.environ.get("RW2_NO_FETCH") == "1"


def _enabled() -> bool:
    """Master switch — defaults OFF so we ship the crosswalk first, then
    flip a CI flag to enable. Lets us land the code without paying for an
    untested API dependency on a hot deploy."""
    return os.environ.get("RW2_ENABLE_NWM") == "1"


def _load_crosswalk() -> dict:
    global _CROSSWALK_CACHE
    if _CROSSWALK_CACHE is not None:
        return _CROSSWALK_CACHE
    if not CROSSWALK_PATH.exists():
        _CROSSWALK_CACHE = {}
        return _CROSSWALK_CACHE
    try:
        _CROSSWALK_CACHE = json.loads(CROSSWALK_PATH.read_text())
    except Exception:
        _CROSSWALK_CACHE = {}
    return _CROSSWALK_CACHE


def comid_for_usgs(usgs_id: str) -> Optional[str]:
    cw = _load_crosswalk()
    entry = cw.get(usgs_id)
    if not entry or "comid" not in entry:
        return None
    return str(entry["comid"])


def _http_json(url: str, *, timeout: int = 15, retries: int = 2) -> Optional[object]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                if resp.status >= 400:
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return None


def _cache_path(comid: str, series: str) -> Path:
    return CACHE_DIR / f"{comid}_{series}.json"


def _read_cache(comid: str, series: str) -> Optional[dict]:
    p = _cache_path(comid, series)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > CACHE_TTL_SEC:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_cache(comid: str, series: str, payload: dict) -> None:
    try:
        _cache_path(comid, series).write_text(json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass


def _series_to_daily_cfs(payload: dict, key: str) -> pd.DataFrame:
    """Pull rows from `payload[key]` and return a daily-mean CFS DataFrame.

    NWPS schema varies by series:
      - medium_range / long_range: ensemble — rows under `.mean.data[]` plus
        per-member arrays
      - medium_range_blend / short_range / analysis_assimilation: deterministic
        — rows under `.series.data[]`
    We try both, take whichever has data. NWPS uses ft³/s, same as our
    internal convention so no conversion needed.
    """
    block = (payload or {}).get(key) or {}
    rows = None
    for inner_key in ("series", "mean"):
        inner = block.get(inner_key) if isinstance(block, dict) else None
        if isinstance(inner, dict) and inner.get("data"):
            rows = inner["data"]
            break
    if not rows:
        return pd.DataFrame(columns=["date", "q_cfs"])
    df = pd.DataFrame(rows)
    if df.empty or "validTime" not in df or "flow" not in df:
        return pd.DataFrame(columns=["date", "q_cfs"])
    df["q_cfs"] = pd.to_numeric(df["flow"], errors="coerce")
    df = df[df["q_cfs"] > NWM_MISSING + 1]  # drop -9999 sentinels
    if df.empty:
        return pd.DataFrame(columns=["date", "q_cfs"])
    ts = pd.to_datetime(df["validTime"], utc=True, errors="coerce")
    df = df.assign(date=ts.dt.tz_convert("UTC").dt.date).dropna(subset=["date"])
    return df.groupby("date", as_index=False)["q_cfs"].mean()


def fetch_streamflow(comid: str, series: str = "medium_range") -> Optional[pd.DataFrame]:
    """Fetch one NWM streamflow series (medium_range, short_range, long_range,
    or medium_range_blend). Returns daily-mean DataFrame with columns
    ['date', 'q_cfs'] or None if the reach has no forecast available."""
    if _no_fetch():
        cached = _read_cache(comid, series)
        if cached:
            df = _series_to_daily_cfs(cached, _series_key(series))
            return df if not df.empty else None
        return None

    cached = _read_cache(comid, series)
    if cached:
        df = _series_to_daily_cfs(cached, _series_key(series))
        if not df.empty:
            return df

    url = f"{API}/reaches/{comid}/streamflow?series={series}"
    payload = _http_json(url)
    if payload is None:
        return None
    _write_cache(comid, series, payload)
    df = _series_to_daily_cfs(payload, _series_key(series))
    return df if not df.empty else None


def _series_key(series: str) -> str:
    """Map URL-style series name to the JSON key NWPS uses in the response."""
    return {
        "medium_range": "mediumRange",
        "medium_range_blend": "mediumRangeBlend",
        "short_range": "shortRange",
        "long_range": "longRange",
        "analysis_assimilation": "analysisAssimilation",
    }.get(series, series)


_HINDCAST_MIN_OVERLAP_DAYS = 7


def hindcast_mae(usgs_id: str, q_hist: pd.DataFrame, *, lookback_days: int = 30) -> Optional[float]:
    """Estimate NWM's MAE on this gauge by comparing recent NWM analysis_assimilation
    flow to the gauge's observed daily-mean flow. NWM's analysis is its post-fact
    best estimate of true streamflow given assimilated observations, so this is
    a *floor* on its forecast skill — actual h+N forecast MAE will be higher
    due to forcing uncertainty over the horizon. Returns CFS or None.

    v13.6: require at least 7 overlapping days. With only 1-2 day overlap on
    new stations, sampling can yield absurdly low MAE (we saw 0.04 cfs on
    01408029) that lets snap-to-winner give NWM 90% of the blend weight on
    nothing. Fewer than 7 days → return None and the caller falls back to the
    persistence-based MAE estimate.
    """
    if not _enabled():
        return None
    comid = comid_for_usgs(usgs_id)
    if not comid:
        return None
    nwm = fetch_streamflow(comid, "analysis_assimilation")
    if nwm is None or nwm.empty or q_hist.empty:
        return None
    nwm = nwm.tail(lookback_days)
    obs = q_hist[["date", "q_cfs"]].copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.date
    merged = nwm.merge(obs, on="date", suffixes=("_nwm", "_obs"))
    if len(merged) < _HINDCAST_MIN_OVERLAP_DAYS:
        return None
    err = (merged["q_cfs_nwm"] - merged["q_cfs_obs"]).abs()
    return float(err.mean()) if len(err) else None


def forecast_daily_cfs(usgs_id: str, horizon: int = 14) -> Optional[list[float]]:
    """Top-level: produce `horizon` days of NWM streamflow forecast for a USGS
    gauge, in CFS, indexed by t+1..t+horizon. Returns None if NWM has no
    forecast for this gauge.

    Strategy:
      1. medium_range_blend (RFC-blended NWM, preferred operationally) →
         covers ~10 days
      2. fall back to medium_range (raw NWM mem1) if blend is missing
      3. extend to day 14 with long_range mean (6-hourly, ~30d)
    """
    if not _enabled():
        return None
    comid = comid_for_usgs(usgs_id)
    if not comid:
        return None

    today = datetime.now(timezone.utc).date()

    # medium_range_blend is the RFC-blended NWM (preferred operationally) —
    # 10-11 days hourly. long_range mean is 6-hourly out to ~30 days, used to
    # fill h11..h14 since blend doesn't reach that far. Two calls is the
    # minimum viable set. Skip the raw medium_range — blend is strictly better
    # when available and falls back to medium_range internally on the NOAA side.
    primary = fetch_streamflow(comid, "medium_range_blend")
    long_range = fetch_streamflow(comid, "long_range")

    if primary is None and long_range is None:
        return None

    # Build a date→cfs map preferring primary over long_range.
    by_date: dict[date, float] = {}
    if long_range is not None:
        for row in long_range.itertuples(index=False):
            by_date[row.date] = float(row.q_cfs)
    if primary is not None:
        for row in primary.itertuples(index=False):
            by_date[row.date] = float(row.q_cfs)

    out: list[float] = []
    for h in range(1, horizon + 1):
        target = today + pd.Timedelta(days=h)
        target = target.date() if hasattr(target, "date") else target
        if target in by_date:
            out.append(by_date[target])
        else:
            out.append(float("nan"))

    if all(pd.isna(v) for v in out):
        return None
    # Forward-fill any gaps (long_range is 6-hourly so daily aggregation can
    # leave one or two NaNs at the boundary).
    last_valid = None
    for i, v in enumerate(out):
        if pd.isna(v):
            if last_valid is not None:
                out[i] = last_valid
        else:
            last_valid = v
    # Backward fill any leading NaNs.
    last_valid = None
    for i in range(len(out) - 1, -1, -1):
        if pd.isna(out[i]):
            if last_valid is not None:
                out[i] = last_valid
        else:
            last_valid = out[i]
    if any(pd.isna(v) for v in out):
        return None
    return out
