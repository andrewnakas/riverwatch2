"""v14.5b: NOAA NBM (National Blend of Models) forecast covariates.

Open-Meteo's `ncep_nbm_conus` model is NBM's CONUS deterministic+probabilistic
daily aggregation. Returns exactly four useful columns out to 11 days:

    nbm_precip_sum        — daily precip total (mm)
    nbm_precip_pop_mean   — mean POP across the day (%)
    nbm_precip_pop_max    — max POP across the day (%)
    nbm_tmean             — mean 2m air temp (C)

Horizon caps at 11 days (NBM upper bound). Days 12-14 stay NaN — LightGBM
handles the missing rows natively as a separate split direction.

Why a *separate* fetch from `app/weather.py`:
- weather.py uses the default Open-Meteo blend (mostly GFS/ECMWF). That's a
  reasonable hydrology forcing but lacks any uncertainty signal.
- NBM is the U.S. operational blend with a calibrated POP. POP is the closest
  proxy we have for forecast spread without going to the full NOMADS qmd
  GRIB extraction. When NBM POP is high the upstream forecast is uncertain,
  and the v14.4 stacker can learn "lean on NWM/persistence more then."
- Keeping the two caches separate makes it easy to disable NBM with
  `RW2_NBM_OFF=1` if Open-Meteo throttles us.

The values returned here are merged into wx_combined alongside the regular
Open-Meteo forecast — so runoff_ridge / lgbm_pooled / timesfm_xreg pick them
up via `_build_features`. The v14.4 stacker also reads h+3 / h+7 NBM mean +
POP directly so the meta-learner sees the uncertainty signal explicitly.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

NO_FETCH = os.environ.get("RW2_NO_FETCH") == "1"
NBM_OFF = os.environ.get("RW2_NBM_OFF") == "1"

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "nbm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NBM_URL = "https://api.open-meteo.com/v1/forecast"
NBM_MAX_DAYS = 11

# Open-Meteo daily variables we request. The mean/max POP pair gives the
# stacker a coarse sense of "is the day likely wet" vs "could go either way."
DAILY_VARS = (
    "precipitation_sum",
    "precipitation_probability_mean",
    "precipitation_probability_max",
    "temperature_2m_mean",
)

# Output column names — namespaced with `nbm_` so they coexist with the
# Open-Meteo blend forecast in wx_combined without collisions.
OUT_COLS = (
    "nbm_precip_sum",
    "nbm_precip_pop_mean",
    "nbm_precip_pop_max",
    "nbm_tmean",
)
_RENAME = dict(zip(DAILY_VARS, OUT_COLS))


def _cache_path(lat: float, lon: float, days: int) -> Path:
    return CACHE_DIR / f"nbm_{lat:.3f}_{lon:.3f}_{days}.json"


def _http_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "riverwatch2/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_forecast(lat: float, lon: float, days: int = NBM_MAX_DAYS,
                   *, max_age_hours: int = 3) -> pd.DataFrame:
    """Pull NBM CONUS daily forecast and return a DataFrame indexed by `date`.

    Returns columns: ['date', 'nbm_precip_sum', 'nbm_precip_pop_mean',
    'nbm_precip_pop_max', 'nbm_tmean']. Empty DataFrame on any failure or
    when NBM_OFF is set.
    """
    if NBM_OFF:
        return _empty()

    days = max(1, min(int(days), NBM_MAX_DAYS))
    cache = _cache_path(lat, lon, days)

    if NO_FETCH:
        if cache.exists():
            return _to_df(json.loads(cache.read_text()))
        # Fall back to any recent cache for this lat/lon.
        candidates = sorted(
            CACHE_DIR.glob(f"nbm_{lat:.3f}_{lon:.3f}_*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if candidates:
            return _to_df(json.loads(candidates[0].read_text()))
        return _empty()

    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        try:
            return _to_df(json.loads(cache.read_text()))
        except Exception:
            pass

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_VARS),
        "models": "ncep_nbm_conus",
        "forecast_days": days,
        "timezone": "UTC",
    }
    url = NBM_URL + "?" + urlencode(params)
    try:
        payload = _http_json(url)
    except Exception:
        return _empty()
    try:
        cache.write_text(json.dumps(payload))
    except Exception:
        pass
    return _to_df(payload)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", *OUT_COLS])


def _to_df(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return _empty()
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date", **_RENAME})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Drop any unexpected columns (Open-Meteo can return extras with model suffixes
    # we didn't ask for).
    keep = ["date"] + [c for c in OUT_COLS if c in df.columns]
    df = df[keep]
    for c in OUT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def patch_wx_fcst(wx_fcst: pd.DataFrame, nbm_df: pd.DataFrame) -> pd.DataFrame:
    """Replace Open-Meteo forecast values with NBM where NBM has data.

    Open-Meteo's default forecast is a generic global blend; NBM is NOAA's
    operational CONUS blend, calibrated against U.S. surface observations.
    For dates where NBM has values (h=1..11) we prefer NBM; for the tail
    days (12-14) and off-CONUS gauges we keep Open-Meteo. Columns patched:

        precipitation_sum     ← NBM nbm_precip_sum
        temperature_2m_mean   ← NBM nbm_tmean

    The other Open-Meteo columns (rain_sum, snowfall_sum, soil_*, et0_*,
    snow_depth_max, etc.) NBM doesn't directly expose at the daily level,
    so they pass through unchanged. _build_features still gets a complete
    forecast tail — just with the precip/tmean inputs upgraded.

    Returns a copy of wx_fcst with the patches applied. Original input is
    not mutated.
    """
    if wx_fcst is None or wx_fcst.empty or nbm_df is None or nbm_df.empty:
        return wx_fcst.copy() if wx_fcst is not None else wx_fcst
    wx = wx_fcst.copy()
    # Build {iso_date: row} for fast lookup.
    nbm_by_date: dict = {}
    for _, r in nbm_df.iterrows():
        d = r["date"]
        d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
        nbm_by_date[d_iso] = r

    def _maybe(col_src, col_dst):
        if col_dst not in wx.columns:
            return
        for i, row in wx.iterrows():
            d = row["date"]
            d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
            n = nbm_by_date.get(d_iso)
            if n is None:
                continue
            v = n.get(col_src)
            if v is None or pd.isna(v):
                continue
            wx.at[i, col_dst] = float(v)

    _maybe("nbm_precip_sum", "precipitation_sum")
    _maybe("nbm_tmean", "temperature_2m_mean")
    return wx


def horizon_features(nbm_df: pd.DataFrame, today: date, horizon: int) -> dict:
    """Pull h-indexed NBM features for the v14.4 stacker.

    Returns `{f"nbm_apcp_h{h}": ..., f"nbm_pop_h{h}": ..., f"nbm_tmean_h{h}": ...}`
    for h in 1..min(horizon, NBM_MAX_DAYS). Missing days are NaN-valued so the
    stacker LightGBM can route them as a separate split direction.
    """
    out: dict = {}
    if nbm_df is None or nbm_df.empty:
        for h in range(1, horizon + 1):
            out[f"nbm_apcp_h{h}"] = float("nan")
            out[f"nbm_pop_h{h}"] = float("nan")
            out[f"nbm_tmean_h{h}"] = float("nan")
        return out
    by_date = {}
    for _, r in nbm_df.iterrows():
        d = r["date"]
        if hasattr(d, "isoformat"):
            d_iso = d.isoformat() if hasattr(d, "year") else str(d)
        else:
            d_iso = str(d)
        by_date[d_iso] = r
    for h in range(1, horizon + 1):
        target = (today + timedelta(days=h)).isoformat()
        r = by_date.get(target)
        if r is None:
            out[f"nbm_apcp_h{h}"] = float("nan")
            out[f"nbm_pop_h{h}"] = float("nan")
            out[f"nbm_tmean_h{h}"] = float("nan")
            continue
        # Use POP-max (more aggressive uncertainty signal than POP-mean).
        out[f"nbm_apcp_h{h}"] = float(r.get("nbm_precip_sum")) if pd.notna(r.get("nbm_precip_sum")) else float("nan")
        pop_v = r.get("nbm_precip_pop_max")
        if pop_v is None or pd.isna(pop_v):
            pop_v = r.get("nbm_precip_pop_mean")
        out[f"nbm_pop_h{h}"] = float(pop_v) if pd.notna(pop_v) else float("nan")
        out[f"nbm_tmean_h{h}"] = float(r.get("nbm_tmean")) if pd.notna(r.get("nbm_tmean")) else float("nan")
    return out
