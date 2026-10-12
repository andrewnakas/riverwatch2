"""Open-Meteo forcing shaped for the MODERN-1 `nldasm9` member.

`app/mblstm_modern.py` needs 6 daily variables. Five are already in
`app/weather.py:DAILY_VARS`; the sixth, `vapor_pressure`, is not available from
Open-Meteo under that name at all. Both the archive and the forecast endpoints
DO serve `dew_point_2m_mean`, from which vapour pressure follows exactly
(Tetens over water), so this module fetches the 6th channel and lets
`mblstm_modern._ensure_vapor_pressure()` do the conversion.

WHY A SEPARATE MODULE rather than extending `weather.DAILY_VARS`: that list is
keyed by `weather.SCHEMA_VERSION`, and its comment is explicit -- the version is
"bumped when DAILY_VARS expands so we re-fetch missing cols". Expanding it would
invalidate the cached history of all 9,851 served gauges and force a full
re-fetch in CI. This member serves 490 gauges, so it pays for its own extra
channel in its own cache namespace and leaves the production member's cache
untouched.

⚠️ TRAIN/SERVE PRODUCT GAP, UNMEASURED AS OF 2026-10-09. `nldasm9` was trained
on NLDAS-2 **areal** (basin-mean) forcing. This module serves Open-Meteo
**point** forcing at the gauge coordinate, which differs on two axes at once:
product (NLDAS-2 vs ERA5-family) and sampling (basin-mean vs gauge pixel). The
sampling axis alone was measured on a matched pair at **+0.013004 median peak r
in areal's favour** (CI [+0.010883, +0.014501], breadth 0.883 --
`benchmarks/b0_sampling_verdict_conus404.json`), i.e. point forcing is the
weaker input. The cost in NSE terms has NOT been measured; that is Gate 1. Until
it is, anything served through this path must be labelled as running on
substituted forcing, never with the record's 0.905829.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from . import weather

# The 5 shared channels plus the one Open-Meteo calls something else.
BASE_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum",
]
DEW_VAR = "dew_point_2m_mean"
DAILY_VARS = BASE_VARS + [DEW_VAR]

CACHE_DIR = weather.CACHE_DIR.parent / "openmeteo_modern"
SCHEMA_VERSION = "m1"


def _cache_path(lat: float, lon: float, start: date, end: date, kind: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}_{SCHEMA_VERSION}_{lat:.3f}_{lon:.3f}_{start.isoformat()}_{end.isoformat()}.json"


def _to_df(payload: dict) -> pd.DataFrame:
    daily = (payload or {}).get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    out = {"date": pd.to_datetime(times).date}
    for v in DAILY_VARS:
        col = daily.get(v)
        out[v] = col if col else [None] * len(times)
    df = pd.DataFrame(out)
    for v in DAILY_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def _cached_or_fetch(url: str, cache: Path, max_age_hours: int) -> pd.DataFrame:
    """Read the cache when fresh; otherwise fetch and write it.

    Honours `weather.NO_FETCH` (the Pages build sets RW2_NO_FETCH=1), in which
    case a stale cache is still used and a missing one yields an empty frame --
    the caller then refuses rather than serving a hole.
    """
    if cache.exists():
        age_h = (pd.Timestamp.utcnow().timestamp() - cache.stat().st_mtime) / 3600.0
        if weather.NO_FETCH or age_h <= max_age_hours:
            try:
                return _to_df(json.loads(cache.read_text()))
            except Exception:
                pass
    if weather.NO_FETCH:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    try:
        payload = weather._http_json(url)
    except Exception:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    try:
        cache.write_text(json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass
    return _to_df(payload)


def fetch_history(lat: float, lon: float, start: date, end: date,
                  *, max_age_hours: int = 24) -> pd.DataFrame:
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS), "timezone": "UTC",
    }
    url = weather.ARCHIVE_URL + "?" + urlencode(params)
    return _cached_or_fetch(url, _cache_path(lat, lon, start, end, "hist"), max_age_hours)


def fetch_forecast(lat: float, lon: float, days: int = 14,
                   *, max_age_hours: int = 3, past_days: int = 0) -> pd.DataFrame:
    """Forecast rows, optionally with `past_days` of recent analysis prepended.

    WHY `past_days`: `mblstm_modern` builds its decoder window from the GAUGE's
    last observation (`last_q_date + 1 ... + horizon`), not from today. The
    forecast endpoint starts at today, so on a gauge that is 2+ days stale the
    earliest decoder rows fall in the PAST, reindex to NaN, and `norm_wx` turns
    them into the training mean -- silently, and at the only lead with a
    published skill number. Asking for a few past days closes that hole.
    """
    today = date.today()
    end = today + timedelta(days=days)
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_VARS), "forecast_days": str(min(max(days, 1), 16)),
        "timezone": "UTC",
    }
    if past_days > 0:
        params["past_days"] = str(min(past_days, 92))
    url = weather.FORECAST_URL + "?" + urlencode(params)
    start_key = today - timedelta(days=past_days)
    return _cached_or_fetch(url, _cache_path(lat, lon, start_key, end, "fcst"), max_age_hours)


# --------------------------------------------------------------------------
# The pre-built forcing table.
#
# One interface, two backends. The Pages build cannot fetch per-station at
# serve time: it sets RW2_NO_FETCH=1, and this module's cache key embeds the
# literal (start, end) pair, so `end = today-1` mints a new filename every day
# and the cache can never warm. Instead a single pre-matrix CI job maintains a
# per-basin table incrementally and hands all 64 shards one artifact, the way
# `snodas-extract` already does.
#
# The table is ALSO the drop-in point for NLDAS-2 areal forcing, which is the
# measured fix for Gate 1's -0.059534. Same path, same schema, same reader --
# only the producer changes, and the `src` column records which rows came from
# where so the public label can say so honestly.
# --------------------------------------------------------------------------

TABLE_DIR = Path(__file__).resolve().parents[1] / "data" / "m1_forcing"

#: What the member actually consumes, in the units the checkpoint was fitted on
#: (degC, mm, MJ m-2 d-1, Pa). `vapor_pressure` is stored already derived, so
#: `mblstm_modern._ensure_vapor_pressure()` no-ops on it.
TABLE_COLS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure",
]


def table_path(station_id: str) -> Path:
    return TABLE_DIR / f"{station_id}.csv.gz"


#: The served member's public label. TWO copies can exist and they can disagree:
#:   - `data/m1_forcing/_served_skill.json` travels with the forcing artifact,
#:     rebuilt by the CI job from the forcing it just published.
#:   - `data/m1_served_skill.json` is committed, and is only as fresh as the
#:     last commit.
#: The artifact copy wins. This is a single function because it was briefly
#: duplicated with different rules in build_static_site.py and
#: build_benchmarks_page.py, and the live site then showed a panel quoting one
#: gate beside a page quoting the other.
LABEL_NAME = "m1_served_skill.json"


def label_path() -> Path:
    side = TABLE_DIR / "_served_skill.json"
    return side if side.exists() else (TABLE_DIR.parent / LABEL_NAME)


def load_table(station_id: str) -> pd.DataFrame | None:
    """The basin's forcing table, or None when it has not been published.

    Returns ONE continuous frame spanning history and forecast. `mblstm_modern`
    reindexes its encoder and decoder windows out of whatever it is handed, so
    the same frame serves as both `wx_hist` and `wx_fcst` -- which is precisely
    what removes the stale-gauge decoder hole described in `fetch_forecast`.

    Absent is fine and must stay fine: the member then refuses with a reason
    rather than serving a hole.
    """
    p = table_path(station_id)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, compression="gzip")
    except Exception:
        return None
    if "date" not in df.columns or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    for c in TABLE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)
