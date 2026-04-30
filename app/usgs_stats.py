"""USGS NWIS daily-statistics fetcher.

Returns the per-day-of-year climatology table that powers the legacy
"Daily discharge statistics" chart on USGS waterdata pages: per-day-of-year
min, 25th, 50th (median), mean, 75th, max — plus the water years of the
record min/max. Values come from the `/nwis/stat/` endpoint with
statReportType=daily.

Cached per site for 30 days because the table only changes when USGS finalizes
a new water year of data.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

NO_FETCH = os.environ.get("RW2_NO_FETCH") == "1"

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "usgs_stats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAT_URL = "https://waterservices.usgs.gov/nwis/stat/"
STAT_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _cache_path(site_no: str) -> Path:
    return CACHE_DIR / f"{site_no}.json"


def fetch_daily_stats(site_no: str, *, max_age_seconds: int = STAT_TTL_SECONDS) -> Optional[dict]:
    """Daily statistics for parameter 00060 (discharge). Returns None on failure.

    Output schema:
      {
        "site_no": "...",
        "begin_date": "1952-04-29",  # earliest date in any year
        "end_date":   "2024-09-30",  # latest finalized water year
        "rows": [
            {"month_day": "01-01", "min_va": .., "p25_va": .., "p50_va": ..,
             "mean_va": .., "p75_va": .., "max_va": ..,
             "min_yr": 1973, "max_yr": 2017, "count": 73},
            ...
        ]
      }
    """
    cache = _cache_path(site_no)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_seconds:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    if NO_FETCH:
        # Return cached value if any, even if older than TTL; otherwise give up.
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                return None
        return None

    params = {
        "format": "rdb",
        "sites": site_no,
        "statReportType": "daily",
        "statTypeCd": "min,p25,p50,mean,p75,max",
        "parameterCd": "00060",
    }
    url = STAT_URL + "?" + urlencode(params)
    try:
        req = Request(url, headers={"User-Agent": "riverwatch2/0.1"})
        with urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        # Fall back to existing stale cache if available, else None
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        return None

    parsed = _parse_rdb(text, site_no)
    if parsed is None:
        return None
    cache.write_text(json.dumps(parsed, separators=(",", ":")))
    return parsed


def _parse_rdb(text: str, site_no: str) -> Optional[dict]:
    """Parse USGS RDB tab-delimited statistics output."""
    header: Optional[list[str]] = None
    rows: list[dict] = []
    begin_date: Optional[str] = None
    end_date: Optional[str] = None
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        if header is None:
            header = raw.split("\t")
            continue
        if raw.startswith("5s") or raw.split("\t")[0] in {"5s", "15s"}:
            # Format/skip row that follows the header
            continue
        parts = raw.split("\t")
        if len(parts) < len(header):
            continue
        rec = dict(zip(header, parts))
        try:
            mm = int(rec.get("month_nu", 0))
            dd = int(rec.get("day_nu", 0))
        except ValueError:
            continue
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            continue

        def _num(key: str) -> Optional[float]:
            v = rec.get(key, "").strip()
            if not v or v in {"--", "Eqp", "Mnt", "Ssn", "Ice"}:
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _yr(key: str) -> Optional[int]:
            v = rec.get(key, "").strip()
            if not v:
                return None
            try:
                return int(v[:4])
            except ValueError:
                return None

        rows.append({
            "month_day": f"{mm:02d}-{dd:02d}",
            "min_va": _num("min_va"),
            "p25_va": _num("p25_va"),
            "p50_va": _num("p50_va"),
            "mean_va": _num("mean_va"),
            "p75_va": _num("p75_va"),
            "max_va": _num("max_va"),
            "min_yr": _yr("min_va_yr"),
            "max_yr": _yr("max_va_yr"),
            "count": int(rec.get("count_nu", "0") or 0) if rec.get("count_nu", "").strip().isdigit() else None,
        })
        # USGS doesn't expose begin/end via this endpoint directly — derive
        # from min_yr/max_yr seen across rows.
    if not rows:
        return None

    yrs = []
    for r in rows:
        for k in ("min_yr", "max_yr"):
            if r.get(k) is not None:
                yrs.append(r[k])
    if yrs:
        begin_date = f"{min(yrs)}-01-01"
        end_date = f"{max(yrs)}-12-31"

    return {
        "site_no": site_no,
        "begin_date": begin_date,
        "end_date": end_date,
        "rows": rows,
    }


def stats_for_today(stats: dict, today: Optional[date] = None) -> Optional[dict]:
    """Return the single daily-stat row matching today's MM-DD."""
    if not stats or not stats.get("rows"):
        return None
    t = today or date.today()
    md = f"{t.month:02d}-{t.day:02d}"
    for r in stats["rows"]:
        if r.get("month_day") == md:
            return r
    return None
