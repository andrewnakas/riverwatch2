"""v14.5c: NOHRSC SNODAS daily 1km SWE + snowmelt extracts.

SNODAS is NOAA/NOHRSC's gridded snow data assimilation product — daily
30-arcsec (~1km) CONUS coverage of SWE, snow depth, snowmelt at base of
pack, and several sublimation/precip channels. Crucially, unlike NBM the
SNODAS archive goes back to 2003-10-01 and is preserved at NSIDC, so
*backtest* MAE numbers can actually move (the holdouts can score against
real SWE/melt inputs at the past dates they evaluate).

This module is a *lazy loader* over a pre-built per-station extracts file
at `data/snodas_extracts/{station_id}.json`. The extract script
(`scripts/build_snodas_extracts.py`) downloads SNODAS tars from NSIDC,
samples SWE (band 1034) and snowmelt-runoff (band 1044) at every active
station's basin centroid, and writes the result. CI caches the extracts
dir so subsequent runs only fetch the daily delta.

Public API mirrors `app/snotel.py` so `_build_features` consumes the
output without changes:

  fetch_swe_history(station_id, start, end) -> DataFrame[date, swe_in, melt_24h_mm]

When no extract exists for a station (off-CONUS, or a build that hasn't
run the extract job yet), returns an empty DataFrame — same fall-through
contract as SNOTEL when there's no nearby site.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

SNODAS_OFF = os.environ.get("RW2_SNODAS_OFF") == "1"

ROOT = Path(__file__).resolve().parents[1]
EXTRACTS_DIR = ROOT / "data" / "snodas_extracts"
EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_path(station_id: str) -> Path:
    return EXTRACTS_DIR / f"{station_id}.json"


def _load_extract(station_id: str) -> Optional[dict]:
    """Return the per-station SNODAS dict {iso_date: {swe_in, melt_24h_mm}} or None."""
    p = _extract_path(station_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fetch_swe_history(station_id: str, start: date, end: date) -> pd.DataFrame:
    """Return SNODAS daily SWE / melt history for one station.

    Output schema matches the SNOTEL fetcher so it slots into the same
    `snotel_df` argument of `_build_features`:
        ['date', 'swe_in', 'melt_24h_mm']

    `swe_in` is column-integrated SWE in inches (SNODAS native is mm,
    converted here so the same `_build_features` swe-window logic fires).
    `melt_24h_mm` is snowmelt at the base of the pack (NOHRSC band 1044),
    24-h total ending 06Z on that date.

    Empty DataFrame when:
      - extract file doesn't exist (off-CONUS, or extract job hasn't run)
      - SNODAS_OFF kill switch is set
    """
    if SNODAS_OFF:
        return pd.DataFrame(columns=["date", "swe_in", "melt_24h_mm"])
    rec = _load_extract(station_id)
    if not rec:
        return pd.DataFrame(columns=["date", "swe_in", "melt_24h_mm"])

    rows: list[dict] = []
    s_iso, e_iso = start.isoformat(), end.isoformat()
    for d_iso, vals in rec.items():
        if not (s_iso <= d_iso <= e_iso):
            continue
        if not isinstance(vals, dict):
            continue
        rows.append({
            "date": d_iso,
            "swe_in": vals.get("swe_in"),
            "melt_24h_mm": vals.get("melt_24h_mm"),
        })
    if not rows:
        return pd.DataFrame(columns=["date", "swe_in", "melt_24h_mm"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    for c in ("swe_in", "melt_24h_mm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def coverage_summary() -> dict:
    """Quick observability hook. How many stations have SNODAS extracts?"""
    files = list(EXTRACTS_DIR.glob("*.json"))
    return {"stations_with_snodas": len(files), "extracts_dir": str(EXTRACTS_DIR)}
