#!/usr/bin/env python3
"""Bulk-build the MB-LSTM training corpus from Daymet (ORNL single-pixel API).

Daymet v4: daily, 1 km, North America, 1980 → last complete calendar year.
One HTTP request returns the full multi-decade record for a point (~3 s), so
this fills the corpus ~1000x faster than Open-Meteo's rate-limited archive.

Columns are mapped into the corpus schema (weather.DAILY_VARS); variables
Daymet doesn't carry (wind, ET0, soil, snow depth) are left NaN — the pilot
trains on the 5 shared "compat" variables (see train_mblstm.py --compat-vars).

Notes:
  - Daymet uses a 365-day calendar: Dec 31 is missing in leap years (the
    trainer's daily reindex leaves that one row NaN — harmless).
  - srad is daylight-average W/m²; shortwave_radiation_sum (MJ/m²/day) is
    srad * dayl / 1e6, matching Open-Meteo's definition at serve time.
  - Existing corpus files (e.g. from the Open-Meteo fetcher) are skipped.

Usage:
  .venv/bin/python scripts/build_mblstm_data_daymet.py --limit 800
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import usgs  # noqa: E402
from app.weather import DAILY_VARS  # noqa: E402

STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"
SOURCES_PATH = ROOT / "data" / "mblstm" / "corpus_sources.json"

DAYMET_URL = "https://daymet.ornl.gov/single-pixel/api/data"
MIN_OVERLAP_DAYS = 8 * 365


def fetch_daymet(lat: float, lon: float, start: date, end: date) -> pd.DataFrame | None:
    params = {
        "lat": f"{lat:.4f}", "lon": f"{lon:.4f}",
        "vars": "prcp,tmax,tmin,srad,dayl",
        "start": start.isoformat(), "end": end.isoformat(),
    }
    req = Request(DAYMET_URL + "?" + urlencode(params),
                  headers={"User-Agent": "riverwatch2/0.1"})
    with urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    # Header block ends at the "year,yday,..." line.
    pos = raw.find("year,yday")
    if pos < 0:
        return None
    df = pd.read_csv(io.StringIO(raw[pos:]))
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    if not {"year", "yday", "prcp", "tmax", "tmin", "srad", "dayl"} <= set(df.columns):
        return None
    dates = pd.to_datetime(df["year"].astype(str)) + pd.to_timedelta(df["yday"] - 1, unit="D")
    out = pd.DataFrame({"date": dates.dt.date})
    out["temperature_2m_max"] = df["tmax"]
    out["temperature_2m_min"] = df["tmin"]
    out["temperature_2m_mean"] = (df["tmax"] + df["tmin"]) / 2.0
    out["precipitation_sum"] = df["prcp"]
    out["shortwave_radiation_sum"] = df["srad"] * df["dayl"] / 1e6
    for c in DAILY_VARS:
        if c not in out.columns:
            out[c] = np.nan
    return out[["date"] + DAILY_VARS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    stations = json.loads(STATIONS_PATH.read_text())["stations"]
    if args.offset:
        stations = stations[args.offset:]
    if args.limit:
        stations = stations[: args.limit]

    sources = {}
    if SOURCES_PATH.exists():
        sources = json.loads(SOURCES_PATH.read_text())

    done = skipped = failed = 0
    t0 = time.time()
    for i, st in enumerate(stations, 1):
        out = CORPUS_DIR / f"{st['id']}.csv.gz"
        if out.exists():
            skipped += 1
            continue
        try:
            q = usgs.fetch_daily_discharge(st["id"], start, end)
            if q.empty or len(q) < MIN_OVERLAP_DAYS:
                failed += 1
                print(f"[{i}/{len(stations)}] {st['id']} SKIP q too short ({len(q)})", flush=True)
                continue
            wx = None
            for attempt in range(2):
                try:
                    wx = fetch_daymet(st["lat"], st["lon"], start, end)
                    break
                except Exception:
                    time.sleep(10)
            if wx is None or len(wx) < MIN_OVERLAP_DAYS:
                failed += 1
                print(f"[{i}/{len(stations)}] {st['id']} SKIP daymet unavailable", flush=True)
                continue
            df = pd.merge(q, wx, on="date", how="inner").sort_values("date").reset_index(drop=True)
            if len(df) < MIN_OVERLAP_DAYS:
                failed += 1
                print(f"[{i}/{len(stations)}] {st['id']} SKIP overlap too short ({len(df)})", flush=True)
                continue
            df.to_csv(out, index=False, compression="gzip")
            sources[st["id"]] = "daymet"
            done += 1
            print(f"[{i}/{len(stations)}] {st['id']} ok rows={len(df)} "
                  f"({df['date'].iloc[0]}..{df['date'].iloc[-1]})", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(stations)}] {st['id']} SKIP error: {exc}", flush=True)
        if done % 25 == 0:
            SOURCES_PATH.write_text(json.dumps(sources, indent=0))
        time.sleep(args.sleep)

    SOURCES_PATH.write_text(json.dumps(sources, indent=0))
    print(f"\ndone={done} skipped(existing)={skipped} failed={failed} in "
          f"{time.time() - t0:.0f}s; corpus now has "
          f"{len(list(CORPUS_DIR.glob('*.csv.gz')))} stations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
