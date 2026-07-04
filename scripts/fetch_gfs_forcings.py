#!/usr/bin/env python3
"""Extract archived GFS forecast forcings at gauge points from dynamical.org.

For each 00z init date, pulls precipitation / 2m temperature / shortwave
radiation for leads 1..336h at every registered station, aggregates to the
5 compat daily variables for lead-days 1..14, and writes one csv.gz per init
to data/mblstm/gfs_fcst/<YYYY-MM-DD>.csv.gz with columns:

    station_id, lead_day, temperature_2m_mean, temperature_2m_max,
    temperature_2m_min, precipitation_sum, shortwave_radiation_sum

These are *real* forecasts (with real forecast error), used to fine-tune the
MB-LSTM decoder so backtests stop being perfect-forcing-flattered — and the
same product can drive the decoder at serve time (latest init).

Attribution: NOAA GFS from dynamical.org (CC BY 4.0).

Usage:
  .venv/bin/python scripts/fetch_gfs_forcings.py --start 2021-05-03 --end 2025-12-29 --stride-days 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
OUT_DIR = ROOT / "data" / "mblstm" / "gfs_fcst"
ZARR_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"
VARS = ["precipitation_surface", "temperature_2m",
        "downward_short_wave_radiation_flux_surface"]
HORIZON = 14


def extract_init(ds, lats, lons, sids, init: date) -> pd.DataFrame | None:
    t_init = np.datetime64(f"{init.isoformat()}T00:00:00")
    if t_init not in ds.init_time.values:
        return None
    sub = ds[VARS].sel(init_time=t_init)
    sub = sub.sel(lead_time=slice(np.timedelta64(1, "h"), np.timedelta64(HORIZON * 24, "h")))
    pt = sub.sel(latitude=lats, longitude=lons, method="nearest").compute()

    lead_h = (pt.lead_time.values / np.timedelta64(1, "h")).astype(float)
    lead_day = np.ceil(lead_h / 24.0).astype(int)  # (24(d-1), 24d] -> day d
    rows = []
    for d in range(1, HORIZON + 1):
        m = lead_day == d
        if m.sum() < 4:
            continue
        temp = pt.temperature_2m.values[m]          # (steps, station), degC
        prcp = pt.precipitation_surface.values[m]   # kg m-2 s-1
        srad = pt.downward_short_wave_radiation_flux_surface.values[m]  # W m-2
        rows.append(pd.DataFrame({
            "station_id": sids,
            "lead_day": d,
            "temperature_2m_mean": np.nanmean(temp, axis=0),
            "temperature_2m_max": np.nanmax(temp, axis=0),
            "temperature_2m_min": np.nanmin(temp, axis=0),
            "precipitation_sum": np.nanmean(prcp, axis=0) * 86400.0,   # mm/day
            "shortwave_radiation_sum": np.nanmean(srad, axis=0) * 0.0864,  # MJ/m2/day
        }))
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-05-03")
    ap.add_argument("--end", default="2025-12-29")
    ap.add_argument("--stride-days", type=int, default=7)
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="output directory (default data/mblstm/gfs_fcst)")
    args = ap.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sts = json.loads(STATIONS_PATH.read_text())["stations"]
    sids = [s["id"] for s in sts]
    lats = xr.DataArray([s["lat"] for s in sts], dims="station")
    lons = xr.DataArray([s["lon"] for s in sts], dims="station")

    inits = []
    d = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    while d <= end:
        inits.append(d)
        d += timedelta(days=args.stride_days)

    print(f"{len(inits)} init dates {inits[0]}..{inits[-1]}, {len(sids)} stations", flush=True)
    ds = xr.open_zarr(ZARR_URL, decode_timedelta=True)

    done = failed = 0
    t0 = time.time()
    for i, init in enumerate(inits, 1):
        out = OUT_DIR / f"{init.isoformat()}.csv.gz"
        if out.exists():
            continue
        try:
            df = extract_init(ds, lats, lons, sids, init)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(inits)}] {init} ERR {exc}", flush=True)
            time.sleep(5)
            continue
        if df is None:
            failed += 1
            print(f"[{i}/{len(inits)}] {init} missing init", flush=True)
            continue
        df.to_csv(out, index=False, compression="gzip")
        done += 1
        if done % 10 == 0:
            rate = (time.time() - t0) / max(done, 1)
            print(f"[{i}/{len(inits)}] {init} ok ({rate:.0f}s/init, "
                  f"~{rate * (len(inits) - i) / 60:.0f} min left)", flush=True)

    print(f"done={done} failed={failed} in {(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
