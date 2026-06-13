#!/usr/bin/env python3
"""Extract archived HRRR forecast forcings at gauge points from dynamical.org.

HRRR is the 3 km CONUS model — far sharper precip placement than 0.25-degree
GFS, but it only forecasts 48 h ahead. So this extracts lead days 1-2 only;
they overlay the matching GFS lead days to form a hybrid decoder forcing
(HRRR d1-2 + GFS d3-14). Same alignment convention as the GFS fetcher: a 00z
init on day D has lead_day 1 = calendar day D, issue date t0 = D-1.

For each 00z init date, writes data/mblstm/hrrr_fcst/<YYYY-MM-DD>.csv.gz with:

    station_id, lead_day, temperature_2m_mean, temperature_2m_max,
    temperature_2m_min, precipitation_sum, shortwave_radiation_sum

HRRR's grid is projected (y, x with 2D lat/lon coords), so stations map to
grid cells through a one-time nearest-neighbour KDTree, then pointwise
vectorized indexing.

Archive coverage: 2018-07-13 -> present. Attribution: NOAA HRRR via
dynamical.org (CC BY 4.0).

Usage:
  .venv/bin/python scripts/fetch_hrrr_forcings.py --start 2021-05-03 --end 2025-12-29 --stride-days 7
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
OUT_DIR = ROOT / "data" / "mblstm" / "hrrr_fcst"
ZARR_URL = "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr"
VARS = ["precipitation_surface", "temperature_2m",
        "downward_short_wave_radiation_flux_surface"]
HORIZON = 2  # HRRR only reaches 48 h


def grid_index(ds, lats: np.ndarray, lons: np.ndarray):
    """Nearest (y, x) grid indices for each station via KDTree on the 2D
    lat/lon coords (Lambert grid — no rectilinear sel possible)."""
    from scipy.spatial import cKDTree

    glat = ds.latitude.values
    glon = ds.longitude.values
    coslat = np.cos(np.deg2rad(np.clip(lats.mean(), 20, 55)))
    pts = np.column_stack([glat.ravel(), glon.ravel() * coslat])
    tree = cKDTree(pts)
    dist, flat = tree.query(np.column_stack([lats, lons * coslat]))
    yi, xi = np.unravel_index(flat, glat.shape)
    # Stations outside CONUS (HRRR domain) land on far-away edge cells; mask
    # anything beyond ~0.1 degree (~10 km) from its nearest cell.
    ok = dist <= 0.1
    return yi, xi, ok


def extract_init(ds, yi, xi, sids, init: date) -> pd.DataFrame | None:
    t_init = np.datetime64(f"{init.isoformat()}T00:00:00")
    if t_init not in ds.init_time.values:
        return None
    sub = ds[VARS].sel(init_time=t_init)
    sub = sub.sel(lead_time=slice(np.timedelta64(1, "h"), np.timedelta64(HORIZON * 24, "h")))
    pt = sub.isel(
        y=xr.DataArray(yi, dims="station"),
        x=xr.DataArray(xi, dims="station"),
    ).compute()

    lead_h = (pt.lead_time.values / np.timedelta64(1, "h")).astype(float)
    lead_day = np.ceil(lead_h / 24.0).astype(int)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-05-03")
    ap.add_argument("--end", default="2025-12-29")
    ap.add_argument("--stride-days", type=int, default=7)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sts = json.loads(STATIONS_PATH.read_text())["stations"]
    lats = np.asarray([s["lat"] for s in sts])
    lons = np.asarray([s["lon"] for s in sts])

    inits = []
    d = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    while d <= end:
        inits.append(d)
        d += timedelta(days=args.stride_days)

    ds = xr.open_zarr(ZARR_URL, decode_timedelta=True)
    yi, xi, ok = grid_index(ds, lats, lons)
    sids = [s["id"] for s, k in zip(sts, ok) if k]
    yi, xi = yi[ok], xi[ok]
    print(f"{len(inits)} init dates {inits[0]}..{inits[-1]}, "
          f"{len(sids)} stations in HRRR domain ({(~ok).sum()} outside)", flush=True)

    done = failed = 0
    t0 = time.time()
    for i, init in enumerate(inits, 1):
        out = OUT_DIR / f"{init.isoformat()}.csv.gz"
        if out.exists():
            continue
        try:
            df = extract_init(ds, yi, xi, sids, init)
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
