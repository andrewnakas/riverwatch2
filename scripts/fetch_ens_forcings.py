#!/usr/bin/env python3
"""Extract archived ensemble forecast forcings (GEFS / ECMWF IFS ENS) at gauge
points from dynamical.org, in the same csv.gz schema as fetch_gfs_forcings.py.

For each 00z init date, pulls precipitation / 2m temperature / shortwave
radiation for leads 1..336h for every registered station, aggregates each
ensemble member to the 5 compat daily variables for lead-days 1..14, and
writes per init:

    <out>/<YYYY-MM-DD>.csv.gz            ensemble-MEAN of the per-member daily
                                         aggregates (drop-in for load_gfs())
    <out>/<YYYY-MM-DD>.members.csv.gz    per-member rows (extra `member` col)
                                         when --members N covers this init

Ensemble mean is taken across per-member daily aggregates (mean of per-member
tmax, not tmax of ensemble-mean trace), which is the meaning of "ens-mean
forcing" the decoder should see.

Units are identical across the dynamical.org GFS/GEFS/ECMWF stores (verified):
precip kg m-2 s-1 avg-rate, temp degC instant, srad W m-2 avg.

Attribution: NOAA GEFS / ECMWF IFS ENS via dynamical.org (CC BY 4.0).

Usage:
  .venv/bin/python scripts/fetch_ens_forcings.py --model ecmwf \
      --start 2024-04-01 --end 2025-12-29 --stride-days 7 \
      --members 10 --members-start 2025-01-01 --members-end 2025-12-31
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

# NOTE: no dask in this venv — xr.open_zarr degrades to eager zarr-native
# reads, which is fine here (per-init point extraction, bounded memory).

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
HORIZON = 14

MODELS = {
    "gefs": {
        "url": "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr",
        "out": ROOT / "data" / "mblstm" / "gefs_fcst",
    },
    "ecmwf": {
        "url": "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr",
        "out": ROOT / "data" / "mblstm" / "ecmwf_fcst",
    },
}
VARS = ["precipitation_surface", "temperature_2m",
        "downward_short_wave_radiation_flux_surface"]
OUT_VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum"]


def extract_init(ds, lats, lons, sids, init: date) -> pd.DataFrame | None:
    """Per-member daily aggregates for one init: DataFrame with columns
    member, station_id, lead_day, <5 compat vars>. None if init absent."""
    t_init = np.datetime64(f"{init.isoformat()}T00:00:00")
    if t_init not in ds.init_time.values:
        return None
    sub = ds[VARS].sel(init_time=t_init)
    sub = sub.sel(lead_time=slice(np.timedelta64(1, "h"), np.timedelta64(HORIZON * 24, "h")))
    # (lead, member, station) after point selection; order enforced below.
    pt = sub.sel(latitude=lats, longitude=lons, method="nearest").compute()
    pt = pt.transpose("lead_time", "ensemble_member", "station")

    lead_h = (pt.lead_time.values / np.timedelta64(1, "h")).astype(float)
    lead_day = np.ceil(lead_h / 24.0).astype(int)
    n_members = pt.sizes["ensemble_member"]
    members = np.arange(n_members)

    frames = []
    for d in range(1, HORIZON + 1):
        m = lead_day == d
        if m.sum() < 4:
            continue
        temp = pt.temperature_2m.values[m]          # (steps, member, station)
        prcp = pt.precipitation_surface.values[m]
        srad = pt.downward_short_wave_radiation_flux_surface.values[m]
        n_st = temp.shape[2]
        frames.append(pd.DataFrame({
            "member": np.repeat(members, n_st),
            "station_id": np.tile(sids, n_members),
            "lead_day": d,
            "temperature_2m_mean": np.nanmean(temp, axis=0).ravel(),
            "temperature_2m_max": np.nanmax(temp, axis=0).ravel(),
            "temperature_2m_min": np.nanmin(temp, axis=0).ravel(),
            "precipitation_sum": (np.nanmean(prcp, axis=0) * 86400.0).ravel(),
            "shortwave_radiation_sum": (np.nanmean(srad, axis=0) * 0.0864).ravel(),
        }))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--stride-days", type=int, default=7)
    ap.add_argument("--members", type=int, default=0,
                    help="also write per-member rows for the first N members "
                         "(inits inside --members-start/--members-end only)")
    ap.add_argument("--members-start", default="2025-01-01")
    ap.add_argument("--members-end", default="2025-12-31")
    args = ap.parse_args()

    spec = MODELS[args.model]
    out_dir = spec["out"]
    out_dir.mkdir(parents=True, exist_ok=True)

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
    mem_lo = date.fromisoformat(args.members_start)
    mem_hi = date.fromisoformat(args.members_end)

    print(f"{args.model}: {len(inits)} init dates {inits[0]}..{inits[-1]}, "
          f"{len(sids)} stations, members={args.members}", flush=True)
    ds = xr.open_zarr(spec["url"], decode_timedelta=True)

    done = failed = 0
    t0 = time.time()
    for i, init in enumerate(inits, 1):
        out = out_dir / f"{init.isoformat()}.csv.gz"
        want_members = args.members > 0 and mem_lo <= init <= mem_hi
        out_mem = out_dir / f"{init.isoformat()}.members.csv.gz"
        if out.exists() and (not want_members or out_mem.exists()):
            continue
        try:
            df = extract_init(ds, lats, lons, sids, init)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(inits)}] {init} ERR {type(exc).__name__}: {exc}", flush=True)
            time.sleep(10)
            continue
        if df is None:
            failed += 1
            print(f"[{i}/{len(inits)}] {init} missing init", flush=True)
            continue
        mean_df = (df.drop(columns=["member"])
                     .groupby(["station_id", "lead_day"], as_index=False)[OUT_VARS].mean())
        mean_df.to_csv(out, index=False, compression="gzip")
        if want_members:
            df[df["member"] < args.members].to_csv(out_mem, index=False, compression="gzip")
        done += 1
        if done % 5 == 0:
            rate = (time.time() - t0) / max(done, 1)
            print(f"[{i}/{len(inits)}] {init} ok ({rate:.0f}s/init, "
                  f"~{rate * (len(inits) - i) / 3600:.1f} h left)", flush=True)

    print(f"done={done} failed={failed} in {(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
