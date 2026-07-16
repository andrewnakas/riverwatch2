#!/usr/bin/env python3
"""Convert a neuralhydrology test_results.p (continuous daily sim) → the
combine_dumps.py stride-14 window grid, so NH LSTM members join the δHBV/v2r
members in the grand ensemble.

NH `test_results.p` schema (verified):
  {basin_id: {'1D': {'xr': Dataset(q_mm_obs, q_mm_sim; coords date,time_step),
                     'NSE': float}}}
The sim is a continuous seq-to-one daily simulation over the test decade
(1989-10-01 → 1999-09-30). It is in SPECIFIC DISCHARGE (mm/day) — we convert
back to cfs (× area_km2 / 2.446576) so it is commensurable with the δHBV dumps
(which are in raw cfs).

Output schema (matches data/mblstm/gpu_dumps_s14/*.csv.gz exactly):
  station_id (int, no leading zeros), t0 (window start), h (lead 1..H),
  truth, ylo, ymed, yhi, ymean, persist   (all flows in cfs)
For a point model ylo==ymed==yhi==ymean=sim; persist = truth at t0-1 (naive).

Windowing: stride-14 t0 grid over the test period, leads h=1..14. ymed at
(t0, h) = the daily sim on date t0+(h-1). This is the SAME construction the
δHBV dumps use (daily sim reshaped into overlapping-free stride-14 windows), so
the two are directly combinable. Restrict to a station subset (the ss3 177) with
--stations-like <existing_dump.csv.gz> so t0/station grids align for combine.

Usage (on the box, after nh-run evaluate):
  python scripts/nh_to_dump.py \
    --results data/nh/runs/rw2_daymet_lstm_mm_XXXX/test/model_epoch030/test_results.p \
    --forcing daymet \
    --stations-like data/mblstm/gpu_dumps_s14/camels531_daymet_combined_ens_s14.csv.gz \
    --out data/mblstm/gpu_dumps_s14/camels531_daymet_nhlstm_s14.csv.gz
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFS_TO_MMDAY_PER_KM2 = 0.0283168 * 86400 / 1e6 * 1000   # = 2.446576
STRIDE = 14
HMAX = 14


def load_areas() -> dict[str, float]:
    attrs = json.loads((ROOT / "data" / "camels_attrs.json").read_text())
    return {k: v.get("area_gages2") for k, v in attrs.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="NH test_results.p")
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--out", required=True, help="output csv.gz (dump grid)")
    ap.add_argument("--stations-like", default="",
                    help="existing dump csv.gz whose (station_id,t0) grid to match")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--hmax", type=int, default=HMAX)
    args = ap.parse_args()

    areas = load_areas()
    with open(args.results, "rb") as fh:
        res = pickle.load(fh)

    # Optional grid template: restrict to the same stations + t0 windows as an
    # existing δHBV dump so combine_dumps.py joins them cleanly.
    keep_stations: set[int] | None = None
    template_t0: list[pd.Timestamp] | None = None
    keep_stations_s: set[str] | None = None
    if args.stations_like:
        tdf = pd.read_csv(args.stations_like, compression="gzip",
                          usecols=["station_id", "t0"], dtype={"station_id": str})
        keep_stations_s = set(tdf["station_id"].astype(str).unique().tolist())
        template_t0 = sorted(pd.to_datetime(tdf["t0"].unique()))

    rows = []
    n_used = 0
    for bid, per in res.items():
        # combine_dumps.py reads station_id as STR — the δHBV/v2r dumps store it
        # ZERO-PADDED (e.g. "01022500"). Emit the same padded string so the
        # (station_id,t0,h) inner-join matches (an int "1022500" silently drops
        # every leading-zero basin → only ~34 western basins survive).
        sid = str(bid).zfill(8)
        if keep_stations_s is not None and sid not in keep_stations_s:
            continue
        area = areas.get(str(bid).zfill(8)) or areas.get(bid)
        if not (area and np.isfinite(area) and area > 0):
            continue
        xr = per["1D"]["xr"]
        # squeeze the singleton time_step dim → daily series indexed by date
        obs = xr["q_mm_obs"].isel(time_step=0).to_series()
        sim = xr["q_mm_sim"].isel(time_step=0).to_series()
        obs.index = pd.to_datetime(obs.index)
        sim.index = pd.to_datetime(sim.index)
        # mm/day → cfs
        to_cfs = area / CFS_TO_MMDAY_PER_KM2
        obs_cfs = obs * to_cfs
        sim_cfs = sim * to_cfs

        dates = obs.index
        if template_t0 is not None:
            t0_list = [t for t in template_t0 if t in set(dates)]
        else:
            # build stride t0 grid: first date, then every `stride` days, leaving
            # room for hmax leads
            all_days = pd.date_range(dates[0], dates[-1], freq="D")
            t0_list = list(all_days[:: args.stride])

        date_set = {d: i for i, d in enumerate(dates)}
        for t0 in t0_list:
            # persist baseline = observed flow the day before the window
            prev = t0 - pd.Timedelta(days=1)
            persist = float(obs_cfs.get(prev, np.nan))
            for h in range(1, args.hmax + 1):
                d = t0 + pd.Timedelta(days=h - 1)
                if d not in date_set:
                    continue
                truth = obs_cfs.get(d, np.nan)
                pred = sim_cfs.get(d, np.nan)
                if not np.isfinite(pred):
                    continue
                rows.append((sid, t0.strftime("%Y-%m-%d"), h,
                             float(truth) if np.isfinite(truth) else np.nan,
                             float(pred), float(pred), float(pred), float(pred),
                             persist))
        n_used += 1

    out = pd.DataFrame(rows, columns=["station_id", "t0", "h", "truth",
                                      "ylo", "ymed", "yhi", "ymean", "persist"])
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, compression="gzip")
    print(f"DONE {args.forcing}: {len(out)} rows, {n_used} basins, "
          f"{out['station_id'].nunique()} stations -> {outp}")
    print(f"  leads {sorted(out['h'].unique())[:3]}..{out['h'].max()}, "
          f"{out['t0'].nunique()} t0 windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
