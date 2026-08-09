#!/usr/bin/env python3
"""Aggregate hourly AORC -> the recipe-v2 daily corpus schema.

Why AORC is worth a member even though CONUS404 is already coming:

  1. It is HOURLY. Every other corpus we have is daily, so tmax/tmin are
     whatever the source product supplied. Here we compute them from the actual
     hourly temperature curve. CONUS404's DAILY store has no native tmax/tmin at
     all -- I had to write T2 into all three temperature columns. AORC fixes
     that properly.
  2. Ghimire et al. 2025 (WRR) found AORC improves peak-flow prediction
     "particularly at the small basin scales" -- which is exactly the pivotal
     basin profile (median 252 km2, slope 27.4, forested) that controls our
     median NSE.
  3. It is an independent analysis, so its errors should decorrelate from the
     gauge-interpolated products (Daymet/Maurer/NLDAS).

Unit conversions (AORC -> corpus):
    APCP_surface        kg/m2 per hour  -> SUM over day = mm/day
    TMP_2maboveground   K               -> mean/max/min in C
    DSWRF_surface       W/m2 hourly     -> mean W/m2 * 0.0864 = MJ/m2/day
    SPFH + PRES         specific humid. -> vapour pressure Pa
                        e = q*p/(0.622 + 0.378*q)
Discharge (q_cfs) is copied from the existing daymet corpus -- it is a gauge
observation and identical across forcing products.
"""
import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
        "vapor_pressure"]


def daily_from_hourly(df):
    t = pd.to_datetime(df["time"])
    d = pd.DataFrame({"date": t.dt.strftime("%Y-%m-%d")})
    tc = df["TMP_2maboveground"].astype(float) - 273.15
    q = df["SPFH_2maboveground"].astype(float)
    p = df["PRES_surface"].astype(float)
    d["tc"] = tc
    d["prcp"] = df["APCP_surface"].astype(float)
    d["srad"] = df["DSWRF_surface"].astype(float)
    d["vp"] = q * p / (0.622 + 0.378 * q)
    g = d.groupby("date", sort=True)
    out = pd.DataFrame({
        "date": list(g.groups.keys()),
        "temperature_2m_mean": g["tc"].mean().round(3).values,
        "temperature_2m_max": g["tc"].max().round(3).values,   # TRUE daily max
        "temperature_2m_min": g["tc"].min().round(3).values,   # TRUE daily min
        "precipitation_sum": g["prcp"].sum().round(4).values,  # hourly accum
        # mean W/m2 -> MJ/m2/day  (86400 s / 1e6)
        "shortwave_radiation_sum": (g["srad"].mean() * 0.0864).round(6).values,
        "vapor_pressure": g["vp"].mean().round(2).values,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/aorc_raw")
    ap.add_argument("--out", default="data/local_corpora/camels_corpus_aorc_v2")
    ap.add_argument("--qsrc", default="data/local_corpora/camels_corpus_daymet_v2")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    raw, out, qsrc = Path(args.raw), Path(args.out), Path(args.qsrc)
    out.mkdir(parents=True, exist_ok=True)
    tars = sorted(raw.glob("water_year_*.tar.gz"))
    if not tars:
        sys.exit(f"no archives in {raw}")
    print(f"{len(tars)} water-year archives")

    # accumulate per basin across water years
    acc = {}
    for i, tp in enumerate(tars, 1):
        with tarfile.open(tp, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".csv"):
                    continue
                bid = Path(m.name).name.split("_")[0].zfill(8)
                if args.limit and bid not in acc and len(acc) >= args.limit:
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                df = pd.read_csv(io.BytesIO(fh.read()))
                acc.setdefault(bid, []).append(daily_from_hourly(df))
        print(f"  [{i}/{len(tars)}] {tp.name} -> {len(acc)} basins", flush=True)

    written = 0
    for bid, parts in acc.items():
        d = pd.concat(parts, ignore_index=True).drop_duplicates("date")
        d = d.sort_values("date")
        fq = qsrc / f"{bid}.csv.gz"
        if fq.exists():
            q = pd.read_csv(fq, usecols=["date", "q_cfs"])
            d = d.merge(q, on="date", how="left")
        else:
            d["q_cfs"] = np.nan
        d[COLS].to_csv(out / f"{bid}.csv.gz", index=False, compression="gzip")
        written += 1
    print(f"\nwritten={written} -> {out}")
    s = pd.read_csv(out / f"{sorted(acc)[0]}.csv.gz")
    print(f"cols={list(s.columns)}")
    print(f"dates {s.date.min()}..{s.date.max()} rows={len(s)}")
    print(f"prcp {s.precipitation_sum.mean():.2f} mm/d | "
          f"tmean {s.temperature_2m_mean.mean():.1f}C | "
          f"tmax-tmin spread {(s.temperature_2m_max-s.temperature_2m_min).mean():.1f}C | "
          f"q finite {s.q_cfs.notna().mean():.3f}")


if __name__ == "__main__":
    main()
