#!/usr/bin/env python3
"""Build an AORC corpus carrying SUB-DAILY intensity features.

Motivation. The shared-error analysis found 1% of days carry 93% of squared
error -- the failure is events, not drift. A daily model is handed a daily
precipitation SUM, which is identical for a catchment that received 40 mm in two
hours and one that received it over twenty-four. Those produce completely
different hydrographs. That information exists in the AORC hourly data we
already hold (2.5 GB, 30 water years, on disk) and we discarded it when
aggregating to daily.

This does NOT retrain at an hourly timestep. That would mean 24x sequence
length, 24x compute, new architecture decisions, and the target (USGS daily mean
discharge) would still be daily. Instead it keeps the daily model and adds
columns that summarise the within-day rainfall structure:

    p_max_1h    peak hourly rate            storm intensity
    p_max_3h    peak rolling 3-hour total   burst duration
    p_hours     hours with measurable rain  concentrated vs spread
    p_cv        std/mean of hourly rain     burstiness
    p_centroid  intensity-weighted hour     timing within the day

Same trainer, same architecture, same 531 basins -- only the input columns
change, so any score difference is attributable to the sub-daily signal alone.

Everything else in the schema is reproduced exactly as build_aorc_corpus.py
computes it, so the new corpus is a strict superset of the existing one and the
two are directly comparable.
"""
import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
        "vapor_pressure"]
INTENSITY = ["p_max_1h", "p_max_3h", "p_hours", "p_cv", "p_centroid"]
COLS = BASE + INTENSITY


def daily_from_hourly(df):
    """Daily aggregates plus within-day rainfall structure."""
    t = pd.to_datetime(df["time"])
    tc = df["TMP_2maboveground"].astype(float) - 273.15
    q = df["SPFH_2maboveground"].astype(float)
    p = df["PRES_surface"].astype(float)
    d = pd.DataFrame({
        "date": t.dt.strftime("%Y-%m-%d"),
        "hour": t.dt.hour,
        "tc": tc,
        "prcp": df["APCP_surface"].astype(float),
        "srad": df["DSWRF_surface"].astype(float),
        "vp": q * p / (0.622 + 0.378 * q),
    })
    # rolling 3-hour totals must be computed on the CONTINUOUS hourly series,
    # before grouping, or bursts spanning a day boundary are missed
    d["p3"] = d["prcp"].rolling(3, min_periods=1).sum()

    g = d.groupby("date", sort=True)
    out = pd.DataFrame({
        "date": list(g.groups.keys()),
        "temperature_2m_mean": g["tc"].mean().round(3).values,
        "temperature_2m_max": g["tc"].max().round(3).values,
        "temperature_2m_min": g["tc"].min().round(3).values,
        "precipitation_sum": g["prcp"].sum().round(4).values,
        "shortwave_radiation_sum": (g["srad"].mean() * 0.0864).round(6).values,
        "vapor_pressure": g["vp"].mean().round(2).values,
        # --- sub-daily structure, the whole point of this corpus ---
        "p_max_1h": g["prcp"].max().round(4).values,
        "p_max_3h": g["p3"].max().round(4).values,
        "p_hours": g["prcp"].apply(lambda s: int((s > 0.01).sum())).values,
    })

    # coefficient of variation of hourly rain: 0 on dry days by convention
    cv = g["prcp"].apply(lambda s: s.std() / s.mean() if s.mean() > 1e-6 else 0.0)
    out["p_cv"] = cv.round(4).values

    # intensity-weighted hour of day; -1 marks a dry day so the model can tell
    # "no rain" from "rain centred at midnight"
    def centroid(sub):
        w = sub["prcp"].to_numpy(float)
        if w.sum() <= 1e-6:
            return -1.0
        return float((w * sub["hour"].to_numpy(float)).sum() / w.sum())
    out["p_centroid"] = g.apply(centroid, include_groups=False).round(3).values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/aorc_raw")
    ap.add_argument("--out", default="data/local_corpora/camels_corpus_aorcx_v2")
    ap.add_argument("--qsrc", default="data/local_corpora/camels_corpus_daymet_v2")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    raw, out, qsrc = Path(a.raw), Path(a.out), Path(a.qsrc)
    out.mkdir(parents=True, exist_ok=True)
    tars = sorted(raw.glob("water_year_*.tar.gz"))
    if not tars:
        sys.exit(f"no archives in {raw}")
    print(f"{len(tars)} water-year archives")

    acc = {}
    for i, tp in enumerate(tars, 1):
        with tarfile.open(tp, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".csv"):
                    continue
                bid = Path(m.name).name.split("_")[0].zfill(8)
                if a.limit and bid not in acc and len(acc) >= a.limit:
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                acc.setdefault(bid, []).append(
                    daily_from_hourly(pd.read_csv(io.BytesIO(fh.read()))))
        print(f"  [{i}/{len(tars)}] {tp.name} -> {len(acc)} basins", flush=True)

    written = 0
    for bid, parts in acc.items():
        d = pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date")
        fq = qsrc / f"{bid}.csv.gz"
        if fq.exists():
            d = d.merge(pd.read_csv(fq, usecols=["date", "q_cfs"]), on="date", how="left")
        else:
            d["q_cfs"] = np.nan
        d[COLS].to_csv(out / f"{bid}.csv.gz", index=False, compression="gzip")
        written += 1
    print(f"\nwritten={written} -> {out}")

    s = pd.read_csv(out / f"{sorted(acc)[0]}.csv.gz")
    wet = s[s.precipitation_sum > 1]
    print(f"sample {sorted(acc)[0]}: rows={len(s)} {s.date.min()}..{s.date.max()}")
    print(f"  daily precip mean {s.precipitation_sum.mean():.3f} mm")
    print(f"  on WET days (>1mm, n={len(wet)}):")
    print(f"    p_max_1h   mean {wet.p_max_1h.mean():.3f} mm/h  max {wet.p_max_1h.max():.2f}")
    print(f"    p_max_3h   mean {wet.p_max_3h.mean():.3f} mm    max {wet.p_max_3h.max():.2f}")
    print(f"    p_hours    mean {wet.p_hours.mean():.1f} h")
    print(f"    p_cv       mean {wet.p_cv.mean():.3f}")
    print(f"    p_centroid mean {wet.p_centroid.mean():.1f} h")

    # the SWE lesson: verify the new columns actually VARY, not just exist
    if wet.p_max_1h.std() < 1e-6 or wet.p_hours.nunique() < 3:
        sys.exit("FATAL: intensity features are constant -- the sub-daily signal "
                 "was not captured")
    frac = (wet.p_max_1h / wet.precipitation_sum).median()
    print(f"\n  median share of a wet day's rain falling in its peak hour: {frac:.1%}")
    print("  (a daily sum alone cannot express this; that is the added signal)")


if __name__ == "__main__":
    main()
