#!/usr/bin/env python3
"""Fold the reconstructed basin 13235000 into the AORC corpus as the 531st basin.

HydroShare's AORC product covers 667 basins but omits 13235000, leaving our
member at 530 of the benchmark 531. We reconstructed it from the official NOAA
zarr using HydroShare's own documented method (GAGES-II polygon +
coverage-weighted mean, per jmframe/CIROH_DL_NextGen/forcing_prep).

Justification for reconstructing rather than dropping, measured on six basins
HydroShare DID publish, spanning 4-588 km2:

    area_km2   ratio    corr
           4   1.048   0.9977
          61   1.022   0.9935
         168   0.993   0.9896
         294   0.759   0.9907
         588   1.226   0.9892

SHAPE reproduces at 0.989-0.998 at every size; only SCALE wanders, and it
wanders independently of correlation. Per-basin normalization in the trainer
absorbs a scale offset but cannot recover a corrupted storm sequence -- so
corr ~0.99 is the statistic that matters for a model INPUT, and it is excellent.

Aggregation matches build_aorc_corpus.py exactly so the new row is consistent
with the other 530:
    precip  summed over the day
    tmax/tmin  TRUE daily extremes from the hourly temperature curve
    srad    mean W/m2 * 0.0864 -> MJ/m2/day
    vp      e = q*p/(0.622 + 0.378*q)
    q_cfs   copied from the daymet corpus (a gauge observation, product-independent)

Writes a provenance marker so any downstream result can state which basins came
from the published product and which one we rebuilt.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASIN = "13235000"
RAW = Path("data/aorc_extra")
CORPUS = Path("data/local_corpora/camels_corpus_aorc_v2")
QSRC = Path("data/local_corpora/camels_corpus_daymet_v2")
COLS = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
        "vapor_pressure"]


def main():
    files = sorted(RAW.glob(f"{BASIN}_WY*.csv"))
    print(f"water-year files found: {len(files)}/31")
    if len(files) < 31:
        sys.exit(f"incomplete: need 31 files (1980 supplies the\n                 1980-10-01..12-31 head), have {len(files)}")

    parts = []
    for f in files:
        d = pd.read_csv(f)
        d["time"] = pd.to_datetime(d["time"])
        parts.append(d)
    h = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    print(f"hourly rows: {len(h)}  {h.time.min()} .. {h.time.max()}")

    tc = h["TMP_2maboveground"].astype(float) - 273.15
    q = h["SPFH_2maboveground"].astype(float)
    p = h["PRES_surface"].astype(float)
    w = pd.DataFrame({
        "date": h["time"].dt.strftime("%Y-%m-%d"),
        "tc": tc,
        "prcp": h["APCP_surface"].astype(float),
        "srad": h["DSWRF_surface"].astype(float),
        "vp": q * p / (0.622 + 0.378 * q),
    })
    g = w.groupby("date", sort=True)
    d = pd.DataFrame({
        "date": list(g.groups.keys()),
        "temperature_2m_mean": g["tc"].mean().round(3).values,
        "temperature_2m_max": g["tc"].max().round(3).values,
        "temperature_2m_min": g["tc"].min().round(3).values,
        "precipitation_sum": g["prcp"].sum().round(4).values,
        "shortwave_radiation_sum": (g["srad"].mean() * 0.0864).round(6).values,
        "vapor_pressure": g["vp"].mean().round(2).values,
    })

    fq = QSRC / f"{BASIN}.csv.gz"
    if not fq.exists():
        sys.exit(f"no discharge source at {fq}")
    qd = pd.read_csv(fq, usecols=["date", "q_cfs"])
    d = d.merge(qd, on="date", how="left")

    # match the window the other 530 basins cover
    ref = pd.read_csv(sorted(CORPUS.glob("*.csv.gz"))[0], usecols=["date"])
    lo, hi = ref.date.min(), ref.date.max()
    d = d[(d.date >= lo) & (d.date <= hi)].reset_index(drop=True)

    print(f"\ndaily rows: {len(d)}  {d.date.min()} .. {d.date.max()}  (others: {lo} .. {hi})")
    spread = (d.temperature_2m_max - d.temperature_2m_min).mean()
    print(f"  precip mean      {d.precipitation_sum.mean():.3f} mm/day")
    print(f"  tmax-tmin spread {spread:.2f} C")
    print(f"  q_cfs finite     {d.q_cfs.notna().mean():.3f}")

    # the SWE lesson: check values, not just that a column exists
    if spread < 1.0:
        sys.exit("FATAL: zero diurnal range -- AORC's whole advantage is missing")
    if not (0.5 < d.precipitation_sum.mean() < 20):
        sys.exit(f"FATAL: implausible precip mean {d.precipitation_sum.mean()}")
    if d.q_cfs.notna().mean() < 0.5:
        sys.exit("FATAL: discharge mostly missing")
    if len(d) != len(ref):
        print(f"  WARNING: {len(d)} rows vs {len(ref)} in the reference basin")

    out = CORPUS / f"{BASIN}.csv.gz"
    d[COLS].to_csv(out, index=False, compression="gzip")
    print(f"\nwrote {out}")

    prov = CORPUS / "PROVENANCE.json"
    json.dump({
        "published_source": "HydroShare c738c05278a34bc9848dd14d61cffab9 (667 basins)",
        "reconstructed": {
            BASIN: {
                "reason": "absent from the HydroShare product",
                "method": "GAGES-II polygon + exactextract coverage-weighted mean "
                          "over s3://noaa-nws-aorc-v1-1-1km, matching the published recipe",
                "validation": "6 published basins 4-588 km2 reproduce at precip "
                              "corr 0.989-0.998; scale ratio varies 0.76-1.23 and is "
                              "absorbed by per-basin normalization",
                "polygon_area_km2": 1163,
                "camels_area_km2": 1163,
            }
        },
    }, open(prov, "w"), indent=1)
    print(f"wrote {prov}")
    print(f"\ncorpus now {len(list(CORPUS.glob('*.csv.gz')))} basins")


if __name__ == "__main__":
    main()
