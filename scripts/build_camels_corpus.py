#!/usr/bin/env python
"""Build MB-LSTM corpus files from the extracted CAMELS-US v1.2 archive.

Reads Daymet basin-mean forcings + USGS streamflow from the extracted
basin_timeseries_v1p2_metForcing_obsFlow archive and writes one csv.gz per
basin in the MB-LSTM corpus schema (compat-vars subset):

    date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
    precipitation_sum, shortwave_radiation_sum

Unit mapping (Daymet -> corpus):
    tmax(C), tmin(C)            -> temperature_2m_max / _min (direct)
    (tmax+tmin)/2               -> temperature_2m_mean
    prcp(mm/day)                -> precipitation_sum (mm)
    srad(W/m2, daylight avg) * dayl(s) / 1e6
                                -> shortwave_radiation_sum (MJ/m2/day)
    streamflow (cfs, -999=missing) -> q_cfs (NaN for missing)

Full Daymet range (1980-01-01 .. 2014-12-31) is kept; streamflow is
left-joined onto the forcing dates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RAW = Path("/Volumes/STORAGE_SD/riverwatch2_data/camels_raw/basin_dataset_public_v1p2")
DEFAULT_OUT = Path("/Volumes/STORAGE_SD/riverwatch2_data/camels_corpus")

CORPUS_COLS = [
    "date",
    "q_cfs",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
]


def index_files(root: Path, sub: str, suffix: str) -> dict[str, Path]:
    """Map 8-digit gauge id -> file path, skipping exFAT '._*' AppleDouble files."""
    out: dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root / sub):
        for fn in filenames:
            if fn.startswith("._") or not fn.endswith(suffix):
                continue
            out[fn[:8]] = Path(dirpath) / fn
    return out


def read_daymet(path: Path) -> pd.DataFrame:
    # 3 header lines (lat, elev, area) then a column-name line, then data.
    df = pd.read_csv(
        path,
        skiprows=4,
        sep=r"\s+",
        header=None,
        names=["year", "mnth", "day", "hr", "dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"],
    )
    date = pd.to_datetime(dict(year=df["year"], month=df["mnth"], day=df["day"]))
    out = pd.DataFrame(
        {
            "date": date,
            "temperature_2m_mean": (df["tmax"] + df["tmin"]) / 2.0,
            "temperature_2m_max": df["tmax"],
            "temperature_2m_min": df["tmin"],
            "precipitation_sum": df["prcp"],
            "shortwave_radiation_sum": df["srad"] * df["dayl"] / 1e6,
        }
    )
    return out


def read_streamflow(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["gauge", "year", "month", "day", "q_cfs", "flag"],
        dtype={"gauge": str},
    )
    date = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    q = df["q_cfs"].astype(float)
    q = q.mask(q <= -998.0, np.nan)  # -999 sentinel -> NaN
    return pd.DataFrame({"date": date, "q_cfs": q})


def build_basin(gid: str, daymet_path: Path, sf_path: Path | None, out_dir: Path) -> dict:
    forc = read_daymet(daymet_path)
    if sf_path is not None:
        sf = read_streamflow(sf_path)
        df = forc.merge(sf, on="date", how="left")
    else:
        df = forc.assign(q_cfs=np.nan)
    df = df[CORPUS_COLS].sort_values("date").reset_index(drop=True)
    stats = {
        "id": gid,
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "q_valid": int(df["q_cfs"].notna().sum()),
    }
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(out_dir / f"{gid}.csv.gz", index=False, compression="gzip")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ids-json", type=Path, default=REPO / "data" / "camels_gauge_ids.json")
    ap.add_argument("--ids-key", default="531")
    args = ap.parse_args()

    ids = json.load(open(args.ids_json))[args.ids_key]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    daymet = index_files(args.raw_dir, "basin_mean_forcing/daymet", "_lump_cida_forcing_leap.txt")
    sflow = index_files(args.raw_dir, "usgs_streamflow", "_streamflow_qc.txt")

    missing = [g for g in ids if g not in daymet]
    if missing:
        print(f"FATAL: {len(missing)} ids missing daymet forcing: {missing[:10]}", file=sys.stderr)
        return 1

    built, no_flow = 0, []
    for i, gid in enumerate(ids, 1):
        sf_path = sflow.get(gid)
        if sf_path is None:
            no_flow.append(gid)
        build_basin(gid, daymet[gid], sf_path, args.out_dir)
        built += 1
        if i % 50 == 0:
            print(f"{i}/{len(ids)}", flush=True)

    print(f"built {built} basins -> {args.out_dir}")
    if no_flow:
        print(f"WARNING: {len(no_flow)} basins had no streamflow file: {no_flow}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
