#!/usr/bin/env python
"""Build MB-LSTM corpus files from the extracted CAMELS-US v1.2 archive.

Reads a basin-mean meteorological forcing set (Daymet, Maurer, or NLDAS) +
USGS streamflow from the extracted basin_timeseries_v1p2_metForcing_obsFlow
archive and writes one csv.gz per basin in the MB-LSTM corpus schema
(compat-vars subset):

    date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
    precipitation_sum, shortwave_radiation_sum, vapor_pressure

(vapor_pressure added 2026-07-10 for the Kratzert-2021 recipe-v2 6-var input.)
All three CAMELS forcing products share the same column layout
(`Year Mnth Day Hr Dayl(s) PRCP SRAD SWE Tmax Tmin Vp`), so one parser and
unit mapping cover them; only the subdirectory + filename suffix differ
(--forcing-set). Unit mapping (product -> corpus):
    tmax(C), tmin(C)            -> temperature_2m_max / _min (direct)
    (tmax+tmin)/2               -> temperature_2m_mean
    prcp(mm/day)                -> precipitation_sum (mm)
    srad(W/m2, daylight avg) * dayl(s) / 1e6
                                -> shortwave_radiation_sum (MJ/m2/day)
    streamflow (cfs, -999=missing) -> q_cfs (NaN for missing)

Full forcing range (1980-01-01 .. 2014-12-31) is kept; streamflow is
left-joined onto the forcing dates.

--merge writes ONE csv.gz per basin with all requested products'
weather columns suffixed (_daymet/_maurer/_nldas) alongside a single q_cfs —
the Kratzert-2021 multi-forcing input layout (train_mblstm --enc-vars
camels3f). Without --merge, each product writes to its own out-dir so a
merged corpus can be assembled later or products trained separately.
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

WEATHER_COLS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "vapor_pressure",
]
CORPUS_COLS = ["date", "q_cfs", *WEATHER_COLS]

# subdir under basin_mean_forcing + filename suffix, per product.
FORCING_SETS = {
    "daymet": ("basin_mean_forcing/daymet", "_lump_cida_forcing_leap.txt"),
    "maurer": ("basin_mean_forcing/maurer", "_lump_maurer_forcing_leap.txt"),
    "nldas": ("basin_mean_forcing/nldas", "_lump_nldas_forcing_leap.txt"),
}


def index_files(root: Path, sub: str, suffix: str) -> dict[str, Path]:
    """Map 8-digit gauge id -> file path, skipping exFAT '._*' AppleDouble files."""
    out: dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root / sub):
        for fn in filenames:
            if fn.startswith("._") or not fn.endswith(suffix):
                continue
            out[fn[:8]] = Path(dirpath) / fn
    return out


def read_forcing(path: Path) -> pd.DataFrame:
    # 3 header lines (lat, elev, area) then a column-name line, then data.
    # Daymet/Maurer/NLDAS share this exact layout (verified 2026-07-06).
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
            "vapor_pressure": df["vp"],  # Pa; Kratzert-2021 6th CAMELS input
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


def build_basin(gid: str, forcing_paths: dict[str, Path], sf_path: Path | None,
                out_dir: Path, merge: bool) -> dict:
    """forcing_paths maps product name -> forcing file. With merge=False there
    is exactly one product (per-product corpus); with merge=True the products'
    weather columns are joined on date and suffixed _<product>."""
    products = list(forcing_paths)
    frames = {p: read_forcing(path).set_index("date") for p, path in forcing_paths.items()}
    if merge and len(products) > 1:
        parts = []
        for p in products:
            parts.append(frames[p][WEATHER_COLS].add_suffix(f"_{p}"))
        wx = pd.concat(parts, axis=1, join="inner").reset_index()
    else:
        wx = frames[products[0]][WEATHER_COLS].reset_index()

    if sf_path is not None:
        sf = read_streamflow(sf_path)
        df = wx.merge(sf, on="date", how="left")
    else:
        df = wx.assign(q_cfs=np.nan)

    weather_out = [c for c in df.columns if c not in ("date", "q_cfs")]
    df = df[["date", "q_cfs", *weather_out]].sort_values("date").reset_index(drop=True)
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
    ap.add_argument("--forcing-set", default="daymet",
                    help="comma-separated products from {daymet,maurer,nldas}; "
                         "multiple with --merge writes one 3-forcing corpus, "
                         "without --merge each writes to <out-dir>_<product>")
    ap.add_argument("--merge", action="store_true",
                    help="join the requested products' weather columns "
                         "(_<product> suffix) into one corpus per basin")
    args = ap.parse_args()

    products = [p.strip() for p in args.forcing_set.split(",") if p.strip()]
    bad = [p for p in products if p not in FORCING_SETS]
    if bad:
        print(f"FATAL: unknown forcing set(s) {bad}; choose from {list(FORCING_SETS)}",
              file=sys.stderr)
        return 1

    ids = json.load(open(args.ids_json))[args.ids_key]
    sflow = index_files(args.raw_dir, "usgs_streamflow", "_streamflow_qc.txt")
    indexes = {}
    for p in products:
        sub, suffix = FORCING_SETS[p]
        idx = index_files(args.raw_dir, sub, suffix)
        miss = [g for g in ids if g not in idx]
        if miss:
            print(f"FATAL: {len(miss)} ids missing {p} forcing: {miss[:10]}", file=sys.stderr)
            return 1
        indexes[p] = idx

    # merge → one out-dir; separate → one out-dir per product.
    if args.merge and len(products) > 1:
        out_dirs = {tuple(products): args.out_dir}
    else:
        out_dirs = {(p,): (args.out_dir if len(products) == 1
                           else Path(f"{args.out_dir}_{p}")) for p in products}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    built, no_flow = 0, []
    for i, gid in enumerate(ids, 1):
        sf_path = sflow.get(gid)
        if sf_path is None:
            no_flow.append(gid)
        for prod_tuple, out_dir in out_dirs.items():
            fpaths = {p: indexes[p][gid] for p in prod_tuple}
            build_basin(gid, fpaths, sf_path, out_dir, merge=args.merge)
        built += 1
        if i % 50 == 0:
            print(f"{i}/{len(ids)}", flush=True)

    print(f"built {built} basins × {len(out_dirs)} corpus(es) "
          f"({', '.join(products)}, merge={args.merge})")
    for pt, d in out_dirs.items():
        print(f"  {'+'.join(pt)} -> {d}")
    if no_flow:
        print(f"WARNING: {len(no_flow)} basins had no streamflow file: {no_flow}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
