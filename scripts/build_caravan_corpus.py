#!/usr/bin/env python3
"""Build MB-LSTM corpus files from base Caravan (Kratzert et al. 2023).

The CAMELS Daymet corpus ends 2014, but the MultiMet HRES eval needs encoder
history through 2016-2024. Base Caravan (Zenodo 7944025) ships per-basin
timeseries with observed streamflow + ERA5-Land daily forcing for the full
1981-present period, keyed 'camels_<usgs8>' — exactly the encoder history the
MultiMet scorer needs (scripts/backtest_multimet.py).

This writes the same compat-vars corpus schema build_camels_corpus.py does, so
the shipped 5-var model can encode Caravan history and be forced by MultiMet
HRES on the decoder side:

    date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
    precipitation_sum, shortwave_radiation_sum

Caravan -> corpus column mapping (ERA5-Land daily aggregates; VERIFY the exact
source names against a real file with --print-columns before a full run — the
paper documents the variables but per-release column spelling can differ):

    streamflow (mm/day)                 -> q_cfs      [see --q-units]
    temperature_2m_mean/_max/_min (C)   -> temperature_2m_* (direct)
    total_precipitation_sum (mm)        -> precipitation_sum
    surface_net_solar_radiation_mean    -> shortwave_radiation_sum (MJ/m2; ERA5
        (W/m2 or J/m2, release-dependent)   net-solar is a proxy for the
                                            Daymet shortwave-down the model was
                                            trained on — flagged, not hidden)

NOTE on q units: Caravan streamflow is mm/day (area-normalized). The model was
trained on q_cfs. With --q-units mm the column is written as-is (a
Caravan-native model would retrain on it); --q-units cfs converts using each
basin's drainage area from the Caravan attributes (area km^2). The MultiMet
scorer compares to AIFL/Google who score in the normalized space, so mm is the
apples-to-apples default there — but the SHIPPED model expects cfs. Resolve
per eval; default mm with a loud note.

Usage:
  .venv/bin/python scripts/build_caravan_corpus.py --print-columns   # inspect
  .venv/bin/python scripts/build_caravan_corpus.py \
      --caravan-dir /Volumes/STORAGE_SD/riverwatch2_data/external/caravan_base/Caravan \
      --ids-json data/camels_gauge_ids.json --ids-key 531 \
      --out-dir /Volumes/STORAGE_SD/riverwatch2_data/caravan_corpus_531
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CARAVAN = Path("/Volumes/STORAGE_SD/riverwatch2_data/external/caravan_base/Caravan")
DEFAULT_OUT = Path("/Volumes/STORAGE_SD/riverwatch2_data/caravan_corpus_531")

CORPUS_COLS = [
    "date", "q_cfs",
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum",
]

# Candidate Caravan source column names per corpus field (first match wins);
# ordered by how the Caravan/GRDC-Caravan releases have spelled them.
SRC_CANDIDATES = {
    "q": ["streamflow"],
    "date": ["date", "Date"],
    "temperature_2m_mean": ["temperature_2m_mean"],
    "temperature_2m_max": ["temperature_2m_max"],
    "temperature_2m_min": ["temperature_2m_min"],
    "precipitation_sum": ["total_precipitation_sum", "total_precipitation"],
    "shortwave": ["surface_net_solar_radiation_mean",
                  "surface_net_solar_radiation", "surface_net_solar_radiation_sum"],
}


# mm/day over a km^2 basin -> cfs:
#   q[mm/d] * A[km^2] = 1e-3 m * 1e6 m^2 / day = 1e3 m^3/day
#   / 86400 s/day * 35.3147 ft^3/m^3
MM_KM2_TO_CFS = 1e3 / 86400.0 * 35.3147


def find_basin_file(caravan_dir: Path, usgs_id: str) -> Path | None:
    """Caravan timeseries live under timeseries/{csv,netcdf}/camels/camels_<id>.*"""
    key = f"camels_{usgs_id}"
    for ext in ("csv", "nc"):
        for cand in caravan_dir.rglob(f"{key}.{ext}"):
            if not cand.name.startswith("._"):
                return cand
    return None


def load_areas(caravan_dir: Path) -> dict[str, float]:
    """usgs_id -> basin area km^2 from Caravan's 'other' attribute table
    (validated against our CAMELS q_cfs corpus within ~6%)."""
    for p in caravan_dir.rglob("attributes_other_camels.csv"):
        if p.name.startswith("._"):
            continue
        a = pd.read_csv(p)
        return {str(g).split("_", 1)[1]: float(ar)
                for g, ar in zip(a["gauge_id"], a["area"]) if pd.notna(ar)}
    return {}


def _resolve(cols: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def read_basin(path: Path) -> pd.DataFrame:
    if path.suffix == ".nc":
        import xarray as xr
        df = xr.open_dataset(path).to_dataframe().reset_index()
    else:
        df = pd.read_csv(path)
    cols = set(df.columns)
    dc = _resolve(cols, SRC_CANDIDATES["date"])
    date = pd.to_datetime(df[dc])

    def col(field):
        src = _resolve(cols, SRC_CANDIDATES[field])
        return df[src].astype(float) if src else pd.Series(np.nan, index=df.index)

    q = col("q")
    q = q.mask(q < 0, np.nan)  # Caravan uses NaN, but guard stray negatives
    out = pd.DataFrame({
        "date": date,
        "q_cfs": q,  # mm/day; see --q-units in main()
        "temperature_2m_mean": col("temperature_2m_mean"),
        "temperature_2m_max": col("temperature_2m_max"),
        "temperature_2m_min": col("temperature_2m_min"),
        "precipitation_sum": col("precipitation_sum"),
        "shortwave_radiation_sum": col("shortwave"),
    })
    return out.sort_values("date").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caravan-dir", type=Path, default=DEFAULT_CARAVAN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ids-json", type=Path, default=REPO / "data" / "camels_gauge_ids.json")
    ap.add_argument("--ids-key", default="531")
    ap.add_argument("--q-units", choices=["mm", "cfs"], default="mm",
                    help="mm = Caravan-native (AIFL/Google-comparable); cfs = "
                         "convert via basin area for the shipped q_cfs model")
    ap.add_argument("--print-columns", action="store_true",
                    help="print the first found basin file's columns and exit "
                         "(use to VERIFY the SRC_CANDIDATES mapping)")
    args = ap.parse_args()

    ids = json.load(open(args.ids_json))[args.ids_key]
    if args.print_columns:
        for gid in ids:
            f = find_basin_file(args.caravan_dir, gid)
            if f:
                print(f"sample basin file: {f}")
                if f.suffix == ".nc":
                    import xarray as xr
                    print("variables:", list(xr.open_dataset(f).variables))
                else:
                    print("columns:", list(pd.read_csv(f, nrows=1).columns))
                for field, cands in SRC_CANDIDATES.items():
                    cols = (set(xr.open_dataset(f).variables) if f.suffix == ".nc"
                            else set(pd.read_csv(f, nrows=1).columns))
                    print(f"  {field}: {_resolve(cols, cands) or 'NOT FOUND — fix SRC_CANDIDATES'}")
                return 0
        print("no basin files found — is --caravan-dir extracted?", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    areas = load_areas(args.caravan_dir) if args.q_units == "cfs" else {}
    if args.q_units == "cfs" and not areas:
        print("FATAL: --q-units cfs needs attributes_other_camels.csv (area km^2) "
              "but none was found under --caravan-dir", file=sys.stderr)
        return 1
    built, missing, no_q, no_area = 0, [], [], []
    for i, gid in enumerate(ids, 1):
        f = find_basin_file(args.caravan_dir, gid)
        if f is None:
            missing.append(gid)
            continue
        df = read_basin(f)
        if args.q_units == "cfs":
            # Caravan streamflow is mm/day (area-normalized); the shipped model
            # was trained on q_cfs and its per-station asinh transform is NOT
            # scale-invariant, so this conversion is mandatory (not cosmetic).
            area = areas.get(gid)
            if area is None:
                no_area.append(gid)
                df["q_cfs"] = np.nan
            else:
                df["q_cfs"] = df["q_cfs"] * area * MM_KM2_TO_CFS
        if df["q_cfs"].notna().sum() == 0:
            no_q.append(gid)
        df = df[CORPUS_COLS]
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df.to_csv(args.out_dir / f"{gid}.csv.gz", index=False, compression="gzip")
        built += 1
        if i % 50 == 0:
            print(f"{i}/{len(ids)}", flush=True)

    print(f"built {built} basins -> {args.out_dir} (q_units={args.q_units})")
    if missing:
        print(f"WARNING: {len(missing)} ids not found in Caravan: {missing[:10]}")
    if no_area:
        print(f"WARNING: {len(no_area)} basins had no area (q_cfs=NaN): {no_area[:10]}")
    if no_q:
        print(f"WARNING: {len(no_q)} basins had no streamflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
