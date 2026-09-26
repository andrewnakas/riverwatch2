#!/usr/bin/env python
"""Build a MULTI-FORCING NeuralHydrology GenericDataset for the Li/Song 2025
"LSTMmulti" member — one CudaLSTM fed all three CAMELS forcing products at once
(Kratzert-2021 multi-forcing input layout). This is the ensemble stream that
seed-averages 0.797 -> 0.824 (+0.027), the single biggest lever in the paper's
0.83 recipe, which the per-forcing datasets can't produce.

Companion to corpus_to_nh.py (single-forcing). Reads the daymet/nldas/maurer
corpora and writes ONE netCDF per basin whose data vars are the 5 forcing vars
SUFFIXED per product (15 total) + the shared q_mm target:

    prcp_daymet, tmax_daymet, ... , srad_daymet,
    prcp_nldas,  ... , srad_nldas,
    prcp_maurer, ... , srad_maurer,
    q_mm

The NH config's dynamic_inputs then lists all 15. Discharge is identical across
the three corpora for a basin (same USGS gauge), so q_mm is taken from daymet.

Run ON the box (netCDF4 ships with NH):
  python scripts/corpus_to_nh_multi.py \
    --daymet gpu1080/corpora/camels_corpus_daymet_v2 \
    --nldas  gpu1080/corpora/camels_corpus_nldas_v2 \
    --maurer gpu1080/corpora/camels_corpus_maurer_v2 \
    --out gpu1080/nh_data_multi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]

# same var mapping as corpus_to_nh.py, applied per forcing with a suffix
FORCING_MAP = {
    "precipitation_sum": "prcp",
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "vapor_pressure": "vp",
    "shortwave_radiation_sum": "srad",
}
# MODERN-1: the product list is now set at runtime (--products), because the
# modern corpus cannot contain maurer -- CAMELS maurer ends 2008-12-31. The
# default below is unchanged so every existing LEDGER 51/53/54/56 NH build
# reproduces byte-for-byte.
FORCINGS = ["daymet", "nldas", "maurer"]
# dynamic inputs = 5 vars x N forcings, suffixed
DYNAMIC_INPUTS = [f"{v}_{f}" for f in FORCINGS for v in FORCING_MAP.values()]


def set_forcings(names: list[str]) -> None:
    """Rebind the module-level forcing list and its derived input names."""
    global FORCINGS, DYNAMIC_INPUTS
    FORCINGS = list(names)
    DYNAMIC_INPUTS = [f"{v}_{f}" for f in FORCINGS for v in FORCING_MAP.values()]

CFS_TO_MMDAY_PER_KM2 = 0.0283168 * 86400 / 1e6 * 1000  # = 2.446...
TARGET = "q_mm"
# identical 27 Addor static set as the single-forcing adapter
STATIC_ATTRS = [
    "p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2", "soil_depth_pelletier",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
    "max_water_content", "sand_frac", "silt_frac", "clay_frac",
    "frac_forest", "lai_max", "gvf_max", "gvf_diff", "root_depth_50",
    "carbonate_rocks_frac", "geol_permeability",
]


def load_camels_531(root: Path) -> list[str]:
    d = json.loads((root / "data" / "camels_gauge_ids.json").read_text())
    return [str(x).strip().zfill(8) for x in d["531"]]


def read_forcing_csv(path: Path) -> pd.DataFrame | None:
    """Return a daily-reindexed df with the 5 forcing cols + q_cfs, or None."""
    df = pd.read_csv(path)
    if not {*FORCING_MAP, "q_cfs", "date"} <= set(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    idx = pd.date_range(df.index[0], df.index[-1], freq="D")
    return df.reindex(idx)


def build_basin(bid: str, corpora: dict[str, Path], area_km2: float,
                out_nc: Path) -> bool:
    """Merge the 3 forcings for one basin into one NH netCDF (15 vars + q_mm).
    Aligns all forcings on a common daily index (their intersection)."""
    if not (area_km2 and np.isfinite(area_km2) and area_km2 > 0):
        return False
    frames = {}
    for f in FORCINGS:
        p = corpora[f] / f"{bid}.csv.gz"
        if not p.exists():
            return False
        d = read_forcing_csv(p)
        if d is None:
            return False
        frames[f] = d
    # common daily index across all three (they share the CAMELS 1980-2014 span)
    # ⚠️ Was frames["daymet"], which made the whole adapter unusable for any product
    # set lacking daymet -- KeyError: 'daymet'. My earlier "both branches verified"
    # test happened to include daymet in the product list, so it never exercised
    # this. Use the FIRST configured product as the reference index instead.
    common = frames[FORCINGS[0]].index
    for f in FORCINGS[1:]:
        common = common.intersection(frames[f].index)
    if len(common) < 3650:  # need ~a decade minimum
        return False
    ds = xr.Dataset(coords={"date": common.values})
    for f in FORCINGS:
        df = frames[f].reindex(common)
        for src, dst in FORCING_MAP.items():
            ds[f"{dst}_{f}"] = ("date", df[src].to_numpy(dtype="float32"))
    # discharge: identical gauge across forcings -> take daymet's, convert to mm/day
    # discharge is identical across products (all copied from one store)
    q_cfs = frames[FORCINGS[0]].reindex(common)["q_cfs"].to_numpy(dtype="float64")
    q_mm = q_cfs * CFS_TO_MMDAY_PER_KM2 / area_km2
    ds[TARGET] = ("date", q_mm.astype("float32"))
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--daymet", default="")
    ap.add_argument("--nldas", default="")
    ap.add_argument("--maurer", default="")
    ap.add_argument("--products", default="",
                    help="MODERN-1: name=dir,name=dir,... in the order the "
                         "suffixed inputs should appear. Overrides the three "
                         "legacy flags; those stay for reproducing LEDGER 51-56.")
    ap.add_argument("--basin-list", default="",
                    help="JSON with a 'cohort' list (e.g. data/m1_cohort.json); "
                         "default is the 531 Kratzert subset")
    ap.add_argument("--out", required=True, help="NH data_dir to create")
    ap.add_argument("--limit", type=int, default=0, help="cap basins (smoke)")
    args = ap.parse_args()

    if args.products:
        corpora = {}
        for item in args.products.split(","):
            if "=" not in item:
                print(f"--products entry {item!r} is not name=dir", file=sys.stderr)
                return 1
            k, v = item.split("=", 1)
            corpora[k] = Path(v)
        set_forcings(list(corpora))
    else:
        for f in ("daymet", "nldas", "maurer"):
            if not getattr(args, f):
                print(f"--{f} is required unless --products is given", file=sys.stderr)
                return 1
        corpora = {"daymet": Path(args.daymet), "nldas": Path(args.nldas),
                   "maurer": Path(args.maurer)}
    for k, v in corpora.items():
        if not v.exists():
            print(f"corpus for {k} does not exist: {v}", file=sys.stderr)
            return 1
    out = Path(args.out)
    if args.basin_list:
        d = json.loads(Path(args.basin_list).read_text())
        ids_531 = [str(x).zfill(8) for x in (d.get("cohort") or d["531"])]
        print(f"basin list: {args.basin_list} -> {len(ids_531)} basins")
    else:
        ids_531 = load_camels_531(ROOT)
    attrs = json.loads((ROOT / "data" / "camels_attrs.json").read_text())
    if args.limit:
        ids_531 = ids_531[: args.limit]

    written, attr_rows = [], []
    for i, bid in enumerate(ids_531, 1):
        area = attrs.get(bid, {}).get("area_gages2")
        if build_basin(bid, corpora, area, out / "time_series" / f"{bid}.nc"):
            written.append(bid)
            row = {"gauge_id": bid}
            a = attrs.get(bid, {})
            for k in STATIC_ATTRS:
                v = a.get(k)
                row[k] = float(v) if v is not None and np.isfinite(v) else np.nan
            attr_rows.append(row)
        if i % 100 == 0:
            print(f"[{i}/{len(ids_531)}] wrote {len(written)} basins", flush=True)

    if not written:
        print("no basins written — check corpus paths", file=sys.stderr)
        return 1
    adf = pd.DataFrame(attr_rows).set_index("gauge_id")
    adf = adf.fillna(adf.mean(numeric_only=True))
    (out / "attributes").mkdir(parents=True, exist_ok=True)
    adf.to_csv(out / "attributes" / "attributes.csv")
    (out / "basins.txt").write_text("\n".join(written) + "\n")
    print(f"DONE multi: {len(written)} basins -> {out}")
    print(f"  dynamic_inputs ({len(DYNAMIC_INPUTS)}): {DYNAMIC_INPUTS}")
    print(f"  target: {TARGET}, statics: {len(STATIC_ATTRS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
