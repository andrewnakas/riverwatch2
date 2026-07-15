#!/usr/bin/env python3
"""Adapt the RiverWatch2 CAMELS corpus → neuralhydrology GenericDataset format.

The Li/Shen 2025 CAMELS-531 record (median NSE ~0.83) is carried by a proper
sequence-to-one rainfall-runoff LSTM ensemble (Kratzert 2021, NSE 0.8082) — NOT
our MB-LSTM encoder-decoder. This converts our per-basin corpus into the exact
input format neuralhydrology's GenericDataset expects, so we can train the
reference CudaLSTM recipe and reproduce that 0.808 rung.

NH GenericDataset layout (verified from the docs):
  <out_dir>/
    time_series/<basin_id>.nc   # one netCDF/basin, coord 'date' (datetime),
                                #   data vars = the 5 forcings + 'q_cfs' target
    attributes/attributes.csv   # static attrs, indexed by basin id (col 'gauge_id')

Our corpus columns → NH dynamic inputs (the paper's 5 vars):
  precipitation_sum        -> prcp
  temperature_2m_max       -> tmax
  temperature_2m_min       -> tmin
  vapor_pressure           -> vp
  shortwave_radiation_sum  -> srad
Target: q_cfs (kept as-is; NSE loss is scale-free per basin). NaN targets are
kept (NH masks them). Statics = the 27 Addor attrs from camels_attrs.json.

Run ON the box where NH is installed (netCDF4 ships with NH). Per-forcing:
  python scripts/corpus_to_nh.py --forcing daymet \
    --corpus-dir data/gpu_corpora/camels_corpus_daymet_v2 \
    --out data/nh/daymet
Also writes basin list files (train = all 531; NH splits by DATE via config).
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

FORCING_MAP = {
    "precipitation_sum": "prcp",
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "vapor_pressure": "vp",
    "shortwave_radiation_sum": "srad",
}
DYNAMIC_INPUTS = list(FORCING_MAP.values())          # prcp,tmax,tmin,vp,srad
# The Kratzert/Li-Shen CAMELS record trains on SPECIFIC DISCHARGE (mm/day, area-
# normalized) — NOT raw cfs. With raw cfs the per-basin NSE loss is dominated by
# big-river basins (basin mean flow spans ~134x), underfitting the small basins
# that set the median NSE (our first gate: raw-cfs nldas = 0.716, below par).
# q_mm/day = q_cfs * 0.0283168 (m3/s per cfs) * 86400 (s/day) / (area_km2 * 1e6 m2)
#            * 1000 (mm/m)  =  q_cfs * 2.446576 / area_km2
CFS_TO_MMDAY_PER_KM2 = 0.0283168 * 86400 / 1e6 * 1000   # = 2.446...
TARGET = "q_mm"       # specific discharge — the record's target
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


def basin_to_nc(csv_path: Path, out_nc: Path, area_km2: float) -> bool:
    """One corpus csv.gz → NH netCDF (coord 'date', vars = 5 forcings + q_mm).
    Converts raw q_cfs → specific discharge (mm/day) using the basin area so the
    NSE loss is comparable across basins (the record's target)."""
    df = pd.read_csv(csv_path)
    if not {*FORCING_MAP, "q_cfs", "date"} <= set(df.columns):
        return False
    if not (area_km2 and np.isfinite(area_km2) and area_km2 > 0):
        return False
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    idx = pd.date_range(df.index[0], df.index[-1], freq="D")
    df = df.reindex(idx)
    q_mm = df["q_cfs"].to_numpy(dtype="float64") * CFS_TO_MMDAY_PER_KM2 / area_km2
    ds = xr.Dataset(coords={"date": idx.values})
    for src, dst in FORCING_MAP.items():
        ds[dst] = ("date", df[src].to_numpy(dtype="float32"))
    ds[TARGET] = ("date", q_mm.astype("float32"))
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forcing", required=True, choices=["daymet", "maurer", "nldas"])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True, help="NH data_dir to create")
    ap.add_argument("--limit", type=int, default=0, help="cap basins (smoke)")
    args = ap.parse_args()

    corpus = Path(args.corpus_dir)
    out = Path(args.out)
    ids_531 = set(load_camels_531(ROOT))
    attrs = json.loads((ROOT / "data" / "camels_attrs.json").read_text())

    files = sorted(p for p in corpus.glob("*.csv.gz")
                   if not p.name.startswith("._")
                   and p.name.split(".")[0] in ids_531)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no 531-subset corpus files in {corpus}", file=sys.stderr)
        return 1

    written, attr_rows = [], []
    for i, p in enumerate(files, 1):
        bid = p.name.split(".")[0]
        area = attrs.get(bid, {}).get("area_gages2")
        if basin_to_nc(p, out / "time_series" / f"{bid}.nc", area):
            written.append(bid)
            row = {"gauge_id": bid}
            a = attrs.get(bid, {})
            for k in STATIC_ATTRS:
                v = a.get(k)
                row[k] = float(v) if v is not None and np.isfinite(v) else np.nan
            attr_rows.append(row)
        if i % 100 == 0:
            print(f"[{i}/{len(files)}] wrote {len(written)} basins", flush=True)

    # attributes.csv (NH indexes by gauge_id); impute NaN attrs to column mean
    adf = pd.DataFrame(attr_rows).set_index("gauge_id")
    adf = adf.fillna(adf.mean(numeric_only=True))
    (out / "attributes").mkdir(parents=True, exist_ok=True)
    adf.to_csv(out / "attributes" / "attributes.csv")

    # basin list (all 531 written; NH splits train/val/test by DATE in the config)
    (out / "basins.txt").write_text("\n".join(written) + "\n")

    print(f"DONE {args.forcing}: {len(written)} basins -> {out}")
    print(f"  time_series/*.nc, attributes/attributes.csv ({len(STATIC_ATTRS)} attrs), "
          f"basins.txt")
    print(f"  dynamic_inputs: {DYNAMIC_INPUTS}  target: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
