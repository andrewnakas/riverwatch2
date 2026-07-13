#!/usr/bin/env python
"""Parse CAMELS-US catchment attributes v2.0 → data/camels_attrs.json.

The recipe-v2 (2026-07-10) static set: the 27 numeric Addor-et-al-2017
catchment attributes that Kratzert 2019/2021's LSTM models use (train_mblstm.py
--static-set camels). Source files (semicolon-delimited, keyed gauge_id, 671
US basins) downloaded from Zenodo record 15529996 to
camels_raw/camels_attributes_v2.0/{clim,topo,soil,vege,geol}.txt.

Categorical attributes (dom_land_cover, geol_1st/2nd_class, *_prec_timing) are
excluded — the model consumes a fixed-width numeric static vector, and these
would need one-hot encoding that the 27-attr Kratzert set does not include.

Output: {gauge_id(8-digit): {attr: float, ...}, ...}, NaN for missing cells
(the trainer median-imputes, matching the gages2 path).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path(
    "/Volumes/STORAGE_SD/riverwatch2_data/camels_raw/camels_attributes_v2.0")
OUT = REPO / "data" / "camels_attrs.json"

# The 27 numeric Kratzert/Addor attributes, grouped by source file. gauge_lat/
# gauge_lon/elev/slope/area come from topo; the rest are the standard clim/
# soil/vege/geol descriptors. (Matches the neuralhydrology CAMELS-US default
# static set minus the categorical dom_land_cover / geol class fields.)
CAMELS_ATTRS = {
    "clim": ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
             "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"],
    "topo": ["elev_mean", "slope_mean", "area_gages2"],
    "soil": ["soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
             "soil_conductivity", "max_water_content", "sand_frac", "silt_frac",
             "clay_frac"],
    "vege": ["frac_forest", "lai_max", "gvf_max", "gvf_diff", "root_depth_50"],
    "geol": ["carbonate_rocks_frac", "geol_permeability"],
}


def main() -> int:
    src = DEFAULT_SRC
    merged: dict[str, dict[str, float]] = {}
    all_attrs: list[str] = []
    for group, cols in CAMELS_ATTRS.items():
        path = src / f"camels_{group}.txt"
        if not path.exists():
            print(f"FATAL: missing {path}")
            return 1
        df = pd.read_csv(path, sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].str.zfill(8)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"FATAL: {group} missing columns {missing}; "
                  f"have {list(df.columns)}")
            return 1
        for _, row in df.iterrows():
            gid = row["gauge_id"]
            rec = merged.setdefault(gid, {})
            for c in cols:
                v = pd.to_numeric(row[c], errors="coerce")
                rec[c] = float(v) if pd.notna(v) else float("nan")
        all_attrs.extend(cols)

    # Serialize NaN as null so json round-trips (trainer imputes on load).
    def _clean(d):
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in d.items()}

    out = {gid: _clean(rec) for gid, rec in sorted(merged.items())}
    OUT.write_text(json.dumps(out))
    n_full = sum(1 for r in out.values() if len(r) == len(all_attrs))
    print(f"wrote {len(out)} basins × {len(all_attrs)} attrs → {OUT}")
    print(f"  {n_full} basins with all attrs present; attr order: {all_attrs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
