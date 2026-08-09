#!/usr/bin/env python3
"""Bridge the INTENSITY corpus into the neuralhydrology layout.

Mirrors build_nh_aorc.py exactly, except the netCDF carries five extra dynamic
variables describing within-day rainfall structure:

    p_max_1h    peak hourly rate
    p_max_3h    peak rolling 3-hour total
    p_hours     hours with measurable rain
    p_cv        burstiness of the hourly series
    p_centroid  intensity-weighted hour of day (-1 on dry days)

The daily columns are byte-identical to the plain AORC corpus (verified: max
daily-precip difference 0.000000 mm), so a member trained here differs from the
plain AORC member in exactly one respect — it can see how the rain arrived
within each day. That isolates the sub-daily signal as the only explanation for
any score difference.

The q_cfs -> q_mm conversion was verified numerically at ratio 1.0000000 when
the original bridge was built; the same constant is reused.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SRC = Path("gpu1080/corpora671/camels_corpus_aorcx_v2")
REF = Path("gpu1080/nh_data/daymet")
OUT = Path("gpu1080/nh_data/aorcx")

MAP = {"prcp": "precipitation_sum", "tmax": "temperature_2m_max",
       "tmin": "temperature_2m_min", "vp": "vapor_pressure",
       "srad": "shortwave_radiation_sum",
       # the sub-daily block
       "p_max_1h": "p_max_1h", "p_max_3h": "p_max_3h",
       "p_hours": "p_hours", "p_cv": "p_cv", "p_centroid": "p_centroid"}


def main():
    (OUT / "time_series").mkdir(parents=True, exist_ok=True)
    (OUT / "attributes").mkdir(parents=True, exist_ok=True)
    shutil.copy(REF / "attributes" / "attributes.csv",
                OUT / "attributes" / "attributes.csv")

    att = pd.read_csv(OUT / "attributes" / "attributes.csv")
    att["gid"] = att.gauge_id.astype(str).str.zfill(8)
    area = dict(zip(att.gid, att.area_gages2.astype(float)))

    want = [b.strip().zfill(8) for b in (REF / "basins.txt").read_text().split()
            if b.strip()]
    written, skipped = [], []
    for b in want:
        f = SRC / f"{b}.csv.gz"
        if not f.exists():
            skipped.append(b)
            continue
        d = pd.read_csv(f)
        d["date"] = pd.to_datetime(d["date"])
        out = pd.DataFrame({"date": d["date"]})
        for k, v in MAP.items():
            out[k] = d[v].astype("float32")
        a = area.get(b)
        if a and a > 0:
            k = 0.0283168 * 86400 / (a * 1e6) * 1000
            out["q_mm"] = (d["q_cfs"].astype(float) * k).astype("float32")
        else:
            out["q_mm"] = np.float32(np.nan)
        out.set_index("date").to_xarray().to_netcdf(OUT / "time_series" / f"{b}.nc")
        written.append(b)

    (OUT / "basins.txt").write_text("\n".join(written) + "\n")
    print(f"wrote {len(written)} basin files -> {OUT}")
    print(f"skipped (absent from the intensity corpus): {skipped}")

    s = xr.open_dataset(OUT / "time_series" / f"{written[0]}.nc").to_dataframe()
    print(f"\nsample {written[0]}: rows={len(s)} "
          f"{s.index.min().date()}..{s.index.max().date()}")
    print(f"  vars: {list(s.columns)}")
    w = s[s.prcp > 1]
    print(f"  on wet days (n={len(w)}): p_max_1h mean {w.p_max_1h.mean():.3f}, "
          f"p_hours mean {w.p_hours.mean():.1f}, p_cv mean {w.p_cv.mean():.3f}")

    # the SWE lesson: confirm the new variables actually vary
    if w.p_max_1h.std() < 1e-6 or w.p_hours.nunique() < 3:
        sys.exit("FATAL: intensity variables are constant in the netCDF")
    if (s.tmax - s.tmin).mean() < 1:
        sys.exit("FATAL: zero diurnal range")
    print("\n  intensity variables vary as expected")


if __name__ == "__main__":
    main()
