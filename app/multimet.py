"""Caravan MultiMet archived-forecast loader (Shalev & Kratzert 2024).

Reads the HRES forecast zarr (Zenodo 14161281, `HRES/timeseries.zarr`) and
maps it into the MB-LSTM decoder-forcing schema so the shipped ensemble can be
scored on the SAME archived IFS-HRES forecasts AIFL and Google's baselines
used — an apples-to-apples number on shared infrastructure, and the honest fix
for the AIFL 2021-24 evaluation window (our own archives are 2025/2026).

Store layout (verified 2026-07-06):
  dims   basin[22492], date[3196], lead_time[10]
  vars   hres_temperature_2m, hres_total_precipitation, hres_surface_pressure,
         hres_surface_net_solar_radiation, hres_surface_net_thermal_radiation
         each shape (basin, date, lead_time)
  basin  Caravan ids; the US CAMELS subset is 'camels_<usgs8>' (671 basins)
  date   'days since 2012-01-01' (proleptic gregorian); an entry D is the
         forecast ISSUE date, lead_time L lands on D + L days
  units  MultiMet ships these already Caravan-normalized: temperature_2m in
         degrees C, total_precipitation in mm/day (verified 2026-07-06 against
         raw cells — NOT the IFS-native K / m; no de-scaling needed).

Variable gap vs our 5-var compat decoder: MultiMet ships temperature_2m only
(no tmax/tmin) and net-solar-radiation (not shortwave-down MJ). We map
temperature_2m -> mean/max/min (all three equal — the daily-mean is the only
signal HRES gives here). Radiation is left NaN in the compat slot:
shortwave_radiation_sum has no direct HRES analogue, and the shipped model
tolerates a single missing decoder channel far better than a units-mismatched
one. This is documented as a known limitation of the MultiMet comparison, not
hidden.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STORE = Path(
    "/Volumes/STORAGE_SD/riverwatch2_data/external/caravan_multimet/HRES/timeseries.zarr")
EPOCH = date(2012, 1, 1)
COMPAT_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum",
]


@lru_cache(maxsize=4)
def _open(store: str):
    import zarr
    z = zarr.open(store, mode="r")
    basins = np.asarray([str(b) for b in z["basin"][:]])
    dates = np.asarray(z["date"][:], dtype=np.int64)
    leads = np.asarray(z["lead_time"][:], dtype=np.int64)
    return z, basins, dates, leads


def us_gauge_ids(store: Path = DEFAULT_STORE) -> list[str]:
    """USGS ids (8-digit) of the CAMELS-US basins present in the store."""
    _, basins, _, _ = _open(str(store))
    return sorted(b.split("_", 1)[1] for b in basins if b.startswith("camels_"))


def issue_dates(store: Path = DEFAULT_STORE) -> list[date]:
    _, _, dates, _ = _open(str(store))
    return [EPOCH + timedelta(days=int(d)) for d in dates]


def forcing_window(usgs_id: str, t0: date, horizon: int = 10,
                   store: Path = DEFAULT_STORE) -> pd.DataFrame | None:
    """Decoder forcing for one (gauge, issue-date) window in the compat schema.

    Returns a (horizon, 6) frame [date, *COMPAT_VARS] with the HRES forecast
    issued on t0, or None if the gauge/date is absent. shortwave_radiation_sum
    is all-NaN (no HRES analogue — see module docstring). horizon is capped at
    the store's 10 leads."""
    z, basins, dates, leads = _open(str(store))
    key = f"camels_{usgs_id}"
    bi = np.flatnonzero(basins == key)
    if not len(bi):
        return None
    di = np.flatnonzero(dates == (t0 - EPOCH).days)
    if not len(di):
        return None
    n = min(horizon, len(leads))
    b, d = int(bi[0]), int(di[0])
    temp_c = np.asarray(z["hres_temperature_2m"][b, d, :n], dtype=float)   # already C
    precip_mm = np.asarray(z["hres_total_precipitation"][b, d, :n], dtype=float)  # already mm
    out = pd.DataFrame({
        "date": [pd.Timestamp(t0) + pd.Timedelta(days=int(leads[i])) for i in range(n)],
        "temperature_2m_mean": temp_c,
        "temperature_2m_max": temp_c,      # HRES daily-mean only; no tmax/tmin
        "temperature_2m_min": temp_c,
        "precipitation_sum": precip_mm,
        "shortwave_radiation_sum": np.full(n, np.nan),  # no HRES analogue
    })
    return out
