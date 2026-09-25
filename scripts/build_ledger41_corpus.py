#!/usr/bin/env python3
"""Ledger 41: build the pretraining corpus from HYSETS.

Emits one csv.gz per pool basin in the SAME schema the CAMELS corpora use, so
`corpus_to_nh.py`'s conventions (unit handling, the q_cfs -> q_mm area
normalisation, the 5 dynamic inputs) apply unchanged. That is deliberate: the
whole point of ledger 41 is that the prior and the finetune target live in the
same input space, and the cheapest way to guarantee that is to run the same
code rather than a parallel implementation that agrees today.

Precipitation and temperature come from a GAUGE-BASED source (Livneh or SCDNA).
Radiation and vapour pressure have no gauge-based source in HYSETS and are taken
from ERA5-Land -- acceptable because the screen that killed ledger 40's prior is
specifically about PRECIPITATION being model output; radiation from a reanalysis
is not the same hazard. The precip column never comes from ERA5-Land, and the
script refuses to let it.

Vapour pressure is derived from the 2 m dewpoint by the Magnus formula, in the
kPa units the CAMELS corpora use.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
HY = ROOT / "data" / "hysets"
POOL = ROOT / "data" / "ledger41_basins.json"
OUT = ROOT / "data" / "local_corpora" / "gages2_hysets_v1"

GAUGE_SOURCES = ("Livneh", "SCDNA")     # precip may ONLY come from these
M3S_TO_CFS = 35.3146667


def open_source(name):
    p = HY / f"HYSETS_2023_update_{name}.nc"
    if not p.exists():
        sys.exit(f"missing {p}")
    return xr.open_dataset(p, decode_timedelta=False)


def pick(ds, *candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    raise KeyError(f"none of {candidates} in {list(ds.variables)}")


def vapour_pressure_kpa(dewpoint_c):
    """Magnus: saturation vapour pressure at the dewpoint IS the actual vp."""
    return 0.6108 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))


def derive_vp_pa(tmin_c):
    """CAMELS daymet vapour pressure, reproduced from tmin.

    MEASURED, not assumed: on 60 CAMELS basins the published `vapor_pressure`
    equals the saturation vapour pressure at tmin (MTCLIM's humid-day
    assumption, which is how Daymet generates the variable) at r = 0.99924
    (worst basin 0.99713), with a units factor of 998.3 -- i.e. CAMELS stores it
    in Pa. Deriving it reproduces the CONVENTION rather than substituting a
    different product's humidity.
    """
    return 1000.0 * 0.6108 * np.exp(17.27 * tmin_c / (tmin_c + 237.3))


def derive_srad_mj(tmax_c, tmin_c, doy, lat_deg, kr=0.14794):
    """Shortwave radiation from the diurnal temperature range (Hargreaves).

    Daymet's srad is itself derived from temperature via MTCLIM, so a
    temperature-based estimate is convention-aligned by construction.

    kr is CALIBRATED, not the textbook 0.16: applying this function to CAMELS'
    own temperature over 150 basins and comparing against the published srad
    showed the textbook value running +8.2% high, so kr = 0.16 x (1/1.0815) =
    0.14794. Same move as the frac_snow threshold and the PET scale -- recover
    the convention from the published values rather than trusting the
    literature constant. Residual quality: r = 0.928, std ratio 0.941.
    Bristow-Campbell matched the correlation with a worse std ratio (1.187).
    """
    lat = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.25)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.25 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws))
    return kr * np.sqrt(np.maximum(tmax_c - tmin_c, 0)) * ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precip-source", default="SCDNA", choices=list(GAUGE_SOURCES),
                    help="MEASURED on 60 shared CAMELS gauges: SCDNA precip has "
                         "std ratio 1.013 and peaks at lag 0 in 98%% of gauges; "
                         "Livneh precip has std ratio 0.814 and peaks at lag +1 "
                         "in 88%% (shifted AND smeared).")
    ap.add_argument("--temp-source", default="Livneh", choices=list(GAUGE_SOURCES),
                    help="MEASURED on the same gauges: Livneh tmax/tmin have std "
                         "ratio 1.008/1.001 at r 0.995 and lag 0 in 100%%; SCDNA "
                         "tmax AND tmin are compressed to std ratio 0.580 -- while "
                         "still correlating at r 0.987, so a correlation check "
                         "passes them and only the distribution check catches it.")
    ap.add_argument("--start", default="1979-01-01",
                    help="corpus start; needs >= seq_length before the first "
                         "training target (365 d)")
    ap.add_argument("--end", default="1995-09-30",
                    help="corpus end; the Li/Song TRAIN window closes 1995-09-30 "
                         "and the prior must not see the scored period")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-radiation", action="store_true",
                    help="skip srad/vp (3-input corpus; NOT the production interface)")
    ap.add_argument("--radiation", default="derive",
                    choices=["derive", "era5land"],
                    help="'derive' builds srad/vp from temperature the way Daymet "
                         "does (vp r=0.999, srad r=0.926 vs CAMELS); 'era5land' "
                         "takes them from the HYSETS ERA5-Land file. Chosen by "
                         "measurement, not preference.")
    ap.add_argument("--shift-days", type=int, default=1,
                    help="days to delay the FORCING columns so they land on the "
                         "CAMELS day-stamp convention. Default 1, MEASURED not "
                         "assumed: on 80 shared gauges HYSETS precip peaks against "
                         "CAMELS precip at lag +1 (r 0.752 vs 0.685 at lag 0) in "
                         "78.8%% of gauges, while HYSETS discharge matches CAMELS "
                         "at r=1.0000 lag 0 in 100%% -- so the offset is in the "
                         "forcings alone. See scripts/ledger41_registration_probe.py. "
                         "Pass 0 to disable and re-measure.")
    args = ap.parse_args()

    for s_ in (args.precip_source, args.temp_source):
        if s_ not in GAUGE_SOURCES:
            sys.exit(f"{s_} is not a gauge-based source")

    pool = json.load(open(POOL))["basins"]
    if args.limit:
        pool = pool[:args.limit]
    print(f"pool: {len(pool)} basins | precip {args.precip_source} | "
          f"temp {args.temp_source}")

    # Neither gauge-based source is usable on its own: each is correct for one
    # of the two variable groups and defective for the other, measured against
    # the same CAMELS gauges. Taking each channel from the source that
    # reproduces CAMELS is the only construction the measurements support.
    met = open_source(args.precip_source)
    tmp = met if args.temp_source == args.precip_source else open_source(args.temp_source)
    v_pr = pick(met, "total_precipitation", "pr", "precip")
    v_tx = pick(tmp, "2m_tasmax", "tasmax", "tmax")
    v_tn = pick(tmp, "2m_tasmin", "tasmin", "tmin")
    v_q = pick(met, "discharge", "q", "streamflow")
    if tmp is not met:
        assert np.array_equal(met["time"].values, tmp["time"].values), (
            "the two HYSETS sources do not share a time axis; mixing channels "
            "across them would misalign every day")
        assert met.sizes[[d for d in met[v_pr].dims if d != "time"][0]] == \
               tmp.sizes[[d for d in tmp[v_tx].dims if d != "time"][0]], (
            "the two HYSETS sources do not share a watershed axis")

    rad = None
    if not args.no_radiation and args.radiation == "era5land":
        rad = open_source("ERA5Land")
        print(f"  ERA5Land vars: {sorted(rad.data_vars)}")
        v_sr = pick(rad, "surface_solar_radiation_downwards", "ssrd", "srad", "rsds")
        v_dp = pick(rad, "2m_dewpoint_temperature", "d2m", "dewpoint", "tdew")

    # HYSETS indexes basins along a 'watershed' dimension; Watershed_ID is 1-based
    wdim = [d for d in met[v_pr].dims if d != "time"][0]
    nwat = met.sizes[wdim]
    if "watershedID" in met.variables:
        wids = met["watershedID"].values
        assert np.array_equal(wids, np.arange(1, len(wids) + 1)), (
            "watershedID is not monotonic 1..N -- the Watershed_ID-1 row index "
            "assumption does not hold and every basin would silently get another "
            "basin's forcings")
    time = pd.to_datetime(met["time"].values)
    sel = (time >= pd.Timestamp(args.start)) & (time <= pd.Timestamp(args.end))
    t = time[sel]
    print(f"  time axis: {time[0].date()}..{time[-1].date()}  "
          f"-> window {t[0].date()}..{t[-1].date()} ({sel.sum()} days)")

    # Read each variable's time window ONCE for every watershed rather than
    # slicing per basin. These files are chunked along time, so a per-basin read
    # decompresses the whole record to keep 6k days of it -- measured at ~10
    # basins/min, i.e. 3+ hours for the pool. The window is contiguous, and
    # 14,425 x ~6,100 float32 is ~350 MB per variable, so it simply fits.
    i0, i1 = int(np.argmax(sel)), int(len(sel) - np.argmax(sel[::-1]))
    print(f"  loading {i1 - i0} days x {nwat} watersheds per variable...", flush=True)
    PR = met[v_pr].values[:, i0:i1]
    TX = tmp[v_tx].values[:, i0:i1]
    TN = tmp[v_tn].values[:, i0:i1]
    Q = met[v_q].values[:, i0:i1]
    SR = DP = None
    if rad is not None:
        SR = rad[v_sr].values[:, i0:i1]
        DP = rad[v_dp].values[:, i0:i1]
    print(f"  loaded", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written, skipped = [], {}
    for i, b in enumerate(pool, 1):
        wid = int(b["hysets_watershed_id"])
        if not 1 <= wid <= nwat:
            skipped[b["gauge_id"]] = "watershed_id_out_of_range"
            continue
        k = wid - 1
        pr = PR[k].astype("float64")
        tx = TX[k].astype("float64")
        tn = TN[k].astype("float64")
        q = Q[k].astype("float64")

        row = {"date": t,
               "q_cfs": q * M3S_TO_CFS,
               "precipitation_sum": pr,
               "temperature_2m_max": tx,
               "temperature_2m_min": tn,
               "temperature_2m_mean": (tx + tn) / 2.0}
        if not args.no_radiation and args.radiation == "derive":
            doy = pd.DatetimeIndex(t).dayofyear.to_numpy()
            row["shortwave_radiation_sum"] = derive_srad_mj(tx, tn, doy, b["lat"])
            row["vapor_pressure"] = derive_vp_pa(tn)
        elif rad is not None:
            sr = SR[k].astype("float64")
            dp = DP[k].astype("float64")
            if np.nanmax(np.abs(sr)) > 1000:      # J/m2 accumulations -> MJ/m2/d
                sr = sr / 1e6
            if np.nanmedian(dp) > 100:            # kelvin -> celsius
                dp = dp - 273.15
            row["shortwave_radiation_sum"] = sr
            row["vapor_pressure"] = vapour_pressure_kpa(dp)

        df = pd.DataFrame(row)
        # Land the forcings on the CAMELS day-stamp convention. Discharge is NOT
        # shifted: it already reproduces CAMELS exactly, and shifting it would
        # break the one alignment that is known to be right.
        if args.shift_days:
            fcols = [c for c in df.columns if c not in ("date", "q_cfs")]
            df[fcols] = df[fcols].shift(args.shift_days)
            df = df.iloc[args.shift_days:].reset_index(drop=True)
        # A NaN in a DYNAMIC input poisons the member and must never be
        # zero-filled; drop the basin instead. NaN in q is fine -- NH masks it.
        dyn = [c for c in df.columns if c not in ("date", "q_cfs")]
        nan_frac = float(df[dyn].isna().to_numpy().mean())
        if nan_frac > 0:
            bad = {c: int(df[c].isna().sum()) for c in dyn if df[c].isna().any()}
            skipped[b["gauge_id"]] = f"nan_in_dynamics {bad}"
            continue
        if float(df["q_cfs"].notna().mean()) < 0.5:
            skipped[b["gauge_id"]] = "q_coverage_below_50pct"
            continue
        df.to_csv(out / f"{b['gauge_id']}.csv.gz", index=False,
                  compression="gzip", float_format="%.5f")
        written.append(b["gauge_id"])
        if i % 200 == 0:
            print(f"  [{i}/{len(pool)}] wrote {len(written)}", flush=True)

    print(f"\nwrote {len(written)} basins -> {out}")
    if skipped:
        from collections import Counter
        print(f"skipped {len(skipped)}:")
        for reason, n in Counter(s.split(" ")[0] for s in skipped.values()).most_common():
            print(f"  {reason:28s} {n}")
    manifest = out.parent / f"{out.name}_manifest.json"
    manifest.write_text(json.dumps(
        {"precip_source": args.precip_source, "temp_source": args.temp_source,
         "shift_days": args.shift_days,
         "radiation": None if args.no_radiation else "ERA5Land",
         "start": args.start, "end": args.end,
         "n_written": len(written), "written": written, "skipped": skipped}, indent=1))
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
