#!/usr/bin/env python3
"""Ledger 41: measure HYSETS' day-stamp convention against CAMELS directly.

The pool lag-scan showed HYSETS/Livneh precipitation peaking one day later than
the CAMELS corpora against discharge, with a two-day smear. That comparison is
confounded: pool basins are non-CAMELS by construction, so product convention
and basin identity move together.

HYSETS contains 644 of the 671 CAMELS basins, which makes a clean test
possible: pull the SAME gauges from HYSETS and correlate their precipitation
against the CAMELS series day by day. Precipitation-versus-precipitation is a
far sharper registration test than precipitation-versus-discharge -- this is
how the Daymet V4R1 offset was found (r 0.07-0.32 at lag 0 against 0.94-0.99 at
lag +1), and that offset was worth +0.229 NSE.

⚠️ These CAMELS basins are read for DIAGNOSIS ONLY. They are excluded from the
pretraining pool and nothing here is written into a training corpus.
"""
import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HY = ROOT / "data" / "hysets"
LAGS = range(-3, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Livneh")
    ap.add_argument("--reference", default="camels_corpus_daymet_v2")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--start", default="1981-01-01")
    ap.add_argument("--end", default="1995-09-30")
    args = ap.parse_args()

    props = pd.read_csv(HY / "HYSETS_watershed_properties.txt")
    props = props[props["Source"].str.upper() == "USGS"]
    wid = {str(r["Official_ID"]).zfill(8): int(r["Watershed_ID"])
           for _, r in props.iterrows()}

    ds = netCDF4.Dataset(HY / f"HYSETS_2023_update_{args.source}.nc", "r")
    t = pd.to_datetime(netCDF4.num2date(
        ds.variables["time"][:], ds.variables["time"].units,
        only_use_cftime_datetimes=False).astype("datetime64[ns]"))
    sel = (t >= pd.Timestamp(args.start)) & (t <= pd.Timestamp(args.end))
    ts = t[sel]

    ref_dir = ROOT / "data" / "local_corpora" / args.reference
    files = [p for p in sorted(ref_dir.glob("*.csv.gz")) if not p.name.startswith("._")]

    rows = []
    for p in files:
        gid = p.name.split(".")[0].zfill(8)
        if gid not in wid:
            continue
        k = wid[gid] - 1
        hy = pd.Series(ds.variables["total_precipitation"][k, :][sel], index=ts)
        cam = pd.read_csv(p, parse_dates=["date"]).set_index("date")["precipitation_sum"]
        cam = cam.reindex(ts)
        ok = hy.notna() & cam.notna()
        if ok.sum() < 1000:
            continue
        a, b = hy[ok].to_numpy(), cam[ok].to_numpy()
        prof = {}
        for L in LAGS:
            s = pd.Series(a).shift(L)
            m = s.notna().to_numpy()
            prof[L] = float(np.corrcoef(s[m], b[m])[0, 1])
        rows.append({"gauge": gid, "peak": max(prof, key=prof.get),
                     "annual_hy": 365.25 * a.mean(), "annual_cam": 365.25 * b.mean(),
                     **{f"lag{L:+d}": prof[L] for L in LAGS}})
        if len(rows) >= args.n:
            break
    ds.close()

    df = pd.DataFrame(rows)
    print(f"{args.source} vs CAMELS {args.reference}, SAME {len(df)} gauges, "
          f"{args.start}..{args.end}\n")
    print("precip-vs-precip correlation by lag (mean over gauges):")
    for L in LAGS:
        c = df[f"lag{L:+d}"].mean()
        mark = "   <-- peak" if L == df[[f"lag{x:+d}" for x in LAGS]].mean().idxmax()[3:] \
            or f"lag{L:+d}" == df[[f'lag{x:+d}' for x in LAGS]].mean().idxmax() else ""
        print(f"  lag {L:+d}: {c:+.4f}{mark}")
    from collections import Counter
    cnt = Counter(df["peak"])
    print(f"\nper-gauge peak lag: {dict(sorted(cnt.items()))}")
    mode, n = cnt.most_common(1)[0]
    print(f"  mode {mode:+d} in {n}/{len(df)} gauges ({100*n/len(df):.1f}%)")
    print(f"\nannual precip mm/yr: {args.source} median {df['annual_hy'].median():.0f}, "
          f"CAMELS median {df['annual_cam'].median():.0f}  "
          f"(ratio {df['annual_hy'].median()/df['annual_cam'].median():.3f})")
    print("\n⚠️ A mean/bias check would call these products equivalent. Only the "
          "lag scan separates them.")
    out = ROOT / "benchmarks" / f"ledger41_registration_{args.source}.json"
    out.write_text(json.dumps({
        "source": args.source, "reference": args.reference, "n_gauges": len(df),
        "mean_corr_by_lag": {f"{L:+d}": float(df[f'lag{L:+d}'].mean()) for L in LAGS},
        "peak_lag_mode": int(mode), "peak_lag_share": float(n / len(df)),
        "peak_lag_distribution": {str(k): int(v) for k, v in sorted(cnt.items())},
        "annual_ratio": float(df['annual_hy'].median() / df['annual_cam'].median()),
    }, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
