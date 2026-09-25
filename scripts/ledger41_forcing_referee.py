#!/usr/bin/env python3
"""Ledger 41: which precipitation product carries more STREAMFLOW signal --
CAMELS' own daymet, or the HYSETS gauge composites the prior was pretrained on?

⛔ SCOPE -- READ BEFORE QUOTING THIS. This answers the SINGLE-FORCING question
only, which is the only arm ledger 41 runs. It does NOT license skipping a
Daymet re-extraction for a MULTI-forcing prior. The 15-input streams
(lstm_multi / multi5 / multi6, 62% of ensemble weight) carry their value in the
DISAGREEMENT BETWEEN slots, not in any slot's standalone quality -- this project
measured the three CAMELS forcings sharing ~70% of their error and disagreeing
27% more on event days. A prior whose slots are SCDNA/Livneh/QC-stations teaches
a different disagreement structure than the finetune inherits, and this screen
cannot see that. The settling test for the multi case is the inter-product
CORRELATION STRUCTURE of {SCDNA, Livneh, QC_stations} against
{daymet, nldas, maurer} on shared gauges -- not run here.

The prior is pretrained on HYSETS (SCDNA precip, Livneh temp) and evaluated on
CAMELS daymet. That substitution is defensible -- SCDNA reproduces CAMELS daymet
at r 0.902, std ratio 1.013, lag 0 in 98% -- but "reproduces the series" is not
"carries the same signal about discharge". Correlation passed both HYSETS
forcing defects before, so it is not trusted alone here.

THE REFEREE IS STREAMFLOW, never another forcing. Forcing-to-forcing agreement
was measured to be an invalid registration criterion (the Maurer realignment
selector ran a 63% false-positive rate against it).

Design, matching the screen that closed the Daymet V4 upgrade:

  * same gauges for every product -- HYSETS holds 644 of the 671 CAMELS basins,
    so product and basin identity do NOT move together
  * per basin, corr(precip(t), dQ(t+k)) for k in -3..+3, dQ = q.diff()
  * take the SIGN-AWARE peak over k (max: rain raises discharge). An abs()
    peak once invented a fusion bug that did not exist, because post-storm
    recession makes dQ strongly negative at k=+2.
  * peak-over-lag makes the comparison REGISTRATION-INVARIANT: each product is
    allowed its own best day-stamp, so this measures information content and
    cannot be won or lost on a one-day offset.

⚠️ CAMELS basins are read for DIAGNOSIS ONLY -- none is in the pretraining pool.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HY = ROOT / "data" / "hysets"
LAGS = range(-3, 4)
SENTINEL = -100.0          # q_cfs carries -999999 fills; they poisoned a corpus once


def peak_over_lag(precip, dq):
    """Sign-aware peak correlation of precip(t) against dQ(t+k)."""
    best_k, best_r = None, -np.inf
    for k in LAGS:
        s = pd.Series(precip).shift(k)
        m = s.notna().to_numpy() & np.isfinite(dq)
        if m.sum() < 500:
            continue
        a, b = s.to_numpy()[m], dq[m]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best_r:                      # sign-aware: rain RAISES discharge
            best_k, best_r = k, r
    return best_k, best_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="camels_corpus_daymet_v2")
    ap.add_argument("--start", default="1981-01-01")
    ap.add_argument("--end", default="1995-09-30")
    ap.add_argument("--limit", type=int, default=0, help="0 = all shared gauges")
    ap.add_argument("--out", default="benchmarks/ledger41_forcing_referee_SINGLE_FORCING.json")
    args = ap.parse_args()

    props = pd.read_csv(HY / "HYSETS_watershed_properties.txt")
    props = props[props["Source"].str.upper() == "USGS"]
    wid = {str(r["Official_ID"]).zfill(8): int(r["Watershed_ID"])
           for _, r in props.iterrows()}

    sources = ["SCDNA", "Livneh"]
    ds, tsel = {}, {}
    for s in sources:
        d = netCDF4.Dataset(HY / f"HYSETS_2023_update_{s}.nc", "r")
        t = pd.to_datetime(netCDF4.num2date(
            d.variables["time"][:], d.variables["time"].units,
            only_use_cftime_datetimes=False).astype("datetime64[ns]"))
        sel = (t >= pd.Timestamp(args.start)) & (t <= pd.Timestamp(args.end))
        ds[s], tsel[s] = d, (sel, t[sel])

    ref_dir = ROOT / "data" / "local_corpora" / args.reference
    files = [p for p in sorted(ref_dir.glob("*.csv.gz")) if not p.name.startswith("._")]

    rows = []
    for p in files:
        gid = p.name.split(".")[0].zfill(8)
        if gid not in wid:
            continue
        cam = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        q = cam["q_cfs"].copy()
        q[q <= SENTINEL] = np.nan                      # drop -999999 fills
        idx = tsel[sources[0]][1]                      # common daily axis
        q = q.reindex(idx)
        dq = q.diff().to_numpy(dtype=float)
        if np.isfinite(dq).sum() < 1000:
            continue

        cand = {"camels_daymet": cam["precipitation_sum"].reindex(idx).to_numpy(dtype=float)}
        for s in sources:
            k = wid[gid] - 1
            sel, _ = tsel[s]
            cand[s] = np.asarray(ds[s].variables["total_precipitation"][k, :][sel],
                                 dtype=float)

        rec = {"gauge": gid}
        ok = True
        for name, pr in cand.items():
            kk, rr = peak_over_lag(pr, dq)
            if kk is None:
                ok = False
                break
            rec[f"{name}_peak"] = rr
            rec[f"{name}_lag"] = kk
        if ok:
            rows.append(rec)
        if args.limit and len(rows) >= args.limit:
            break
    for d in ds.values():
        d.close()

    df = pd.DataFrame(rows)
    names = ["camels_daymet"] + sources
    print(f"STREAMFLOW REFEREE -- {len(df)} gauges shared by HYSETS and CAMELS, "
          f"{args.start}..{args.end}")
    print("peak-over-lag corr(precip, dQ), registration-invariant, sign-aware\n")
    print(f"  {'product':16s} {'median':>8s} {'mean':>8s} {'peak-lag mode':>16s}")
    for n in names:
        c = Counter(df[f"{n}_lag"])
        mode, cnt = c.most_common(1)[0]
        print(f"  {n:16s} {df[f'{n}_peak'].median():8.4f} "
              f"{df[f'{n}_peak'].mean():8.4f} {f'{mode:+d} ({100*cnt/len(df):.0f}%)':>16s}")

    print("\npaired against CAMELS daymet (the product the model is scored on):")
    res = {}
    for n in sources:
        d = df[f"{n}_peak"] - df["camels_daymet_peak"]
        wins = int((d > 0).sum())
        print(f"  {n:16s} paired median {d.median():+.4f}   "
              f"wins {wins}/{len(df)} ({100*wins/len(df):.1f}%)")
        res[n] = {"paired_median": float(d.median()),
                  "wins": wins, "n": int(len(df)),
                  "win_share": float(wins / len(df))}

    out = ROOT / args.out
    out.write_text(json.dumps({
        "scope": ("SINGLE-FORCING arm only. Does NOT license skipping a Daymet "
                  "re-extraction for a multi-forcing prior, whose value is the "
                  "disagreement BETWEEN slots, not any slot's standalone quality."),
        "reference": args.reference, "window": [args.start, args.end],
        "n_gauges": int(len(df)),
        "median_peak": {n: float(df[f"{n}_peak"].median()) for n in names},
        "peak_lag_mode": {n: int(Counter(df[f"{n}_lag"]).most_common(1)[0][0]) for n in names},
        "paired_vs_camels_daymet": res,
    }, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
