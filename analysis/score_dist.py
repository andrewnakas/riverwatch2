#!/usr/bin/env python3
"""Score a DISTRIBUTIONAL member (CMAL/GMM) against its point-model control.

The distributional A/B is not decided by overall NSE -- it is decided by
PEAK-DAY error, because 96.5% of with-q error and 88.4% of the worst no-q
basins' error is on top-10% flow days. This reports all of it:

  1. day-1 median NSE            (comparable to every other member we have)
  2. PEAK-DAY squared-error share and peak-day NSE  <-- the targeted quantity
  3. MEDIAN vs MEAN readout      (ymed vs ymean) -- my "mean is biased low"
                                 claim was WRONG; the mean of a right-skewed
                                 predictive dist is HIGHER. Measure, don't assume.
  4. Interval coverage           (ylo..yhi) overall and on peak days -- only
                                 meaningful for a distributional run, and the
                                 direct evidence for the peak-uncertainty story.

Usage:
  score_dist.py --dist gpu1080/dumps/camels531ls_cmal_nhlstm_s111.csv.gz \
                --ctrl gpu1080/dumps/camels531ls_multi_nhlstm_s111.csv.gz
"""
import argparse

import numpy as np
import pandas as pd

COLS = ["station_id", "t0", "h", "truth", "ylo", "ymed", "yhi", "ymean"]


def load(path):
    d = pd.read_csv(path, usecols=lambda c: c in COLS)
    d = d[d.h == 1].copy()
    return d.dropna(subset=["truth", "ymed"])


def nse(truth, pred):
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    ok = np.isfinite(truth) & np.isfinite(pred)
    if ok.sum() < 2:
        return np.nan
    truth, pred = truth[ok], pred[ok]
    den = ((truth - truth.mean()) ** 2).sum()
    if den <= 0:
        return np.nan
    return 1.0 - ((pred - truth) ** 2).sum() / den


def per_basin(df, col):
    return df.groupby("station_id").apply(
        lambda g: nse(g["truth"], g[col]), include_groups=False)


def peak_mask(g, q=0.90):
    """Top-10% flow days, defined PER BASIN on observed flow."""
    return g["truth"] >= g["truth"].quantile(q)


def report(name, d):
    print(f"\n=== {name} ===")
    print(f"  rows {len(d):,}  basins {d.station_id.nunique()}")

    for col in ("ymed", "ymean"):
        if col in d.columns and d[col].notna().any():
            s = per_basin(d, col)
            print(f"  day-1 median NSE [{col:<5}] = {s.median():.4f}")

    # peak-day error share + peak-day NSE
    shares, pk_nse = [], []
    for _, g in d.groupby("station_id"):
        m = peak_mask(g)
        if m.sum() < 5:
            continue
        se = (g["ymed"] - g["truth"]) ** 2
        tot = se.sum()
        if tot > 0:
            shares.append(se[m].sum() / tot)
        pk_nse.append(nse(g.loc[m, "truth"], g.loc[m, "ymed"]))
    if shares:
        print(f"  peak-day (top-10%) SHARE of squared error = "
              f"{np.nanmedian(shares):.3f}")
    if pk_nse:
        print(f"  peak-day NSE (median over basins)         = "
              f"{np.nanmedian(pk_nse):.4f}")

    # interval coverage -- distributional runs only
    if {"ylo", "yhi"} <= set(d.columns) and (d.yhi > d.ylo).any():
        cov = ((d.truth >= d.ylo) & (d.truth <= d.yhi))
        print(f"  80% interval coverage: overall {cov.mean():.3f} "
              f"(nominal 0.800)")
        pk = d.groupby("station_id", group_keys=False).apply(
            peak_mask, include_groups=False)
        if pk.any():
            print(f"  80% interval coverage: PEAK days {cov[pk.values].mean():.3f}"
                  f"   (vs nominal 0.800; UNDER = over-confident peaks, OVER = intervals too wide)")
        width = (d.yhi - d.ylo) / d.truth.replace(0, np.nan)
        print(f"  median relative interval width = {width.median():.3f}")
    else:
        print("  (point model: ylo==ymed==yhi, no interval to score)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--ctrl", required=True)
    a = ap.parse_args()

    d, c = load(a.dist), load(a.ctrl)
    report("DISTRIBUTIONAL", d)
    report("CONTROL (point)", c)

    common = sorted(set(d.station_id) & set(c.station_id))
    print(f"\n=== HEAD-TO-HEAD on {len(common)} shared basins ===")
    dd = d[d.station_id.isin(common)]
    cc = c[c.station_id.isin(common)]
    sd, sc = per_basin(dd, "ymed"), per_basin(cc, "ymed")
    common_idx = sd.index.intersection(sc.index)
    sd, sc = sd.loc[common_idx], sc.loc[common_idx]
    delta = sd.median() - sc.median()
    print(f"  dist {sd.median():.4f}  ctrl {sc.median():.4f}  "
          f"DELTA {delta:+.4f}")
    win = (sd > sc).sum()
    print(f"  basins improved: {win}/{len(sd)} ({100*win/len(sd):.1f}%)")
    print("\n  NOTE: member NSE does NOT decide this. Judge on the seed-matched"
          "\n  ENSEMBLE delta -- member skill and ensemble value have repeatedly"
          "\n  pointed in OPPOSITE directions in this campaign.")


if __name__ == "__main__":
    main()
