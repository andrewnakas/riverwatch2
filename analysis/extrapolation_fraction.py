#!/usr/bin/env python3
"""What fraction of CAMELS-531 peaks required RATING-CURVE EXTRAPOLATION?

THE QUESTION
------------
A daily discharge value is only as good as the rating curve that produced it.
Where the observed peak exceeds the largest DIRECT field measurement ever made
at that gauge, the reported flow is an EXTRAPOLATION -- and Kiang 2018 measured
41-200% uncertainty in that regime versus 3-17% where the curve is interpolated.

Coxon 2015 found 44% of UK high-flow gauges unassessable for exactly this
reason. No one has published the CAMELS-531 equivalent. It decides whether the
~104 flashy basins' apparent headroom is real or an artifact
(headroom-where-effort-pays).

    extrapolated_i  :=  q_max_observed_i  >  q_max_field_measured_i

Also reports the ratio q_max / q_meas_max, which says HOW FAR beyond the
measured range the peak sits -- the severity, not just the incidence.
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", default="basin_peaks.csv")
    ap.add_argument("--meas", default="field_measurements.csv")
    a = ap.parse_args()

    p = pd.read_csv(a.peaks, dtype={"station_id": str})
    m = pd.read_csv(a.meas, dtype={"station_id": str})
    p["station_id"] = p.station_id.str.zfill(8)
    m["station_id"] = m.station_id.str.zfill(8)

    df = p.merge(m, on="station_id", how="inner")
    print(f"joined {len(df)} basins "
          f"(peaks {len(p)}, measurements {len(m)})")

    miss = df.q_meas_max.isna().sum()
    if miss:
        print(f"  {miss} basins have NO field measurements -- excluded")
    df = df.dropna(subset=["q_meas_max"])
    df = df[df.q_meas_max > 0]
    print(f"  {len(df)} basins usable\n")

    df["extrap"] = df.q_max > df.q_meas_max
    df["ratio"] = df.q_max / df.q_meas_max

    n, k = len(df), int(df.extrap.sum())
    print("=" * 62)
    print(f"  PEAKS BEYOND THE HIGHEST DIRECT MEASUREMENT: "
          f"{k}/{n} = {100*k/n:.1f}%")
    print("=" * 62)
    print("  (Coxon 2015 could not assess high-flow uncertainty for 44% of")
    print("   500 UK gauges for this reason. This is the CAMELS-531 number.)\n")

    print(f"{'cohort':<10}{'n':>5}{'extrapolated':>14}{'med ratio':>11}"
          f"{'p90 ratio':>11}")
    print("-" * 51)
    for c in ("steady", "flashy"):
        s = df[df.cohort == c]
        if not len(s):
            continue
        print(f"{c:<10}{len(s):>5}{100*s.extrap.mean():>13.1f}%"
              f"{s.ratio.median():>11.2f}{s.ratio.quantile(0.9):>11.2f}")
    print(f"{'ALL':<10}{n:>5}{100*k/n:>13.1f}%"
          f"{df.ratio.median():>11.2f}{df.ratio.quantile(0.9):>11.2f}")

    # severity among the extrapolated ones
    ex = df[df.extrap]
    if len(ex):
        print(f"\nAmong the {len(ex)} extrapolated basins, the peak sits")
        print(f"  {ex.ratio.median():.2f}x (median) and "
              f"{ex.ratio.quantile(0.9):.2f}x (p90) above the highest")
        print("  direct measurement.")

    # restrict to good/excellent-rated measurements: a stricter, fairer bar
    if "q_meas_max_good" in df.columns:
        g = df.dropna(subset=["q_meas_max_good"])
        g = g[g.q_meas_max_good > 0]
        if len(g):
            kg = int((g.q_max > g.q_meas_max_good).sum())
            print(f"\nUsing only GOOD/EXCELLENT-rated measurements as the bar:")
            print(f"  {kg}/{len(g)} = {100*kg/len(g):.1f}% of peaks are beyond")
            print("  the highest well-rated gauging. High flows are measured")
            print("  rarely AND badly -- the two compound.")

    print("\nINTERPRETATION")
    print("  A high fraction means our 'literature' ceiling scenario is too")
    print("  OPTIMISTIC for those basins -- their true sigma is nearer Kiang's")
    print("  41-200% than Coxon's 13%, their ceiling is lower, and part of the")
    print("  apparent headroom is not recoverable by any model.")
    print("  A low fraction means the headroom is real and worth chasing.")


if __name__ == "__main__":
    main()
