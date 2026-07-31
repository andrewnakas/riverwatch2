#!/usr/bin/env python3
"""A DEPLOYABLE with-q number to replace the oracle 0.9016.

The published with-q record (day-1 median NSE 0.9016 vs Nearing 2022's 0.879)
combined the three per-forcing members with `--fit-weights`, i.e. weights
[0.411, 0.263, 0.326] optimized against TEST observations. By the standard this
project applies elsewhere that is an ORACLE, and a reviewer would reject it.

This computes what can be claimed without touching test observations:

  1. plain mean of the 3 with-q forcing members  -- zero fitted parameters,
     the same rule Li/Song use for their headline. Fully deployable.
  2. 4-seed variants where available (deeper seed averaging is not fitting).
  3. the oracle fit-weights number, reproduced and clearly labelled, so the
     gap between "what we can claim" and "what weighting could buy" is explicit.

If the plain mean still clears 0.879, the record beat survives without any
oracle component -- which is the version that can go in a paper.

Metric matches the campaign: day-1 (h==1) median per-basin NSE, >=20 samples
and var>1e-9 per basin, raw/unclipped, 531 basins.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/mblstm/gpu_dumps_s14")
RECORD = 0.879          # Nearing et al. 2022 (HESS 26:5493), AR-LSTM
PUBLISHED_ORACLE = 0.9016
FORCINGS = ("daymet", "maurer", "nldas")


def load(tag, forcing):
    """tag: '' for the 2-seed members, '4' for the 4-seed ones."""
    p = D / f"camels531_{forcing}_withq{tag}_full531.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["station_id", "t0", "h", "truth", "ymed"])
    df = df[df.h == 1].copy()
    df["station_id"] = df["station_id"].astype(str).str.zfill(8)
    return df[["station_id", "t0", "truth", "ymed"]].rename(columns={"ymed": forcing})


def merge(tag):
    out = None
    for f in FORCINGS:
        d = load(tag, f)
        if d is None:
            return None
        out = (d if out is None else
               out.merge(d[["station_id", "t0", f]], on=["station_id", "t0"]))
    return out


def med_nse(df, pred):
    nses = []
    for _, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = pred[g.index] if isinstance(pred, np.ndarray) else g[pred].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        nses.append(1 - np.mean((y - p) ** 2) / np.var(y))
    return float(np.median(nses)), len(nses)


def report(tag, label):
    m = merge(tag)
    if m is None:
        print(f"\n{label}: dumps missing, skipped")
        return
    m = m.reset_index(drop=True)
    print(f"\n=== {label} ===")
    print(f"  rows={len(m)} basins={m.station_id.nunique()}")

    for f in FORCINGS:
        s, _ = med_nse(m, f)
        print(f"  member {f:8s} day-1 medNSE = {s:.4f}")

    X = m[list(FORCINGS)].to_numpy(float)
    plain, n = med_nse(m, X.mean(1))
    print(f"  [DEPLOYABLE] plain mean       = {plain:.4f}  (basins={n})")
    print(f"               vs Nearing 0.879 : {plain - RECORD:+.4f} "
          f"{'BEATS' if plain > RECORD else 'below'}")

    # best 2-of-3 subset, still reported as oracle since chosen on test here
    best = None
    for k in (2,):
        for sub in itertools.combinations(FORCINGS, k):
            s, _ = med_nse(m, m[list(sub)].to_numpy(float).mean(1))
            if best is None or s > best[0]:
                best = (s, sub)
    print(f"  [oracle] best 2-member subset = {best[0]:.4f}  {best[1]}")

    try:
        from scipy.optimize import nnls
        y = m["truth"].to_numpy(float)
        w, _ = nnls(X, y)
        w = w / w.sum() if w.sum() > 0 else np.ones(3) / 3
        s, _ = med_nse(m, X @ w)
        print(f"  [oracle] NNLS fit-weights     = {s:.4f}  w={np.round(w,3).tolist()}")
    except Exception as e:
        print(f"  NNLS skipped: {e}")


def main():
    print("Published claim: day-1 median NSE %.4f via --fit-weights "
          "[0.411,0.263,0.326]" % PUBLISHED_ORACLE)
    print("Those weights were fit on TEST observations -> ORACLE, not publishable "
          "as a deployable record.\n")
    print(f"Record to beat: Nearing et al. 2022 AR-LSTM = {RECORD} "
          "(CAMELS-531, day-1 nowcast, 1-day-lag Q)")

    report("", "2-seed members (the published set)")
    report("4", "4-seed members (deeper averaging - still deployable)")

    print("\n=== READING ===")
    print("  Plain mean has ZERO fitted parameters and is the rule Li/Song use")
    print("  for their own headline, so it is the honest number for a paper.")
    print("  If it clears 0.879, the record beat stands without any oracle.")


if __name__ == "__main__":
    main()
