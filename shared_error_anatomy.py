#!/usr/bin/env python3
"""Anatomy of the 70% shared error: WHEN and HOW is every member wrong together?

The Krogh-Vedelsby decomposition says 70% of our member error is common to all
members and therefore untouchable by ensembling. That is a number, not a
diagnosis. Before proposing a fix, characterise the failure:

  Q1 is the shared error BIAS or TIMING?
     A constant multiplicative/additive offset is trivially correctable.
     A phase error (right shape, wrong day) is not.
  Q2 does it concentrate on HIGH FLOWS, LOW FLOWS, or RISING LIMBS?
     Each implies a different remedy.
  Q3 is it CONCENTRATED IN TIME (a few catastrophic days) or spread thin?
     A few days means events; spread means systematic misfit.
  Q4 which BASIN TYPES carry it?
     If it tracks aridity or dam density, the cause is physical, not modelling.

The answers decide the technique, so measuring them beats guessing. Fixes that
attack the wrong failure mode have already cost this campaign real time --
variance inflation targeted dispersion when the problem was transferability.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

D = "data/mblstm/gpu_dumps_s14"


def load(path, subset):
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(subset)]
    d["pred"] = (d.ylo + d.yhi) / 2
    return d[["station_id", "t0", "truth", "pred"]]


def main():
    ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
    mem = {}
    for f in ("daymet", "maurer", "nldas"):
        p = f"{D}/camels531_{f}_withq5_full531.csv.gz"
        if os.path.exists(p):
            mem[f] = load(p, ids)
    p = f"{D}/camels531_aorc_withq_full531.csv.gz"
    if os.path.exists(p):
        mem["aorc"] = load(p, ids)
    cols = list(mem)

    m = None
    for name, d in mem.items():
        x = d.rename(columns={"pred": name})
        m = x if m is None else m.merge(x[["station_id", "t0", name]],
                                       on=["station_id", "t0"])
    m["ens"] = m[cols].to_numpy(float).mean(1)
    m["err"] = m.ens - m.truth
    m["t0"] = pd.to_datetime(m.t0)
    print(f"members {cols}  rows {len(m)}  basins {m.station_id.nunique()}\n")

    # ---- Q1: bias vs timing -------------------------------------------------
    print("=== Q1: is the shared error BIAS or TIMING? ===")
    stats = []
    for sid, g in m.groupby("station_id"):
        y, p = g.truth.to_numpy(float), g.ens.to_numpy(float)
        if len(y) < 30 or np.var(y) < 1e-9:
            continue
        # decompose MSE into bias^2 + variance-mismatch + phase(1-r) terms
        mse = np.mean((y - p) ** 2)
        bias2 = (p.mean() - y.mean()) ** 2
        sy, sp = np.std(y), np.std(p)
        r = np.corrcoef(y, p)[0, 1]
        var_term = (sp - sy) ** 2
        phase = 2 * sp * sy * (1 - r)
        stats.append((bias2 / mse, var_term / mse, phase / mse))
    a = np.array(stats)
    print(f"  basins: {len(a)}")
    print(f"  bias^2        share of MSE: {np.median(a[:,0])*100:5.1f}%")
    print(f"  variance-miss share of MSE: {np.median(a[:,1])*100:5.1f}%")
    print(f"  PHASE (1-r)   share of MSE: {np.median(a[:,2])*100:5.1f}%")
    print("  -> a large phase share means timing, which post-hoc scaling cannot fix")

    # ---- Q2: where in the flow range ---------------------------------------
    print("\n=== Q2: which part of the hydrograph? ===")
    m["qtile"] = m.groupby("station_id").truth.transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    for q in range(5):
        s = m[m.qtile == q]
        rel = np.abs(s.err) / s.truth.clip(lower=1e-6)
        lab = ["lowest 20%", "20-40%", "40-60%", "60-80%", "highest 20%"][q]
        print(f"  {lab:12s} median |err|/obs = {rel.median():6.3f}  "
              f"mean signed err = {s.err.mean():+9.1f} cfs")

    # ---- Q3: concentrated in time? -----------------------------------------
    print("\n=== Q3: concentrated in a few days, or spread thin? ===")
    m["sq_err"] = m.err ** 2
    tot = m.sq_err.sum()
    for frac in (0.01, 0.05, 0.10):
        k = int(len(m) * frac)
        share = m.sq_err.nlargest(k).sum() / tot
        print(f"  worst {frac*100:4.1f}% of days carry {share*100:5.1f}% of total squared error")
    print("  -> heavy concentration means EVENTS, not systematic misfit")

    # ---- Q4: rising vs falling limb ----------------------------------------
    print("\n=== Q4: rising limb vs falling limb ===")
    m = m.sort_values(["station_id", "t0"])
    m["dq"] = m.groupby("station_id").truth.diff()
    rise, fall = m[m.dq > 0], m[m.dq < 0]
    for lab, s in (("rising", rise), ("falling", fall)):
        rel = np.abs(s.err) / s.truth.clip(lower=1e-6)
        print(f"  {lab:8s} n={len(s):7d}  median |err|/obs {rel.median():6.3f}  "
              f"mean signed {s.err.mean():+9.1f} cfs")
    print("  -> under-prediction on rising limbs is the classic LSTM event failure")

    m.drop(columns=["sq_err"]).to_csv("benchmarks/shared_error_rows.csv.gz",
                                      index=False, compression="gzip")
    print("\nwrote benchmarks/shared_error_rows.csv.gz")


if __name__ == "__main__":
    main()
