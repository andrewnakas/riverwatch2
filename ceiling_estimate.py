#!/usr/bin/env python3
"""Estimate the LSTM-ensemble ceiling for CAMELS from member disagreement.

The field asserts a ceiling exists but has never put a number on it. This
estimates one from data we already have, using a bias-variance decomposition of
the ensemble error.

THE IDEA. For an ensemble of M members with predictions f_i and truth y, the
mean squared error of the ensemble decomposes exactly (Krogh & Vedelsby 1995):

    MSE(ensemble) = mean_i MSE(f_i) - mean_i (f_i - f_bar)^2
                    \_______________/   \___________________/
                      average error          DIVERSITY

The second term is inter-member disagreement, and it is the part that ensembling
removes. What survives is the error the members SHARE -- their common bias
against truth. No amount of extra members of the same kind removes it.

So the shared-error component gives an achievable-NSE bound for THIS FAMILY of
models on THIS data. Adding decorrelated members raises the diversity term and
eats into the gap; nothing raises it past the shared floor except a genuinely
different information source or better data.

WHAT THIS IS NOT. It is not the true hydrologic predictability limit -- gauge
error (up to 20% of observed discharge) sets a lower bound on achievable MSE
that this cannot see, because our only estimate of truth IS the gauge. So the
number produced here is an UPPER bound on what better LSTM ensembling can reach,
and the real limit is at or below it.

Two independent estimates are reported so they can be checked against each
other:
  A) extrapolate the measured diversity term to M -> infinity
  B) per-basin best-member oracle, an upper bound on recombination alone
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

D = "data/mblstm/gpu_dumps_s14"
IDS = "data/camels_gauge_ids.json"


def load(path, subset):
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(subset)]
    d["pred"] = (d.ylo + d.yhi) / 2
    return d[["station_id", "t0", "truth", "pred"]]


def nse_from(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def main():
    ids = set(str(x).zfill(8) for x in json.load(open(IDS))["531"])
    members = {}
    for f in ("daymet", "maurer", "nldas"):
        p = f"{D}/camels531_{f}_withq5_full531.csv.gz"
        if os.path.exists(p):
            members[f] = load(p, ids)
    p = f"{D}/camels531_aorc_withq_full531.csv.gz"
    if os.path.exists(p):
        members["aorc"] = load(p, ids)
    if len(members) < 3:
        sys.exit("need at least 3 members")
    print(f"members: {list(members)}")

    m = None
    for name, d in members.items():
        x = d.rename(columns={"pred": name})
        m = x if m is None else m.merge(x[["station_id", "t0", name]],
                                       on=["station_id", "t0"])
    cols = list(members)
    print(f"aligned rows: {len(m)}  basins: {m.station_id.nunique()}\n")

    rows = []
    for sid, g in m.groupby("station_id"):
        y = g.truth.to_numpy(float)
        P = g[cols].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(P).all(axis=1)
        y, P = y[ok], P[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        fbar = P.mean(axis=1)
        var_y = np.var(y)
        avg_member_mse = np.mean([(np.mean((y - P[:, i]) ** 2)) for i in range(P.shape[1])])
        diversity = np.mean([(np.mean((P[:, i] - fbar) ** 2)) for i in range(P.shape[1])])
        ens_mse = np.mean((y - fbar) ** 2)
        best = max(nse_from(y, P[:, i]) for i in range(P.shape[1]))
        rows.append({
            "sid": sid, "var_y": var_y,
            "avg_member_mse": avg_member_mse,
            "diversity": diversity,
            "ens_mse": ens_mse,
            "ens_nse": 1 - ens_mse / var_y,
            "best_member_nse": best,
        })

    df = pd.DataFrame(rows)
    M = len(cols)
    print(f"=== decomposition check (Krogh-Vedelsby), M={M} ===")
    lhs = df.ens_mse
    rhs = df.avg_member_mse - df.diversity
    print(f"  max |MSE_ens - (avg_member_MSE - diversity)| = {np.abs(lhs-rhs).max():.3e}")
    print("  (should be ~0; confirms the identity holds on our data)\n")

    print("=== A) extrapolate diversity to infinite members ===")
    # With M exchangeable members, diversity scales as (1 - 1/M) * sigma^2 where
    # sigma^2 is the per-member spread around the infinite-ensemble mean. So the
    # M->inf ensemble MSE is avg_member_mse - sigma^2, with sigma^2 = diversity*M/(M-1).
    sigma2 = df.diversity * M / (M - 1)
    mse_inf = (df.avg_member_mse - sigma2).clip(lower=0)
    nse_inf = 1 - mse_inf / df.var_y
    print(f"  current ensemble  median NSE {df.ens_nse.median():.4f}")
    print(f"  M -> infinity     median NSE {nse_inf.median():.4f}")
    print(f"  headroom from MORE OF THE SAME members: "
          f"{nse_inf.median()-df.ens_nse.median():+.4f}")

    print("\n=== B) per-basin best-member oracle (recombination bound) ===")
    print(f"  oracle median NSE {df.best_member_nse.median():.4f}")
    print(f"  headroom over ensemble: {df.best_member_nse.median()-df.ens_nse.median():+.4f}")

    print("\n=== the SHARED error: what ensembling can never remove ===")
    shared_frac = ((df.avg_member_mse - sigma2).clip(lower=0) / df.avg_member_mse)
    print(f"  median share of member error that is COMMON to all members: "
          f"{shared_frac.median()*100:.1f}%")
    print(f"  median share removable by ensembling: {(1-shared_frac.median())*100:.1f}%")

    print("\n=== interpretation ===")
    print(f"  Ensembling more members of this FAMILY asymptotes at ~"
          f"{nse_inf.median():.3f} median NSE.")
    print("  That is an UPPER bound on better LSTM ensembling, not the")
    print("  hydrologic limit: gauge error (to 20% of observed Q) sits below it")
    print("  and is invisible here, because our only estimate of truth is the")
    print("  gauge itself.")
    print("  Passing it requires a different information source, not more or")
    print("  better-weighted members of the same kind.")

    df.to_csv("benchmarks/ceiling_estimate_per_basin.csv", index=False)
    print("\nwrote benchmarks/ceiling_estimate_per_basin.csv")


if __name__ == "__main__":
    main()
