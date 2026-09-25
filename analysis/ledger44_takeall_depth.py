#!/usr/bin/env python3
"""LEDGER 44 / S3-PRICING — WHAT SEEDING UP THE TAKE-ALL ENSEMBLE IS WORTH.

S1b measured BREADTH on the take-all 20-stream configuration: +0.002444 paired
all-leads (CI excludes zero), +0.000396 h==1. D-PARTIAL against a +0.003 bar.

But that configuration is SEED-STARVED: 8 of its 11 new members have exactly ONE
seed. Their seed noise is entirely uncorrected, so the take-all ensemble sits far
further from its own infinite-seed limit than the shipped 9 does. Seeding them up
is precisely what unbounded GPU buys, and this prices it BEFORE any GPU is spent.

Same closed-form estimator as ledger44_infinite_seed_limit.py:
    MSE_inf = MSE_obs - SUM_i w_i^2 * sigmahat_i^2 / k_i

⚠️ k=1 STREAMS HAVE NO WITHIN-STREAM VARIANCE ESTIMATE. That is handled honestly
by reporting a BRACKET rather than a point:
  * LOWER BOUND  — k=1 streams contribute 0 to the correction (they cannot be
                   corrected, so this understates the headroom).
  * ESTIMATE     — k=1 streams are assigned the mean per-row sigma^2 of the
                   multi-seed LSTM streams, which are the same architecture
                   family trained the same way.
The truth is between them, and BOTH are reported.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())

import gate_eval as G                                                  # noqa: E402
from phase_c import load_seed_avg                                      # noqa: E402
from ledger44_ensembling_law import fit_global_invmse, per_basin_nse   # noqa: E402
from ledger44_breadth_curve import (SHIPPED, KEPT_ORDER, paths_for,    # noqa: E402
                                    pool_tags, SOLO_BAR)

CV_FIT_END, CV_VAL_START = G.CV_FIT_END, G.CV_VAL_START
RECORD = 0.8362893021622821


def load_per_seed(names, tag_seeds, train):
    ids = set(str(x).zfill(8) for x in json.load(open(G.SUBSET))["531"])
    per, truth = {}, None
    for nm in names:
        ps, shift = paths_for(nm, tag_seeds, train)
        cols = []
        for p in ps:
            d = load_seed_avg([Path(p)], shift_days=shift)
            d = d[d.station_id.isin(ids)].set_index(["station_id", "date"])
            cols.append(d["pred"])
            if truth is None:
                truth = d["truth"]
        per[nm] = cols
    idx = None
    for cols in per.values():
        for s in cols:
            idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.intersection(truth.index)
    mat = {nm: np.stack([s.reindex(idx).to_numpy(float) for s in cols])
           for nm, cols in per.items()}
    return (mat, np.array([b for b, _ in idx]), np.array([d for _, d in idx]),
            truth.reindex(idx).to_numpy(float))


def limit(mat, sids, truth, w, cols, impute):
    """Infinite-seed median NSE. impute=False -> lower bound (k=1 contribute 0)."""
    ks = {c: mat[c].shape[0] for c in cols}
    var = {c: (mat[c].var(axis=0, ddof=1) if ks[c] > 1 else None) for c in cols}
    multi = [v for c, v in var.items() if v is not None]
    pooled = np.mean(np.stack(multi), axis=0) if multi else None
    W = dict(zip(cols, w))
    corr = np.zeros(len(truth))
    for c in cols:
        v = var[c]
        if v is None:
            if not impute:
                continue
            v = pooled
        corr = corr + W[c] ** 2 * v / max(ks[c], 1)
    yhat = sum(W[c] * mat[c].mean(axis=0) for c in cols)
    out = []
    for b in np.unique(sids):
        m = sids == b
        y, p, cc = truth[m], yhat[m], corr[m]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p, cc = y[ok], p[ok], cc[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out.append(1.0 - (np.mean((y - p) ** 2) - np.mean(cc)) / np.var(y))
    return float(np.median(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/ledger44_takeall_depth.json")
    args = ap.parse_args()

    tag_seeds = dict(pool_tags())
    names = SHIPPED + [f"pool_{t}" for t in KEPT_ORDER]
    print(f"take-all = {len(names)} streams", file=sys.stderr)
    print("loading TRAIN per-seed...", file=sys.stderr)
    TR = load_per_seed(names, tag_seeds, True)
    print("loading TEST per-seed...", file=sys.stderr)
    TE = load_per_seed(names, tag_seeds, False)
    trm, trs, trd, trt = TR
    tem, tes, ted, tet = TE
    print(f"  TEST frame {len(tet):,} rows / {len(set(tes))} basins", file=sys.stderr)

    res = {}
    print("\n" + "=" * 78)
    print("SEED DEPTH OF EACH STREAM IN THE TAKE-ALL CONFIGURATION")
    print("=" * 78)
    for nm in names:
        k = trm[nm].shape[0]
        print(f"  {nm.replace('pool_',''):<16} k={k}" + ("   <-- SEED-STARVED" if k == 1 else ""))
    starved = sum(1 for nm in names if trm[nm].shape[0] == 1)
    print(f"\n  {starved} of {len(names)} streams have a single seed")

    for label, cols in (("shipped 9", SHIPPED), ("take-all", names)):
        w = fit_global_invmse(np.stack([trm[c].mean(axis=0) for c in cols]), trt)
        yhat = (np.stack([tem[c].mean(axis=0) for c in cols]) * w[:, None]).sum(axis=0)
        meas = float(np.median(per_basin_nse(tes, tet, yhat)))
        lo = limit(tem, tes, tet, w, cols, impute=False)
        hi = limit(tem, tes, tet, w, cols, impute=True)
        res[label] = {"measured": meas, "limit_lower": lo, "limit_estimate": hi,
                      "headroom_lower": lo - meas, "headroom_estimate": hi - meas,
                      "n_streams": len(cols)}
        print("\n" + "=" * 78)
        print(f"{label.upper()} — HELD-OUT, h==1, on the take-all common frame")
        print("=" * 78)
        print(f"  measured                       {meas:.6f}")
        print(f"  infinite-seed LOWER BOUND      {lo:.6f}   headroom {lo-meas:+.6f}")
        print(f"  infinite-seed ESTIMATE         {hi:.6f}   headroom {hi-meas:+.6f}")

    b, t = res["shipped 9"], res["take-all"]
    print("\n" + "=" * 78)
    print("⇒ WHAT UNBOUNDED GPU BUYS ON TOP OF BREADTH (h==1, same frame)")
    print("=" * 78)
    print(f"  shipped 9 measured                       {b['measured']:.6f}")
    print(f"  take-all measured                        {t['measured']:.6f}   "
          f"({t['measured']-b['measured']:+.6f} from breadth)")
    print(f"  take-all + infinite seeds, lower bound   {t['limit_lower']:.6f}   "
          f"({t['limit_lower']-b['measured']:+.6f} total)")
    print(f"  take-all + infinite seeds, estimate      {t['limit_estimate']:.6f}   "
          f"({t['limit_estimate']-b['measured']:+.6f} total)")
    res["total_vs_shipped"] = {
        "breadth_only": t["measured"] - b["measured"],
        "breadth_plus_depth_lower": t["limit_lower"] - b["measured"],
        "breadth_plus_depth_estimate": t["limit_estimate"] - b["measured"]}
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
