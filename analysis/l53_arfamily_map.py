#!/usr/bin/env python3
"""LEDGER 54 GATE 0 — what would it take to reach 0.90? (zero GPU)

The 6-member ensemble is 5 MB-LSTM members (solo 0.843-0.872) + 1 AR-LSTM member
(`nhar`, solo 0.886) at equal weight, i.e. the AR family holds 1/6 of the weight.
Adding K more AR-LSTM members at equal weight is, to first order, the same as
moving the AR family's weight to K/(5+K). This maps that curve BEFORE any member
is built, and measures whether the AR family is a genuinely separate error block.

Three read-outs:
  1. ERROR-CORRELATION BLOCK STRUCTURE — is `nhar` a new block or a 6th cousin?
  2. THE AR-WEIGHT CURVE — ensemble median NSE vs the AR family's weight share,
     on val1 (honest) and test1; where is the optimum and what is the ceiling?
  3. SEED-STREAMS vs SEED-AVERAGE — treat `nhar`'s 3 seeds as 3 separate members
     (8 streams, equal weight) vs one 3-seed-averaged member. The difference is
     what "more AR members" buys that "more seeds" cannot.

⚠️ DIAGNOSTIC ONLY. A weight curve is a fitted object; nothing here selects the
shipped composition (that stays equal-weight over built members, decided on val1
with the paired statistic). This decides WHAT TO BUILD.

usage: l53_arfamily_map.py --frame val1 --out X.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)

FIVE = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1"]


def per_basin_err_corr(m, cols):
    """Mean per-basin Pearson correlation of the members' errors, pairwise."""
    err = {c: (m[c].to_numpy(float) - m.truth.to_numpy(float)) for c in cols}
    idx = m.groupby("station_id").indices
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for b in cols[i:]:
            rs = []
            for s, ii in idx.items():
                x, y = err[a][ii], err[b][ii]
                k = np.isfinite(x) & np.isfinite(y)
                if k.sum() < 20:
                    continue
                x, y = x[k], y[k]
                if x.std() < 1e-9 or y.std() < 1e-9:
                    continue
                rs.append(np.corrcoef(x, y)[0, 1])
            v = float(np.median(rs)) if rs else np.nan
            out.loc[a, b] = out.loc[b, a] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print(f"LEDGER 54 GATE 0 — frame={a.frame}\n")
    parts = [l51.load_member(x, a.frame) for x in FIVE]
    parts.append(l51.load_member("nhar", a.frame, seeds={501, 502, 503}))
    per_seed = {s: l51.load_member("nhar", a.frame, seeds={s}).rename(columns={"nhar": f"nhar_s{s}"})
                for s in (501, 502, 503)}
    m = parts[0]
    for d in parts[1:]:
        m = m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    for s, d in per_seed.items():
        m = m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    print(f"joined rows {len(m):,}  basins {m.station_id.nunique()}\n")
    res = {"frame": a.frame, "rows": int(len(m)), "basins": int(m.station_id.nunique())}

    # ---- 1. error-correlation block structure --------------------------------
    cols = FIVE + ["nhar"]
    C = per_basin_err_corr(m, cols)
    print("MEDIAN PER-BASIN ERROR CORRELATION")
    print(C.round(3).to_string(), "\n")
    mb = [C.loc[x, y] for i, x in enumerate(FIVE) for y in FIVE[i + 1:]]
    cross = [C.loc[x, "nhar"] for x in FIVE]
    res["err_corr"] = {"matrix": C.round(6).to_dict(),
                       "mb_lstm_block_mean": float(np.mean(mb)),
                       "nhar_vs_mb_mean": float(np.mean(cross))}
    print(f"  MB-LSTM block (10 pairs) mean r = {np.mean(mb):.4f}")
    print(f"  nhar vs MB-LSTM  (5 pairs) mean r = {np.mean(cross):.4f}")
    print(f"  => nhar sits {np.mean(mb) - np.mean(cross):+.4f} below the existing block\n")

    # ---- 2. the AR-weight curve ---------------------------------------------
    print("AR-FAMILY WEIGHT CURVE  (w = AR share; K = equivalent # of equal-weight AR members)")
    print(f"  {'w':>6} {'K':>6}  {'median NSE':>11}  {'vs equal-weight A':>18}")
    curve = []
    base_w = None
    for K in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 25]:
        w = K / (5.0 + K)
        W = {**{c: (1 - w) / 5.0 for c in FIVE}, "nhar": w}
        med = float(l51.per_basin(m, cols, W).median())
        if K == 1:
            base_w = med
        curve.append({"K": K, "w": w, "median": med})
    for c in curve:
        d = "" if base_w is None else f"{c['median'] - base_w:+.6f}"
        print(f"  {c['w']:6.3f} {c['K']:6.1f}  {c['median']:11.6f}  {d:>18}")
    best = max(curve, key=lambda c: c["median"])
    res["ar_weight_curve"] = curve
    res["curve_optimum"] = best
    print(f"\n  OPTIMUM: w={best['w']:.3f} (K≈{best['K']:.1f} AR members) -> {best['median']:.6f}")
    print(f"  gap to 0.90 at the optimum: {0.90 - best['median']:+.6f}\n")

    # ---- 3. seed-streams vs seed-average ------------------------------------
    seedcols = [f"nhar_s{s}" for s in (501, 502, 503)]
    e_avg = l51.per_basin(m, cols)                       # 6 members, nhar = 3-seed avg
    e_str = l51.per_basin(m, FIVE + seedcols)            # 8 streams, one per seed
    d, lo, hi, br, n = l51.paired_ci(e_str, e_avg)
    res["seed_streams_vs_average"] = {"streams_median": float(e_str.median()),
                                      "average_median": float(e_avg.median()),
                                      "paired_delta": d, "ci": [lo, hi], "breadth": br}
    print("SEED-STREAMS vs SEED-AVERAGE (both use the same 3 networks)")
    print(f"  6 members, nhar = 3-seed average : {e_avg.median():.6f}")
    print(f"  8 streams, one per nhar seed     : {e_str.median():.6f}")
    print(f"  paired {d:+.6f}  CI[{lo:+.6f},{hi:+.6f}]  breadth {br:.3f}")
    print("  (a positive delta means WEIGHT SHARE is what matters, not network count)\n")

    # ---- solo table ----------------------------------------------------------
    res["solo"] = {c: float(l51.per_basin(m, [c]).median()) for c in cols + seedcols}
    print("SOLO")
    for k, v in res["solo"].items():
        print(f"  {k:<12} {v:.6f}")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
