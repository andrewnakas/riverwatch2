#!/usr/bin/env python3
"""LEDGER 54 — is `nhar0` a DISTINCT member from `nhar`, or effectively a copy?

They share an architecture and differ only in the training-time lagged-discharge
holdout (0.5 vs 0.0). If their errors are near-identical the ensemble gains
nothing from carrying both, and the duplicate-member logic applies.

Reports: median per-basin error correlation between every pair, and the paired
delta of adding `nhar0` to the shipped 6-member ensemble.

usage: l54_member_pair.py --frame val1 [--out X.json]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)

FIVE = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--extra", default="nhar0", help="candidate 7th member")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    X = a.extra

    parts = [l51.load_member(x, a.frame) for x in FIVE] + \
            [l51.load_member("nhar", a.frame), l51.load_member(X, a.frame)]
    m = parts[0]
    for d in parts[1:]:
        m = m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    print(f"\njoined {len(m):,} rows  {m.station_id.nunique()} basins\n")

    cols = FIVE + ["nhar", X]
    err = {c: (m[c].to_numpy(float) - m.truth.to_numpy(float)) for c in cols}
    idx = m.groupby("station_id").indices

    def corr(x, y):
        rs = []
        for s, ii in idx.items():
            p, q = err[x][ii], err[y][ii]
            k = np.isfinite(p) & np.isfinite(q)
            if k.sum() < 20:
                continue
            p, q = p[k], q[k]
            if p.std() < 1e-9 or q.std() < 1e-9:
                continue
            rs.append(np.corrcoef(p, q)[0, 1])
        return float(np.median(rs))

    r_pair = corr("nhar", X)
    r_nhar_mb = float(np.mean([corr("nhar", c) for c in FIVE]))
    r_x_mb = float(np.mean([corr(X, c) for c in FIVE]))
    mb_block = float(np.mean([corr(x, y) for i, x in enumerate(FIVE) for y in FIVE[i + 1:]]))
    print("MEDIAN PER-BASIN ERROR CORRELATION")
    print(f"  nhar  vs {X:<10} r = {r_pair:.4f}   <- 1.0 would mean a duplicate")
    print(f"  nhar  vs MB-LSTM mean  r = {r_nhar_mb:.4f}")
    print(f"  {X:<5} vs MB-LSTM mean  r = {r_x_mb:.4f}")
    print(f"  MB-LSTM internal block r = {mb_block:.4f}\n")

    e6 = l51.per_basin(m, FIVE + ["nhar"])
    e7 = l51.per_basin(m, cols)
    d, lo, hi, br, n = l51.paired_ci(e7, e6)
    print(f"  6-member (base5 + nhar)      {e6.median():.6f}")
    print(f"  7-member (+ {X})         {e7.median():.6f}")
    print(f"  paired {d:+.6f}  CI[{lo:+.6f},{hi:+.6f}]  breadth {br:.3f}  "
          f"{'CI excludes 0' if (lo > 0 or hi < 0) else 'CI straddles 0'}")
    print(f"\n  solo nhar  {l51.per_basin(m, ['nhar']).median():.6f}"
          f"   solo {X} {l51.per_basin(m, [X]).median():.6f}")

    res = {"frame": a.frame, "extra": X, "rows": int(len(m)), "basins": int(m.station_id.nunique()),
           "err_corr": {"nhar_vs_extra": r_pair, "nhar_vs_mb": r_nhar_mb,
                        "extra_vs_mb": r_x_mb, "mb_block": mb_block},
           "median_6": float(e6.median()), "median_7": float(e7.median()),
           "paired": {"delta": d, "ci": [lo, hi], "breadth": br,
                      "significant": bool(lo > 0 or hi < 0)}}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
