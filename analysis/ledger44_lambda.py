#!/usr/bin/env python3
"""LEDGER 44 / S6 — RE-TUNE THE SHRINKAGE FOR A 19-MEMBER POOL. TRAIN-SIDE ONLY.

⛔⛔ THIS SCRIPT NEVER TOUCHES THE TEST WINDOW. By design and by construction:
it loads only TRAIN dumps. The test window has already been read several times in
this ledger and must not become a tuning surface.

THE PREDICTION, WRITTEN BEFORE THE NUMBERS
------------------------------------------
The production rule is  w = lam*(1/n) + (1-lam)*normalize(mse^-theta),
with lam=0.25, theta=4.0 -- tuned when the ensemble had NINE members.

S5 measured that the pool's value collapses when its weights are badly estimated
(+0.001110 at 1.6M fitting rows vs +0.000013 at 115k, TRAIN-side). 19 members
means 19 weights from the same data that fitted 9, so the inverse-MSE vector is
noisier, and shrinkage toward equal weight is exactly the remedy for that.

⇒ PREDICTED, a priori: the optimal lam for the 19-member pool is HIGHER than
0.25, and re-tuning it recovers part of the pool's value WITHOUT changing the
weight-fit frame (which the train-side adjudication in S5 rejected).

FALSIFIABLE: if the optimal lam for 19 members is <= 0.25, or if the curve is
flat, the mechanism claimed in S5 is wrong and this closes.

PROTOCOL — an honest train-side split, no test data anywhere
------------------------------------------------------------
  fit weights   : TRAIN rows before 1990-10-01
  select lambda : TRAIN val slice 1990-10-01..1995-09-30, h==1
Selecting on the val slice is the campaign's standard train-side selection
surface. It is in-sample for the NETWORKS (that bias is now measured and is why
no test claim is made here), but it contains no test rows.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger44_gate_transfer import (build_streams, merge, per_basin_nse,  # noqa: E402
                                    THETA)
from ledger44_breadth_allleads import KEPT, add, boot                     # noqa: E402
from phase_lever import VAL_START, VAL_END                                # noqa: E402

LAMS = [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0]


def weights(fit, cols, lam, theta=THETA):
    truth = fit["truth"].to_numpy(float)
    mse = np.array([np.nanmean((fit[c].to_numpy(float) - truth) ** 2) for c in cols])
    raw = mse ** (-theta)
    raw = raw / raw.sum()
    w = lam * np.ones(len(cols)) / len(cols) + (1 - lam) * raw
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/ledger44_lambda_trainside.json")
    args = ap.parse_args()
    pool = [t for t in KEPT if t != "aorc"]
    print("building TRAIN only (test window is NOT loaded)...", file=sys.stderr)
    TR = merge(build_streams(True))
    for t in pool:
        TR = add(TR, t, True)
    ship = [c for c in TR.columns if c not in ("station_id", "date", "h", "truth")
            and not c.startswith("pool_")]
    allc = ship + [f"pool_{t}" for t in pool]

    h1 = TR[TR.h == 1]
    fit = h1[h1.date < VAL_START]
    val = h1[(h1.date >= VAL_START) & (h1.date <= VAL_END)]
    print(f"fit {len(fit):,} rows | select {len(val):,} rows | "
          f"{len(ship)} vs {len(allc)} streams\n")

    print("=" * 78)
    print("SHRINKAGE CURVE, TRAIN-SIDE (select on the val slice, h==1)")
    print("=" * 78)
    print(f"{'lambda':>7} | {'shipped 9':>11} | {'pooled 19':>11} | {'pool - ship':>12}")
    print("-" * 52)
    rows = []
    for lam in LAMS:
        s = per_basin_nse(val.assign(_e=(val[ship].to_numpy(float)
                                         * weights(fit, ship, lam)).sum(axis=1)), "_e")
        p = per_basin_nse(val.assign(_e=(val[allc].to_numpy(float)
                                         * weights(fit, allc, lam)).sum(axis=1)), "_e")
        idx = s.index.intersection(p.index)
        d = (p[idx] - s[idx])
        rows.append({"lam": lam, "ship": float(s[idx].median()),
                     "pool": float(p[idx].median()), "paired": float(d.median()),
                     "breadth": float((d > 0).mean())})
        print(f"{lam:>7.2f} | {s[idx].median():>11.6f} | {p[idx].median():>11.6f} | "
              f"{d.median():>+12.6f}")

    best_p = max(rows, key=lambda r: r["pool"])
    best_s = max(rows, key=lambda r: r["ship"])
    print("\n" + "=" * 78)
    print("THE PREDICTION, TESTED")
    print("=" * 78)
    print(f"  best lambda for the SHIPPED 9  : {best_s['lam']:.2f}  "
          f"(median {best_s['ship']:.6f})")
    print(f"  best lambda for the POOLED 19  : {best_p['lam']:.2f}  "
          f"(median {best_p['pool']:.6f})")
    print(f"  production lambda              : 0.25")
    higher = best_p["lam"] > 0.25
    print(f"\n  predicted: optimal lambda for 19 members > 0.25")
    print(f"  measured : {best_p['lam']:.2f}  => "
          f"{'✅ PREDICTION HELD' if higher else '⛔ PREDICTION REFUTED'}")
    at25 = [r for r in rows if r["lam"] == 0.25][0]
    gain = best_p["pool"] - at25["pool"]
    print(f"\n  pooled-19 at lambda=0.25 : {at25['pool']:.6f}")
    print(f"  pooled-19 at best lambda : {best_p['pool']:.6f}   "
          f"recovered {gain:+.6f} (TRAIN-SIDE)")
    print(f"\n  ⚠️ TRAIN-SIDE ONLY. In-sample understates ensembling ~2.3x and")
    print(f"     overstates member swaps; no test claim follows from this number.")
    json.dump({"curve": rows, "best_lambda_pool": best_p["lam"],
               "best_lambda_ship": best_s["lam"], "prediction_held": bool(higher),
               "trainside_gain_vs_lam025": float(gain),
               "note": "TRAIN-SIDE ONLY. No test rows loaded."},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
