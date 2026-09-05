#!/usr/bin/env python3
"""LEDGER 51 — choose the with-q combination rule on the HELD-OUT val window.

Every prior with-q combination number was chosen on rows the models had trained
on. This fits and selects on 1980-10-01..1989-09-30 (outside both the training
window and the test decade) with a three-way split, then freezes a JSON that is
applied to the test frame exactly once.

Rules considered (per-basin weights are excluded: closed on both tracks, and the
oracle gap is not reachable by any inverse-MSE estimator):
  - equal weight (the incumbent; with-q members were previously exchangeable)
  - global inverse-MSE, w_i = MSE_i^-theta, shrunk toward equal by lam
  - drop-one-member subsets
Readout is also selected here (mid vs ymed), since a probe trained with a
different loss may prefer a different slot.

usage: l51_withq_weights.py --members daymet,maurer,nldas,aorc,fused3 --out w.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.expanduser("~/riverwatch2"), "analysis"))
from l51_withq_score import join, per_basin  # noqa: E402

FIT_END = "1985-10-01"     # fit MSE here
SEL_END = "1989-09-30"     # select the rule here; test is never touched


def mse_weights(df, members, theta, lam):
    w = {}
    for m in members:
        e = df[m].to_numpy(float) - df.truth.to_numpy(float)
        w[m] = max(float(np.mean(e * e)), 1e-12) ** (-theta)
    tot = sum(w.values())
    w = {m: v / tot for m, v in w.items()}
    n = len(members)
    return {m: (1 - lam) * v + lam / n for m, v in w.items()}   # shrink to equal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="daymet,maurer,nldas,aorc,fused3")
    ap.add_argument("--out", default="benchmarks/l51_withq_weights.json")
    a = ap.parse_args()
    members = [x for x in a.members.split(",") if x]

    best = None
    for readout in ("mid", "ymed"):
        m = join(members, "val7", readout=readout)
        m["_dt"] = pd.to_datetime(m.t0)
        fit = m[m._dt < FIT_END]
        sel = m[m._dt >= FIT_END]
        print(f"\nreadout={readout}  fit rows {len(fit):,}  select rows {len(sel):,}")
        eq = per_basin(sel, members).median()
        print(f"  equal weight                      {eq:.6f}")
        cands = [("equal", None, eq)]
        for theta in (0.5, 1.0, 2.0, 4.0, 6.0):
            for lam in (0.0, 0.25, 0.5):
                w = mse_weights(fit, members, theta, lam)
                v = per_basin(sel, members, w).median()
                cands.append((f"invmse t={theta} lam={lam}", w, v))
                print(f"  invmse theta={theta:<4} lam={lam:<5}       {v:.6f}"
                      f"   w={ {k: round(x, 3) for k, x in w.items()} }")
        for drop in members:
            rest = [x for x in members if x != drop]
            v = per_basin(sel, rest).median()
            cands.append((f"drop {drop}", {x: 1.0 for x in rest}, v))
            print(f"  drop {drop:<28} {v:.6f}")
        top = max(cands, key=lambda c: c[2])
        print(f"  ==> best({readout}): {top[0]} = {top[2]:.6f}")
        if best is None or top[2] > best[3]:
            best = (readout, top[0], top[1], top[2])

    readout, rule, w, v = best
    out = {"readout": readout, "rule": rule, "weights": w, "select_median_nse": v,
           "members": members, "fit_window": ["1980-10-01", FIT_END],
           "select_window": [FIT_END, SEL_END],
           "note": "chosen on the held-out val window; apply to test ONCE"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nFROZEN: readout={readout} rule={rule} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
