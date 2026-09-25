#!/usr/bin/env python3
"""LEDGER 44 / S5 — THE WEIGHT-FIT FRAME IS WORTH MORE THAN THE MEMBER POOL.

THE OBSERVATION THAT FORCED THIS
--------------------------------
Scored on the SAME 183,195 test rows at lead 1, the pooled-minus-shipped delta is
    +0.000337  when the weights are fit on TRAIN h==1 rows      (the h==1 FRAME)
    +0.002431  when the weights are fit on TRAIN ALL-LEAD rows  (per-lead table)
Same members, same scored rows, same estimator. Only the WEIGHT-FIT ROWS differ:
196,636 rows vs 2,752,904 -- 14x more data for the same 9 or 19 free parameters.

⚠️⚠️ ORDER OF DISCOVERY, STATED PLAINLY: this was noticed AFTER seeing held-out
numbers. Choosing a weight-fit frame because its TEST score is higher is
selection on test, and would be gate-shopping.

⇒ SO THE CHOICE IS ADJUDICATED TRAIN-SIDE, WHERE IT COSTS NOTHING TO BE WRONG.
Both weight vectors are fit on TRAIN rows and compared on the TRAIN VAL SLICE at
h==1 (1990-10-01..1995-09-30). No test information enters that comparison. Only
if all-leads fitting wins TRAIN-SIDE is it a legitimate a-priori choice, and the
test read is then reported as a consequence, not as the reason.

THE 2x2, on both surfaces:
    {shipped 9, pooled 19} x {weights fit on h==1 rows, weights fit on all leads}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger44_gate_transfer import (build_streams, merge, invmse_weights,   # noqa: E402
                                    per_basin_nse, TEST_END)
from ledger44_breadth_allleads import KEPT, add, boot                       # noqa: E402
from phase_lever import VAL_START, VAL_END                                  # noqa: E402

RECORD = 0.8362893021622821


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/ledger44_weightframe.json")
    args = ap.parse_args()
    pool = [t for t in KEPT if t != "aorc"]          # AMENDMENT 2: full frame
    print("building TRAIN...", file=sys.stderr)
    TR = merge(build_streams(True))
    print("building TEST...", file=sys.stderr)
    TE = merge(build_streams(False))
    for t in pool:
        TR = add(TR, t, True)
        TE = add(TE, t, False)
    TE = TE[TE.date <= TEST_END].copy()
    ship = [c for c in TR.columns if c not in ("station_id", "date", "h", "truth")
            and not c.startswith("pool_")]
    allc = ship + [f"pool_{t}" for t in pool]

    # the two weight-fit frames, BOTH entirely inside TRAIN
    tr_h1 = TR[TR.h == 1]
    fit_h1 = tr_h1[tr_h1.date < VAL_START]
    fit_al = TR[TR.date < VAL_START]
    print(f"\nweight-fit frames: h==1 {len(fit_h1):,} rows | "
          f"all-leads {len(fit_al):,} rows  ({len(fit_al)/len(fit_h1):.1f}x)")

    W = {("ship", "h1"): invmse_weights(fit_h1, ship),
         ("ship", "all"): invmse_weights(fit_al, ship),
         ("pool", "h1"): invmse_weights(fit_h1, allc),
         ("pool", "all"): invmse_weights(fit_al, allc)}

    def sc(df, cols, w, tag):
        d = df.copy()
        d[tag] = (d[cols].to_numpy(float) * w).sum(axis=1)
        return per_basin_nse(d, tag)

    out = {"pool": pool, "fit_rows": {"h1": int(len(fit_h1)), "all": int(len(fit_al))}}
    for surf, df in (("TRAIN val slice (in-sample, h==1)",
                      TR[(TR.h == 1) & (TR.date >= VAL_START) & (TR.date <= VAL_END)]),
                     ("HELD-OUT test (h==1)", TE[TE.h == 1])):
        key = "train" if surf.startswith("TRAIN") else "test"
        print("\n" + "=" * 78)
        print(f"{surf}   n={len(df):,} rows")
        print("=" * 78)
        res = {}
        for cfg, cols in (("ship", ship), ("pool", allc)):
            for wf in ("h1", "all"):
                s = sc(df, cols, W[(cfg, wf)], "_e")
                res[f"{cfg}_{wf}"] = s
                print(f"  {cfg:<5} weights fit on {wf:<4} -> median NSE {s.median():.6f}")
        # the two questions, as PAIRED deltas
        print()
        for label, a, b in (
                ("weight frame, shipped  (all vs h1)", "ship_h1", "ship_all"),
                ("weight frame, pooled   (all vs h1)", "pool_h1", "pool_all"),
                ("member pool @ h1-fit weights      ", "ship_h1", "pool_h1"),
                ("member pool @ all-fit weights     ", "ship_all", "pool_all"),
                ("BOTH: ship/h1 -> pool/all         ", "ship_h1", "pool_all")):
            x, y = res[a], res[b]
            idx = x.index.intersection(y.index)
            d = (y[idx] - x[idx])
            lo, hi = boot(d.to_numpy(), n=3000)
            out[f"{key}::{label.strip()}"] = {
                "paired": float(d.median()), "boot95": [lo, hi],
                "breadth": float((d > 0).mean()),
                "from": float(x[idx].median()), "to": float(y[idx].median())}
            star = " *CI excludes 0*" if lo > 0 else ""
            print(f"  {label}  {d.median():>+10.6f}  "
                  f"[{lo:>+9.6f},{hi:>+9.6f}]  breadth {(d>0).mean():.4f}{star}")
        if key == "test":
            print(f"\n  vs THE RECORD {RECORD:.6f}:")
            for k in ("ship_h1", "ship_all", "pool_h1", "pool_all"):
                print(f"    {k:<9} {res[k].median():.6f}   {res[k].median()-RECORD:+.6f}")
            out["test_medians"] = {k: float(res[k].median())
                                   for k in ("ship_h1", "ship_all", "pool_h1", "pool_all")}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
