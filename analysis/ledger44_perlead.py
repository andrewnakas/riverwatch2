#!/usr/bin/env python3
"""LEDGER 44 / S4 — IS THE BREADTH GAIN A HORIZON EFFECT OR A PHASE EFFECT?

THE CONFOUND
------------
`h==1` has been called "the phase lever" throughout this campaign — a 1-in-14
fixed-weekday date subsample. That is true, but it is only half of what h==1 is.
h is the LEAD. h==1 is simultaneously:
    (a) lead 1 — the shortest, most accurate horizon, and
    (b) the t0-grid dates only — a 1-in-14 date subsample.
Those two have never been separated, and the ledger-44 breadth result now depends
on which one it is: pooling the rejected members buys +0.002109 paired on all
leads and +0.000337 at h==1.

THE DISCRIMINATOR
-----------------
The 14 lead-sets PARTITION the calendar (date = t0 + h - 1, t0 on a stride-14
grid), so no lead is a privileged sample of dates. Therefore:
  * a MONOTONE TREND of the gain with h  => a HORIZON effect. The pooled members
    buy long-lead robustness, and the record's frame (lead 1) is simply where
    they help least. The all-leads gain then does NOT transfer to the record.
  * a FLAT profile with h==1 low          => a sampling/noise effect, and the
    h==1 reading is an unlucky draw rather than a real absence.

ONE weight vector throughout (production rule, fit on TRAIN all-leads rows), so
the per-lead deltas decompose the all-leads number rather than re-fitting 14
times. The h=1 point therefore need not equal the h==1-FRAME number, which uses
weights fit on TRAIN h==1 rows; that difference is itself reported.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/ledger44_perlead.json")
    args = ap.parse_args()
    pool = [t for t in KEPT if t != "aorc"]      # AMENDMENT 2: full frame
    print(f"pool ({len(pool)}): {pool}", file=sys.stderr)

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
    print(f"  TEST {len(TE):,} rows / {TE.station_id.nunique()} basins", file=sys.stderr)

    # ONE weight vector each, fit on TRAIN all-leads rows
    w_s = invmse_weights(TR, ship)
    w_a = invmse_weights(TR, allc)
    TE["_ship"] = (TE[ship].to_numpy(float) * w_s).sum(axis=1)
    TE["_all"] = (TE[allc].to_numpy(float) * w_a).sum(axis=1)

    print("\n" + "=" * 86)
    print("PER-LEAD DECOMPOSITION — pooled(19) minus shipped(9), one weight vector")
    print("=" * 86)
    print(f"{'h':>3} {'rows':>9} {'shipped':>10} {'pooled':>10} "
          f"{'paired d':>11} {'boot95':>26} {'breadth':>8}")
    print("-" * 86)
    rows = []
    for h in range(1, 15):
        g = TE[TE.h == h]
        b = per_basin_nse(g, "_ship")
        c = per_basin_nse(g, "_all")
        idx = b.index.intersection(c.index)
        d = (c[idx] - b[idx])
        lo, hi = boot(d.to_numpy(), n=2000)
        rows.append({"h": h, "rows": int(len(g)), "shipped": float(b[idx].median()),
                     "pooled": float(c[idx].median()), "paired": float(d.median()),
                     "boot95": [lo, hi], "breadth": float((d > 0).mean())})
        print(f"{h:>3} {len(g):>9,} {b[idx].median():>10.6f} {c[idx].median():>10.6f} "
              f"{d.median():>+11.6f} [{lo:>+9.6f},{hi:>+9.6f}] {(d>0).mean():>8.4f}")

    hs = np.array([r["h"] for r in rows], float)
    ds = np.array([r["paired"] for r in rows])
    slope, inter = np.polyfit(hs, ds, 1)
    r = float(np.corrcoef(hs, ds)[0, 1])
    from scipy import stats as st
    sp = st.spearmanr(hs, ds)
    print("\n" + "=" * 86)
    print("IS IT A HORIZON EFFECT?")
    print("=" * 86)
    print(f"  trend of the gain with lead : {slope:+.6f} NSE per lead")
    print(f"  pearson r (gain vs h)       : {r:+.3f}")
    print(f"  spearman rho                : {sp.statistic:+.3f}   p = {sp.pvalue:.4g}")
    print(f"  gain at h=1                 : {ds[0]:+.6f}")
    print(f"  gain at h=14                : {ds[-1]:+.6f}")
    print(f"  ratio h=14 / h=1            : {ds[-1]/ds[0] if ds[0] else float('nan'):.2f}x")
    verdict = ("HORIZON EFFECT — the gain grows with lead; the record's frame (lead 1) "
               "is where the pooled members help LEAST"
               if sp.pvalue < 0.05 and sp.statistic > 0 else
               "NO MONOTONE HORIZON TREND — the h==1 shortfall is not explained by lead")
    print(f"\n  => {verdict}")
    json.dump({"pool": pool, "per_lead": rows, "slope_per_lead": float(slope),
               "pearson_r": r, "spearman_rho": float(sp.statistic),
               "spearman_p": float(sp.pvalue), "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
