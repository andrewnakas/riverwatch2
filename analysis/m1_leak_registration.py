#!/usr/bin/env python3
"""LEAK AUDIT — precipitation registration on the BUILT modern corpus.

THE RISK. A product that stores at date t the rain falling on t+1 hands the decoder
TOMORROW'S RAIN at the step it is scored on. This campaign measured that exact defect at
**+0.229 NSE** on Daymet V4R1, and it is invisible to mean/bias checks. A +1-day shift was
applied to Daymet precipitation at build time; if the direction were wrong the record would
be inflated by a future-information leak.

THE TEST, self-contained (no frozen reference needed, so it runs on the TEST window):
  for k in -3..+3:  corr( precip[t] , relative dQ[t+k] )   where relative dQ = dQ/Q
  k = 0  => rain at t drives the flow change at t. CORRECT.
  k > 0  => precip[t] predicts the flow change at t+k, so the value at row t is rain that
            falls LATER. A decoder reading the target-day row then receives FUTURE rain, and
            loses the causal driver. ⛔ MISREGISTERED.
  k < 0  => the label trails its effect; the product is STALE. Conservative, not future info.

⚠️ CORRECTED 2026-10-07. The first version of this file had these two cases INVERTED and
printed "NO LEAK" for gridMET, which peaks at k=+1. Only reading the per-lag numbers caught
it. An audit tool that can emit a false clean verdict is worse than no tool, so the verdict
logic below now keys on k>0.
Peak-over-lag is registration-sensitive by construction, which is the point.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd

LAGS = list(range(-3, 4))


def product_peak(cdir, ids, y0, y1, nbasin):
    files = [os.path.join(cdir, f"{g}.csv.gz") for g in sorted(ids)]
    files = [f for f in files if os.path.exists(f)][:nbasin]
    per_basin = []
    for f in files:
        d = pd.read_csv(f, dtype={"date": str})
        d = d[(d.date >= f"{y0}-01-01") & (d.date <= f"{y1}-12-31")]
        if len(d) < 400 or "precipitation_sum" not in d or "q_cfs" not in d:
            continue
        p = d.precipitation_sum.to_numpy(float)
        q = d.q_cfs.to_numpy(float)
        dq = np.full_like(q, np.nan)
        dq[1:] = np.diff(q)
        rel = dq / np.maximum(q, 1e-6)          # relative change, scale-free
        row = {}
        for k in LAGS:
            a = p if k >= 0 else p[-k:]
            b = np.roll(rel, -k)
            n = min(len(p), len(rel))
            aa, bb = p[:n], np.roll(rel, -k)[:n]
            m = np.isfinite(aa) & np.isfinite(bb)
            if m.sum() < 400:
                row[k] = np.nan
                continue
            row[k] = float(np.corrcoef(aa[m], bb[m])[0, 1])
        per_basin.append(row)
    if not per_basin:
        return None
    med = {k: float(np.nanmedian([r[k] for r in per_basin])) for k in LAGS}
    peak = max(LAGS, key=lambda k: (med[k] if np.isfinite(med[k]) else -9))
    modes = []
    for r in per_basin:
        vals = {k: r[k] for k in LAGS if np.isfinite(r[k])}
        if vals:
            modes.append(max(vals, key=vals.get))
    mode = int(pd.Series(modes).mode().iloc[0]) if modes else None
    return {"n_basins": len(per_basin), "median_r_by_lag": med,
            "peak_lag_of_median": peak, "modal_peak_lag": mode,
            "frac_basins_peaking_at_0": float(np.mean([m == 0 for m in modes])) if modes else None,
            "frac_basins_peaking_negative": float(np.mean([m < 0 for m in modes])) if modes else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True, help="name=dir,name=dir")
    ap.add_argument("--y0", type=int, default=2019)
    ap.add_argument("--y1", type=int, default=2025)
    ap.add_argument("--basins", type=int, default=120)
    ap.add_argument("--cohort", default="data/m1_cohort.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [str(x).zfill(8) for x in json.load(open(a.cohort))["cohort"]]
    out = {"window": [a.y0, a.y1], "lags": LAGS, "products": {}}
    print(f"\nwindow {a.y0}-{a.y1}   up to {a.basins} cohort basins per product\n")
    hdr = "product         n   " + "".join(f"{k:>8}" for k in LAGS) + "   peak  mode  %@0   %neg"
    print(hdr); print("-" * len(hdr))
    verdicts = []
    for spec in a.products.split(","):
        name, cdir = spec.split("=", 1)
        r = product_peak(cdir, ids, a.y0, a.y1, a.basins)
        if r is None:
            print(f"{name:<15} no data"); continue
        out["products"][name] = r
        cells = "".join(f"{r['median_r_by_lag'][k]:>8.3f}" for k in LAGS)
        print(f"{name:<15}{r['n_basins']:>3} {cells}   {r['peak_lag_of_median']:>4}  "
              f"{r['modal_peak_lag']:>4}  {r['frac_basins_peaking_at_0']:>.2f}  "
              f"{r['frac_basins_peaking_negative']:>.2f}")
        verdicts.append((name, r["peak_lag_of_median"], r["modal_peak_lag"],
                         r["frac_basins_peaking_negative"]))
    print()
    future = [v for v in verdicts if v[1] > 0 or v[2] > 0]
    stale = [v for v in verdicts if (v[1] < 0 or v[2] < 0) and v not in future]
    if future:
        print("⛔ MISREGISTERED — a POSITIVE peak lag means row t holds rain that falls LATER,")
        print("   so a decoder reading the target-day row receives future rain and loses the")
        print("   causal driver. Shift the precip column +1 row per lag, as Daymet was:")
        for v in future:
            print(f"    {v[0]}: peak-of-median k={v[1]}, modal k={v[2]}")
    if stale:
        print("⚠️ STALE (conservative, not future information):")
        for v in stale:
            print(f"    {v[0]}: peak-of-median k={v[1]}, modal k={v[2]}")
    if not future and not stale:
        print("✅ ALL PRODUCTS REGISTERED AT k=0.")
    out["verdict"] = "MISREGISTERED" if future else ("STALE" if stale else "CLEAN")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
