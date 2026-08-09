#!/usr/bin/env python3
"""Confirm or kill the phase hypothesis on a STRIDE-1 dump.

The shared-error decomposition says 88% of ensemble MSE is phase error, with
rising limbs under-predicted and falling limbs over-predicted. That is the
signature of a hydrograph arriving late -- but a decomposition cannot prove the
error is a genuine time shift rather than event-specific magnitude noise that
merely looks phase-like once projected onto (1-r).

Stride-1 gives consecutive days, so the model prediction can be shifted against
truth by a real one or two days. Three tests, from cheap to decisive:

  T1 GLOBAL LAG. Shift all predictions by +/-1, +/-2 days. If NSE improves at a
     nonzero shift, the ensemble has a uniform lag and the fix could be as cheap
     as a routing offset.
  T2 PER-BASIN LAG. Fit the best shift per basin and see how much it buys, and
     whether the optimal shift correlates with basin size or slope (bigger, flatter
     catchments should lag more if this is real routing).
  T3 EVENT-CONDITIONAL LAG. Restrict to the 1% of days that carry 93% of the
     error. If the lag is concentrated there, the problem is EVENT timing, and
     sub-daily precipitation intensity is the natural fix. If the lag is uniform
     across all days, it is a routing constant instead.

The three outcomes imply different work, which is the point of measuring rather
than assuming.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

DUMP = "data/mblstm/gpu_dumps_s14/phase_probe_stride1.csv.gz"


def nse(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def shifted(y, p, k):
    """Compare pred[t] against obs[t+k]; k>0 means the model was EARLY."""
    if k > 0:
        return y[k:], p[:-k]
    if k < 0:
        return y[:k], p[-k:]
    return y, p


def main():
    if not os.path.exists(DUMP):
        sys.exit(f"no stride-1 dump yet at {DUMP}")
    d = pd.read_csv(DUMP, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d["pred"] = (d.ylo + d.yhi) / 2
    d["t0"] = pd.to_datetime(d.t0)
    d = d.sort_values(["station_id", "t0"])
    print(f"stride-1 dump: {len(d)} rows, {d.station_id.nunique()} basins, "
          f"{d.t0.min().date()}..{d.t0.max().date()}")

    # confirm consecutive days
    gaps = d.groupby("station_id").t0.diff().dt.days.dropna()
    print(f"row spacing: median {gaps.median():.0f} day(s)  "
          f"({(gaps == 1).mean()*100:.0f}% are consecutive)\n")
    if gaps.median() != 1:
        print("WARNING: not stride-1; lag tests below are not interpretable\n")

    # ---- T1 global lag ------------------------------------------------------
    print("=== T1: global lag ===")
    print(f"{'shift':>7} {'median NSE':>12}")
    base = None
    for k in (-2, -1, 0, 1, 2):
        out = []
        for sid, g in d.groupby("station_id"):
            y, p = g.truth.to_numpy(float), g.pred.to_numpy(float)
            yy, pp = shifted(y, p, k)
            ok = np.isfinite(yy) & np.isfinite(pp)
            yy, pp = yy[ok], pp[ok]
            if len(yy) < 30 or np.var(yy) < 1e-9:
                continue
            out.append(nse(yy, pp))
        med = np.median(out)
        if k == 0:
            base = med
        print(f"{k:>7} {med:>12.4f}{'  <- current' if k == 0 else ''}")
    print("  shift<0 advances the model, i.e. treats it as LATE\n")

    # ---- T2 per-basin best lag ---------------------------------------------
    print("=== T2: per-basin best lag ===")
    rows = []
    for sid, g in d.groupby("station_id"):
        y, p = g.truth.to_numpy(float), g.pred.to_numpy(float)
        if len(y) < 60 or np.var(y) < 1e-9:
            continue
        best_k, best_v = 0, nse(y, p)
        for k in (-2, -1, 1, 2):
            yy, pp = shifted(y, p, k)
            ok = np.isfinite(yy) & np.isfinite(pp)
            if ok.sum() < 30 or np.var(yy[ok]) < 1e-9:
                continue
            v = nse(yy[ok], pp[ok])
            if v > best_v:
                best_k, best_v = k, v
        rows.append((sid, best_k, nse(y, p), best_v))
    r = pd.DataFrame(rows, columns=["sid", "k", "nse0", "nse_best"])
    print(f"  basins: {len(r)}")
    print(f"  optimal shift distribution: "
          f"{r.k.value_counts().sort_index().to_dict()}")
    print(f"  median NSE at shift 0    : {r.nse0.median():.4f}")
    print(f"  median NSE at best shift : {r.nse_best.median():.4f}  "
          f"({r.nse_best.median()-r.nse0.median():+.4f})")
    print("  NOTE: per-basin best shift is an ORACLE. It bounds what a perfect")
    print("        per-basin routing correction could buy, and would need to")
    print("        transfer from the train window to be deployable.\n")

    # ---- T3 event-conditional ----------------------------------------------
    print("=== T3: is the lag concentrated on high-error EVENT days? ===")
    d["err2"] = (d.pred - d.truth) ** 2
    thr = d.err2.quantile(0.99)
    for lab, sub in (("event days (top 1% err)", d[d.err2 >= thr]),
                     ("ordinary days", d[d.err2 < thr])):
        s = sub.sort_values(["station_id", "t0"])
        dq = s.groupby("station_id").truth.diff()
        rise = s[dq > 0]
        fall = s[dq < 0]
        print(f"  {lab:24s} n={len(s):6d}  "
              f"rising mean err {(rise.pred-rise.truth).mean():+9.1f}  "
              f"falling {(fall.pred-fall.truth).mean():+9.1f}")
    print("\n  A large rising/falling asymmetry ON EVENT DAYS ONLY implies event")
    print("  timing -> sub-daily precipitation intensity is the lever.")
    print("  A uniform asymmetry implies a routing constant instead.")


if __name__ == "__main__":
    main()
