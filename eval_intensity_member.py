#!/usr/bin/env python3
"""Compare the intensity member against plain AORC — overall AND on event days.

The intensity member fits the training window ~13% better at every matched
epoch. That is consistent with three different stories and discriminates none of
them:

  1. genuine signal      peak intensity predicts storm response  -> test improves
  2. extra capacity      5 more inputs = more ways to fit noise   -> test flat/worse
  3. redundant encoding  p_max_1h correlates 0.72 with the daily
                         sum, so part of the gain may re-encode
                         precipitation the model already had      -> test flat

This is the in-sample inflation trap already documented in this campaign (dHBV
scored 0.906 on train against 0.723 on test), so training loss is not the gate.

The discriminating test has two parts:

  A. test-period NSE on identical basins -- does it generalise at all?
  B. NSE and limb bias ON EVENT DAYS -- the mechanism claim is specifically
     that intensity helps where the model under-predicts storm peaks. If the
     overall number improves but event-day performance does not, the mechanism
     story is wrong even though the score went up, and we should say so.

Part B is the one that makes this more than score-chasing.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

L = "gpu1080/dumps"


def load(pattern, subset):
    fs = sorted(f for f in glob.glob(f"{L}/{pattern}")
                if "_TRAIN" not in f and not any(q in f for q in
                    ("_OLD", "_test_", "_bak", "_broken")))
    if not fs:
        return None, []
    frames = []
    for f in fs:
        d = pd.read_csv(f, usecols=["station_id", "t0", "h", "truth", "ymed"])
        d = d[d.h == 1].copy()
        d["station_id"] = d.station_id.astype(str).str.zfill(8)
        d = d[d.station_id.isin(subset)]
        frames.append(d.set_index(["station_id", "t0"])[["truth", "ymed"]])
    # seed-average
    acc = frames[0].copy()
    for f in frames[1:]:
        acc = acc.join(f, rsuffix="_x", how="inner")
        acc["ymed"] = (acc.ymed + acc.ymed_x) / 2
        acc = acc[["truth", "ymed"]]
    return acc.reset_index(), [os.path.basename(f) for f in fs]


def med_nse(df, col="ymed"):
    out = []
    for _, g in df.groupby("station_id"):
        y, p = g.truth.to_numpy(float), g[col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out.append(1 - np.mean((y - p) ** 2) / np.var(y))
    return (np.median(out) if out else np.nan), len(out)


def main():
    ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
    plain, fp = load("camels531ls_aorc_nhlstm_s*.csv.gz", ids)
    inten, fi = load("camels531ls_aorcx_nhlstm_s*.csv.gz", ids)
    if plain is None or inten is None:
        sys.exit(f"missing dumps — plain:{bool(plain)} intensity:{bool(inten)}")
    print(f"plain AORC members : {fp}")
    print(f"intensity members  : {fi}\n")

    m = plain.merge(inten, on=["station_id", "t0"], suffixes=("_p", "_x"))
    # truth must agree; if it does not the two runs are not comparable
    dt = (m.truth_p - m.truth_x).abs().max()
    print(f"max |truth difference| between the two dumps: {dt:.6f}")
    if dt > 1e-3:
        sys.exit("FATAL: the dumps disagree on observed discharge — not comparable")
    m["truth"] = m.truth_p
    print(f"aligned rows {len(m)}, basins {m.station_id.nunique()}\n")

    print("=== A. overall test-period skill, identical basins ===")
    for lab, col in (("plain AORC", "ymed_p"), ("+ intensity", "ymed_x")):
        v, n = med_nse(m, col)
        print(f"  {lab:14s} median NSE {v:.4f}  ({n} basins)")
    a, _ = med_nse(m, "ymed_p")
    b, _ = med_nse(m, "ymed_x")
    print(f"  delta {b-a:+.4f}")

    # bootstrap the delta over basins
    per = []
    for sid, g in m.groupby("station_id"):
        y = g.truth.to_numpy(float)
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        f = lambda c: 1 - np.mean((y - g[c].to_numpy(float)) ** 2) / np.var(y)
        per.append((f("ymed_p"), f("ymed_x")))
    per = np.array(per)
    rng = np.random.default_rng(0)
    d = np.array([np.median(per[i, 1]) - np.median(per[i, 0])
                  for i in (rng.integers(0, len(per), len(per)) for _ in range(2000))])
    print(f"  bootstrap 95% CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]"
          f"   P(delta>0) = {(d>0).mean():.3f}")

    print("\n=== B. THE MECHANISM TEST: event days ===")
    m["err_p"] = (m.ymed_p - m.truth) ** 2
    thr = m.err_p.quantile(0.99)
    ev, ord_ = m[m.err_p >= thr], m[m.err_p < thr]
    m2 = m.sort_values(["station_id", "t0"])
    m2["dq"] = m2.groupby("station_id").truth.diff()
    for lab, sub in (("event days (top 1% err)", m2[m2.err_p >= thr]),
                     ("ordinary days", m2[m2.err_p < thr])):
        r = sub[sub.dq > 0]
        print(f"  {lab:26s} n={len(sub):6d}")
        print(f"      rising-limb mean error   plain {(r.ymed_p-r.truth).mean():+10.1f}   "
              f"intensity {(r.ymed_x-r.truth).mean():+10.1f}")
        print(f"      RMSE                     plain "
              f"{np.sqrt(((sub.ymed_p-sub.truth)**2).mean()):10.1f}   "
              f"intensity {np.sqrt(((sub.ymed_x-sub.truth)**2).mean()):10.1f}")
    print("\n  The claim is that intensity helps where the model under-predicts")
    print("  storm peaks. If event-day rising-limb bias does NOT shrink, the")
    print("  mechanism story is wrong even if the overall number improved.")


if __name__ == "__main__":
    main()
