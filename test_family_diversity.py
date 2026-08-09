#!/usr/bin/env python3
"""Does a DIFFERENT MODEL FAMILY actually decorrelate more than another LSTM?

The ceiling analysis found 70% of member error is shared across members and
concluded that passing 0.930 needs "a different model family, better target
data, or a genuinely different information source". The first of those is a
recommendation we have never tested, and we can: the no-q streams include three
LSTM forcings, an LSTMmulti, and three dHBV members, all already dumped.

dHBV is a differentiable conceptual hydrology model -- it routes water through
explicit storage rather than learning a black-box mapping. If model family is a
real axis of diversity, dHBV residuals should correlate LESS with LSTM residuals
than LSTM residuals do with each other. If they correlate the SAME, the
recommendation is wrong and the shared error is a property of the data or the
target, not of the architecture.

This matters because it decides where the remaining effort should go. It is also
a fair test: our own note records dHBV as "net negative" on own-skill, but that
was never measured on the DECORRELATION axis, which is what the ceiling analysis
actually cares about.

Compares three groups of pairwise residual correlations:
    LSTM  <-> LSTM   (within family)
    dHBV  <-> dHBV   (within family)
    LSTM  <-> dHBV   (ACROSS families)
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

L = "gpu1080/dumps"
H = "dhbv_dumps"
QUAR = ("_OLD", "_test_", "_bak", "_broken", "_TRAIN")


def load(path, subset, shift_days=0):
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ymed"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(subset)]
    d["t0"] = pd.to_datetime(d.t0)
    if shift_days:
        d["t0"] = d.t0 + pd.Timedelta(days=shift_days)
    return d.set_index(["station_id", "t0"])[["truth", "ymed"]]


def main():
    ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])

    members = {}
    for f in sorted(glob.glob(f"{L}/camels531ls_*_nhlstm_s*.csv.gz")):
        b = os.path.basename(f)
        if any(q in b for q in QUAR):
            continue
        members[("LSTM", b.split("_")[1] + "_" + b.split("_")[-1][:-7])] = (f, 0)
    for f in sorted(glob.glob(f"{H}/camels531ls_*_dhbv_s*.csv.gz")):
        b = os.path.basename(f)
        if any(q in b for q in QUAR):
            continue
        # dHBV dumps carry a t0 offset of one day relative to the LSTM grid
        members[("dHBV", b.split("_")[1] + "_" + b.split("_")[-1][:-7])] = (f, 1)

    if not members:
        sys.exit("no member dumps found")
    print(f"members found: {len(members)}")
    for (fam, name), (f, sh) in members.items():
        print(f"  {fam:5s} {name}")

    # residuals on a common grid
    res = {}
    common = None
    for key, (f, sh) in members.items():
        try:
            d = load(f, ids, sh)
        except Exception as e:
            print(f"  skip {key}: {str(e)[:50]}")
            continue
        r = (d.ymed - d.truth).rename("res")
        res[key] = r
        common = r.index if common is None else common.intersection(r.index)

    if common is None or len(common) == 0:
        sys.exit("no common (station, date) grid across members")
    print(f"\ncommon grid: {len(common)} rows, "
          f"{len(set(i[0] for i in common))} basins")

    R = pd.DataFrame({k: v.reindex(common) for k, v in res.items()}).dropna()
    print(f"after dropna: {len(R)} rows\n")

    groups = {"LSTM<->LSTM": [], "dHBV<->dHBV": [], "LSTM<->dHBV": []}
    for a, b in itertools.combinations(R.columns, 2):
        c = np.corrcoef(R[a], R[b])[0, 1]
        if a[0] == b[0]:
            groups[f"{a[0]}<->{a[0]}"].append(c)
        else:
            groups["LSTM<->dHBV"].append(c)

    print("=== pairwise residual correlation by family pairing ===")
    print(f"{'pairing':14s} {'n':>4} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}")
    summary = {}
    for k, v in groups.items():
        if not v:
            continue
        a = np.array(v)
        summary[k] = a.mean()
        print(f"{k:14s} {len(a):>4} {a.mean():>8.4f} {np.median(a):>8.4f} "
              f"{a.min():>8.4f} {a.max():>8.4f}")

    print()
    if "LSTM<->LSTM" in summary and "LSTM<->dHBV" in summary:
        within = summary["LSTM<->LSTM"]
        across = summary["LSTM<->dHBV"]
        print(f"within-LSTM  {within:.4f}")
        print(f"across-family {across:.4f}")
        print(f"difference    {across - within:+.4f}")
        print()
        if across < within - 0.05:
            print("=> MODEL FAMILY IS A REAL DIVERSITY AXIS. dHBV errs materially")
            print("   differently from an LSTM, so the ceiling analysis's first")
            print("   recommendation is supported and worth pursuing.")
        elif across < within:
            print("=> family helps only marginally. Real but small; a hybrid member")
            print("   would buy less than the ceiling analysis implies.")
        else:
            print("=> FAMILY IS NOT A DIVERSITY AXIS HERE. dHBV residuals correlate")
            print("   with LSTM residuals as much as LSTMs do with each other, so")
            print("   the shared error is a property of the DATA or the TARGET, not")
            print("   of the architecture. The recommendation should be retracted.")


if __name__ == "__main__":
    main()
