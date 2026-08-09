#!/usr/bin/env python3
"""Score the with-q ensemble with a 5-SEED AORC member, on the full 531 basins.

Two things change against the 0.9196 record, and the script reports both
separately so they are not conflated:

  1. AORC goes from 4 seeds to 5, matching the other forcings. The measured
     4->5 gain on the other members was +0.011 to +0.017 each, which diluted
     across four members projects to about +0.003.
  2. The basin set goes from 530 to 531. The reconstructed basin 13235000 is now
     in the corpus, so the AORC member covers every benchmark basin and the
     "530 not 531" caveat on the record disappears.

Reporting them separately matters because the second is not a skill improvement
-- it is a coverage fix -- and a combined number would let a caveat removal look
like a gain. So this scores three configurations on a common grid:

    baseline 3-forcing, 5 seeds
    + AORC 4 seeds   (reproduces the 0.9196 record)
    + AORC 5 seeds   (the new candidate)

each on both the 530-basin intersection and the full 531, so the seed effect and
the coverage effect can be read independently.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

D = "data/mblstm/gpu_dumps_s14"


def load(path, subset):
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(subset)]
    d["pred"] = (d.ylo + d.yhi) / 2
    return d[["station_id", "t0", "truth", "pred"]]


def metrics(m, col):
    nse, kge, alpha = [], [], []
    for _, g in m.groupby("station_id"):
        y, p = g.truth.to_numpy(float), g[col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9 or np.std(p) < 1e-9:
            continue
        nse.append(1 - np.mean((y - p) ** 2) / np.var(y))
        r = np.corrcoef(y, p)[0, 1]
        a = np.std(p) / np.std(y)
        b = np.mean(p) / np.mean(y)
        kge.append(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))
        alpha.append(a)
    return (np.median(nse), np.median(kge), np.median(alpha), len(nse),
            np.array(nse))


def main():
    ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])

    p5 = f"{D}/camels531_aorc_withq5_full531.csv.gz"
    p4 = f"{D}/camels531_aorc_withq_full531.csv.gz"
    if not os.path.exists(p5):
        sys.exit(f"5-seed AORC dump not ready: {p5}")

    base = {}
    for f in ("daymet", "maurer", "nldas"):
        base[f] = load(f"{D}/camels531_{f}_withq5_full531.csv.gz", ids)
    a5 = load(p5, ids)
    a4 = load(p4, ids) if os.path.exists(p4) else None

    def merge(frames):
        m = None
        for name, d in frames.items():
            x = d.rename(columns={"pred": name})
            m = x if m is None else m.merge(x[["station_id", "t0", name]],
                                           on=["station_id", "t0"])
        return m

    # basin sets: the 4-seed dump predates the reconstructed basin
    s5 = set(a5.station_id)
    print(f"AORC 5-seed covers {len(s5)} of the 531 benchmark basins")
    if a4 is not None:
        s4 = set(a4.station_id)
        print(f"AORC 4-seed covers {len(s4)}  (missing: {sorted(s5 - s4) or 'none'})")
    print()

    rows = []
    for label, frames in (
            ("baseline 3 forcings x 5 seeds", dict(base)),
            ("+ AORC 4 seeds (the 0.9196 record)",
             dict(base, aorc=a4) if a4 is not None else None),
            ("+ AORC 5 seeds (candidate)", dict(base, aorc=a5)),
    ):
        if frames is None:
            continue
        m = merge(frames)
        cols = [c for c in frames]
        m["ens"] = m[cols].to_numpy(float).mean(1)
        n, k, al, nb, arr = metrics(m, "ens")
        rows.append((label, n, k, al, nb, arr))

    print(f"{'config':38s} {'medNSE':>8} {'medKGE':>8} {'alpha':>7} {'basins':>7}")
    for label, n, k, al, nb, _ in rows:
        print(f"{label:38s} {n:>8.4f} {k:>8.4f} {al:>7.4f} {nb:>7d}")

    # the seed effect, isolated on a COMMON basin set
    if len(rows) == 3:
        print()
        print("=== isolating the two effects ===")
        b, r4, r5 = rows[0], rows[1], rows[2]
        print(f"  AORC 4 seeds vs baseline : {r4[1]-b[1]:+.4f}  ({r4[4]} basins)")
        print(f"  AORC 5 seeds vs baseline : {r5[1]-b[1]:+.4f}  ({r5[4]} basins)")
        print(f"  5th seed contributes     : {r5[1]-r4[1]:+.4f}")
        if r5[4] != r4[4]:
            print(f"  NOTE basin counts differ ({r4[4]} vs {r5[4]}), so the 5th-seed")
            print("       figure above mixes a seed effect with a coverage change.")
            print("       The seed effect alone needs both scored on the same set.")

        rng = np.random.default_rng(0)
        arr = r5[5]
        bs = np.array([np.median(rng.choice(arr, len(arr))) for _ in range(2000)])
        print(f"\n  5-seed CI95 [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")
        print(f"  previous record 0.9196 (530 basins); Nearing 2022 0.879")


if __name__ == "__main__":
    main()
