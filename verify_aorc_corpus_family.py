#!/usr/bin/env python3
"""Verify the AORC NH corpus is comparable to the corpora the other streams use.

The split error was a CONFIG mistake caught by comparing configs. But the same
class of error can live in the DATA: if gpu1080/nh_data/aorc were built over a
different date range, or with a different discharge convention, or on a
different basin set from nh_data/daymet, every member trained on it would again
be quietly incomparable — and no config check would notice.

So this compares the corpora directly, on the axes that would break a join or a
metric:

  1. basin sets — identical, or the inner join silently shrinks
  2. date coverage — identical, or the test window is not what the config claims
  3. the DISCHARGE TARGET — q_mm must agree between corpora for the same basin,
     because it is the same gauge observation. If it does not, one of the two
     has a unit or area error and every NSE computed on it is wrong.
  4. forcing magnitudes — sane relative to daymet, not identical (AORC is a
     different product, so differences are expected; absurdities are not)

Check 3 is the important one. Forcings are allowed to differ between products;
the target is not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REF = Path("gpu1080/nh_data/daymet")
NEW = Path("gpu1080/nh_data/aorc")


def basins(d):
    return [b.strip().zfill(8) for b in (d / "basins.txt").read_text().split() if b.strip()]


def series(d, b):
    return xr.open_dataset(d / "time_series" / f"{b}.nc").to_dataframe()


def main():
    rb, nb = set(basins(REF)), set(basins(NEW))
    print("1. basin sets")
    print(f"   daymet {len(rb)}   aorc {len(nb)}   shared {len(rb & nb)}")
    only_r, only_n = sorted(rb - nb), sorted(nb - rb)
    if only_r or only_n:
        print(f"   daymet-only: {only_r[:5]}   aorc-only: {only_n[:5]}")
    fails = []
    if rb != nb:
        fails.append("basin sets differ")
    else:
        print("   OK identical")

    shared = sorted(rb & nb)
    probe = shared[:40]

    print("\n2. date coverage (40 basins)")
    bad = 0
    for b in probe:
        r, n = series(REF, b), series(NEW, b)
        if r.index.min() != n.index.min() or r.index.max() != n.index.max() \
           or len(r) != len(n):
            if bad < 3:
                print(f"   MISMATCH {b}: daymet {r.index.min().date()}..{r.index.max().date()}"
                      f" ({len(r)})  aorc {n.index.min().date()}..{n.index.max().date()} ({len(n)})")
            bad += 1
    if bad:
        fails.append(f"{bad}/{len(probe)} basins differ in date coverage")
        print(f"   FAIL {bad}/{len(probe)} basins differ")
    else:
        r = series(REF, probe[0])
        print(f"   OK all {len(probe)} match: {r.index.min().date()}..{r.index.max().date()}"
              f" ({len(r)} rows)")

    print("\n3. DISCHARGE TARGET agreement (the check that matters)")
    worst, worst_b = 0.0, None
    for b in probe:
        r, n = series(REF, b), series(NEW, b)
        j = r[["q_mm"]].join(n[["q_mm"]], rsuffix="_n", how="inner")
        ok = np.isfinite(j.q_mm) & np.isfinite(j.q_mm_n)
        if ok.sum() == 0:
            continue
        d = float(np.abs(j.q_mm[ok] - j.q_mm_n[ok]).max())
        if d > worst:
            worst, worst_b = d, b
    print(f"   max |q_mm difference| over {len(probe)} basins: {worst:.6g}  ({worst_b})")
    if worst > 1e-4:
        fails.append(f"discharge target differs by up to {worst:.4g}")
        print("   FAIL the target is not the same observation in both corpora")
    else:
        print("   OK the discharge target is identical, as it must be")

    print("\n4. forcing magnitudes vs daymet (differences expected, absurdity not)")
    rows = []
    for b in probe[:20]:
        r, n = series(REF, b), series(NEW, b)
        j = r.join(n, rsuffix="_n", how="inner")
        rows.append([j.prcp.mean(), j.prcp_n.mean(),
                     (j.tmax - j.tmin).mean(), (j.tmax_n - j.tmin_n).mean()])
    a = np.array(rows, float)
    print(f"   precip mean   daymet {a[:,0].mean():.3f}   aorc {a[:,1].mean():.3f}"
          f"   ratio {a[:,1].mean()/a[:,0].mean():.3f}")
    print(f"   diurnal range daymet {a[:,2].mean():.2f}    aorc {a[:,3].mean():.2f}")
    ratio = a[:, 1].mean() / a[:, 0].mean()
    if not (0.5 < ratio < 2.0):
        fails.append(f"precip ratio {ratio:.2f} is implausible")
        print("   FAIL precipitation differs by more than a factor of two")
    else:
        print("   OK within a plausible inter-product range")

    print()
    if fails:
        print(f"CORPUS CHECK FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("CORPUS CHECK PASSED — aorc is comparable to daymet")


if __name__ == "__main__":
    main()
