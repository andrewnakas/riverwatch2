#!/usr/bin/env python3
"""Can AORC's 2002 inhomogeneity be REMOVED by relative homogenization?

Established: AORC switches precip source at 2002 (Stage II/CMORPH -> Stage IV),
and its agreement with peers drops -0.0429 across that boundary while peer-only
pairs drop -0.0042. It fits train better than daymet (0.00833 vs 0.00995) yet
scores worst (0.7009 vs 0.7542) -- classic train->test transfer failure.

Standard climatology answer: RELATIVE HOMOGENIZATION (Pettitt / SNHT for
detection, ratio adjustment against a composite reference). The candidate series
is compared to reference series that share the climate signal but not the
artificial break. We have three natural references -- daymet, maurer, nldas --
none of which changes at 2002.

The question this answers BEFORE any GPU is spent: is AORC's break a
correctable LEVEL/SCALE shift (fixable by a ratio adjustment) or a change in the
day-to-day CORRELATION STRUCTURE (not fixable, because no multiplicative factor
restores a lost temporal relationship)?

  - If the ratio AORC/reference shifts at 2002 but daily correlation is stable,
    a per-basin ratio adjustment fixes it and retraining is worth it.
  - If daily correlation itself drops, the product is reporting DIFFERENT
    day-to-day rainfall, and no adjustment can recover the training-period
    relationship. Retraining would waste ~9 GPU-hours.

Refs: relative homogenization / SNHT / Pettitt -- Coll et al. 2020 (Int J
Climatol 40, doi:10.1002/joc.6575); Climatol (Guijarro).
"""
import glob
import os

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
C = "data/local_corpora"
PEERS = ["daymet", "maurer", "nldas"]
BREAK = "2002-01-01"
E1, E2 = ("1996-01-01", "2001-12-31"), ("2002-01-01", "2008-12-31")


def series(prod, b):
    f = f"{C}/camels_corpus_{prod}_v2/{b}.csv.gz"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, usecols=["date", "precipitation_sum"])
    d["date"] = pd.to_datetime(d.date)
    return d.set_index("date").precipitation_sum


def main():
    basins = sorted(os.path.basename(f)[:8]
                    for f in glob.glob(f"{C}/camels_corpus_aorc_v2/*.csv.gz"))[:150]
    rows = []
    for b in basins:
        a = series("aorc", b)
        if a is None:
            continue
        refs = {p: series(p, b) for p in PEERS}
        if any(v is None for v in refs.values()):
            continue
        M = pd.DataFrame({"aorc": a, **refs}).dropna()
        if len(M) < 3000:
            continue
        M["ref"] = M[PEERS].mean(axis=1)     # composite reference
        e, l = M.loc[E1[0]:E1[1]], M.loc[E2[0]:E2[1]]
        if len(e) < 800 or len(l) < 800:
            continue
        # LEVEL shift: ratio of totals vs the reference
        r_e = e.aorc.sum() / max(e.ref.sum(), 1e-9)
        r_l = l.aorc.sum() / max(l.ref.sum(), 1e-9)
        # STRUCTURE shift: daily correlation with the reference
        c_e = np.corrcoef(e.aorc, e.ref)[0, 1]
        c_l = np.corrcoef(l.aorc, l.ref)[0, 1]
        rows.append(dict(basin=b, ratio_e=r_e, ratio_l=r_l, d_ratio=r_l - r_e,
                         corr_e=c_e, corr_l=c_l, d_corr=c_l - c_e))
    T = pd.DataFrame(rows)
    print(f"basins: {len(T)}   break tested at {BREAK}\n")

    print("=== IS IT A LEVEL SHIFT? (ratio to composite reference) ===")
    print(f"  1996-2001 ratio : {T.ratio_e.mean():.4f}")
    print(f"  2002-2008 ratio : {T.ratio_l.mean():.4f}")
    print(f"  change          : {T.d_ratio.mean():+.4f}  "
          f"(sd across basins {T.d_ratio.std():.4f})")
    print()
    print("=== IS IT A STRUCTURE SHIFT? (daily corr with reference) ===")
    print(f"  1996-2001 corr  : {T.corr_e.mean():.4f}")
    print(f"  2002-2008 corr  : {T.corr_l.mean():.4f}")
    print(f"  change          : {T.d_corr.mean():+.4f}  "
          f"(sd {T.d_corr.std():.4f})")
    print()
    lvl, str_ = abs(T.d_ratio.mean()), abs(T.d_corr.mean())
    print("=== VERDICT ===")
    if str_ > 0.02:
        print(f"  Daily correlation with the reference changed by {T.d_corr.mean():+.4f}.")
        print("  AORC reports DIFFERENT day-to-day rainfall after 2002, not merely")
        print("  a rescaled version of the same rainfall.")
        print("  => A ratio/quantile adjustment CANNOT fix this. Homogenization")
        print("     restores levels, not lost temporal correspondence.")
        print("     Retraining on a level-adjusted AORC would NOT recover skill.")
    elif lvl > 0.02:
        print(f"  Level shifted {T.d_ratio.mean():+.4f} while daily structure held")
        print(f"  ({T.d_corr.mean():+.4f}). This IS the correctable case:")
        print("  a per-basin, per-era ratio adjustment fitted on the TRAIN period")
        print("  would remove the break. Retraining is worth ~9 GPU-hours.")
    else:
        print("  Neither level nor structure moved much at this boundary --")
        print("  re-examine whether 2002 is the right breakpoint.")


if __name__ == "__main__":
    main()
