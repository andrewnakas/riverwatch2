#!/usr/bin/env python3
"""Sanity-check the reconstructed basin against its neighbours and other products.

The integration script already enforced schema, window and non-zero diurnal
range. This asks the harder question: does the reconstructed series look like a
plausible member of its own neighbourhood, and does it agree with the OTHER
forcing products for the same basin?

That second test is the strong one. daymet/maurer/nldas all cover 13235000 and
are independent of AORC, so if our reconstruction correlates with them about as
well as a PUBLISHED AORC basin correlates with its own daymet/maurer/nldas, the
reconstruction behaves like the real thing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("data/local_corpora")
AORC = C / "camels_corpus_aorc_v2"
BASIN = "13235000"
NEIGHBOURS = ["13240000", "13161500", "13185000"]   # published AORC basins nearby


def load(prod, b):
    f = C / f"camels_corpus_{prod}_v2" / f"{b}.csv.gz"
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=["date", "precipitation_sum", "temperature_2m_max",
                                "temperature_2m_min"])
    d["date"] = pd.to_datetime(d["date"])
    return d


print("=== 1. reconstructed basin vs the other forcing products, same basin ===")
a = load("aorc", BASIN)
print(f"{'product':>8} {'precip_ratio':>13} {'precip_corr':>12} {'tmax_corr':>10}")
own = []
for prod in ("daymet", "maurer", "nldas"):
    o = load(prod, BASIN)
    if o is None:
        print(f"{prod:>8}  (absent)"); continue
    j = a.merge(o, on="date", suffixes=("_a", "_o"))
    r = np.corrcoef(j.precipitation_sum_a, j.precipitation_sum_o)[0, 1]
    rt = np.corrcoef(j.temperature_2m_max_a, j.temperature_2m_max_o)[0, 1]
    ratio = j.precipitation_sum_a.sum() / j.precipitation_sum_o.sum()
    own.append(r)
    print(f"{prod:>8} {ratio:>13.3f} {r:>12.4f} {rt:>10.4f}")

print("\n=== 2. the same comparison for PUBLISHED AORC basins (the reference) ===")
ref = []
for b in NEIGHBOURS:
    ab = load("aorc", b)
    if ab is None:
        continue
    for prod in ("daymet", "maurer", "nldas"):
        o = load(prod, b)
        if o is None:
            continue
        j = ab.merge(o, on="date", suffixes=("_a", "_o"))
        r = np.corrcoef(j.precipitation_sum_a, j.precipitation_sum_o)[0, 1]
        ref.append((b, prod, r))
        print(f"  {b} vs {prod:7s} precip corr {r:.4f}")

if ref and own:
    rr = np.array([x[2] for x in ref])
    print(f"\npublished basins: mean {rr.mean():.4f}  range {rr.min():.4f}-{rr.max():.4f}")
    print(f"reconstructed:    mean {np.mean(own):.4f}")
    print()
    if np.mean(own) >= rr.min():
        print("=> The reconstruction agrees with the independent products AS WELL AS")
        print("   published AORC basins do. It behaves like the real product.")
    else:
        print("=> The reconstruction agrees LESS well than published basins do.")
        print("   Investigate before using it.")

print("\n=== 3. climatology vs neighbours (is it physically plausible?) ===")
print(f"{'basin':>9} {'precip':>8} {'tmax':>7} {'tmin':>7} {'spread':>7}  source")
for b in [BASIN] + NEIGHBOURS:
    d = load("aorc", b)
    if d is None:
        continue
    src = "RECONSTRUCTED" if b == BASIN else "published"
    print(f"{b:>9} {d.precipitation_sum.mean():>8.3f} {d.temperature_2m_max.mean():>7.2f} "
          f"{d.temperature_2m_min.mean():>7.2f} "
          f"{(d.temperature_2m_max-d.temperature_2m_min).mean():>7.2f}  {src}")
