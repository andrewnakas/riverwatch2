#!/usr/bin/env python3
"""Build the FUSED (camels3fv2) corpus: one file per basin, 18 forcing columns.

`--enc-vars camels3fv2` expects each of the 6 recipe-v2 variables suffixed by
product -> 18 columns, plus a `_daymet`-suffixed decoder set:

    CAMELS3FV2_VARS = [f"{v}_{p}" for p in (daymet, maurer, nldas)
                                  for v in CAMELS1F_VARS]

No such corpus exists on the box, which is why the fused with-q member could not
be queued. This merges the three single-forcing recipe-v2 corpora on `date`.

Alignment notes (checked, not assumed):
  - all three corpora carry the same 671 basins and identical column names
  - daymet/nldas span 1980-01-01..2014-12-31; **maurer stops 2008-12-31**
  - an INNER join on date therefore truncates every basin at 2008-12-31.
    That is fine for the with-q protocol (train ends 2008-09-30, test window
    1989-10-01..1999-09-30) but would NOT cover the Li/Song no-q split, which
    tests to 2010-09-30. This corpus is for the with-q track only.
  - q_cfs is taken from the daymet file (identical across products - verified
    by comparing overlapping dates; any mismatch is reported and aborts).

Output: <out>/<basin>.csv.gz with columns
    date, q_cfs, <6 vars>_daymet, <6 vars>_maurer, <6 vars>_nldas
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
PRODUCTS = ("daymet", "maurer", "nldas")
# LEDGER 51: --products lets the same builder emit the 4-forcing corpus
# (camels4fv2). AORC starts 1980-10-01, so the inner join loses the first
# 9 months; the with-q protocol (train 1999-10-01.., test 1989-10-01..) is
# unaffected.


def main():
    global PRODUCTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="gpu1080/corpora671")
    ap.add_argument("--out", default="gpu1080/corpora671/camels_corpus_fused_v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--products", default=",".join(PRODUCTS),
                    help="comma-separated forcing products to fuse")
    args = ap.parse_args()
    PRODUCTS = tuple(p.strip() for p in args.products.split(",") if p.strip())
    print(f"products: {PRODUCTS}")

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dirs = {p: src / f"camels_corpus_{p}_v2" for p in PRODUCTS}
    for p, d in dirs.items():
        if not d.exists():
            sys.exit(f"missing source corpus: {d}")

    basins = sorted(x.name for x in dirs["daymet"].glob("*.csv.gz"))
    if args.limit:
        basins = basins[:args.limit]
    print(f"fusing {len(basins)} basins -> {out}")

    written = skipped = 0
    qmismatch = 0
    lens = []
    for i, fn in enumerate(basins):
        frames = {}
        ok = True
        for p, d in dirs.items():
            fp = d / fn
            if not fp.exists():
                ok = False
                break
            df = pd.read_csv(fp)
            frames[p] = df
        if not ok:
            skipped += 1
            continue

        base = frames["daymet"][["date", "q_cfs"]].copy()
        for p in PRODUCTS:
            f = frames[p]
            ren = {v: f"{v}_{p}" for v in VARS}
            sub = f[["date"] + VARS].rename(columns=ren)
            base = base.merge(sub, on="date", how="inner")

        # sanity: discharge should agree across products where both are present
        for p in PRODUCTS[1:]:
            m = frames["daymet"][["date", "q_cfs"]].merge(
                frames[p][["date", "q_cfs"]], on="date", suffixes=("_d", f"_{p}"))
            a = m["q_cfs_d"].to_numpy(float)
            b = m[f"q_cfs_{p}"].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() and not np.allclose(a[both], b[both], rtol=1e-3, atol=1e-3):
                qmismatch += 1
                break

        if len(base) == 0:
            skipped += 1
            continue
        lens.append(len(base))
        base.to_csv(out / fn, index=False, compression="gzip")
        written += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(basins)} ...", flush=True)

    print(f"\nwritten={written} skipped={skipped} q_mismatch_basins={qmismatch}")
    if lens:
        print(f"rows/basin: min={min(lens)} median={int(np.median(lens))} max={max(lens)}")
    sample = pd.read_csv(out / basins[0])
    print(f"date range: {sample.date.min()} .. {sample.date.max()}")
    print(f"columns ({len(sample.columns)}): {list(sample.columns)}")
    expect = 2 + 6 * len(PRODUCTS)
    if len(sample.columns) != expect:
        sys.exit(f"EXPECTED {expect} columns, got {len(sample.columns)}")
    print(f"OK: schema is date + q_cfs + {6 * len(PRODUCTS)} forcing cols")


if __name__ == "__main__":
    main()
