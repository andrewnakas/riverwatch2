#!/usr/bin/env python3
"""Compute each NH member dump's standalone median NSE and flag whether it lands
in the paper's expected single-member range (Li/Shen 2025 Table D1: single-forcing
LSTM ~0.735 avg, per-forcing scatter). Catches a bad member early instead of after
a 27h run. Run over all dumps in a dir; prints a table + PASS/CHECK per member."""
from __future__ import annotations
import glob
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# expected single-member median-NSE band per forcing (from our own runs + the paper)
EXPECT = {
    "daymet": (0.72, 0.78),   # strong forcing; NH gate gave 0.750
    "nldas":  (0.70, 0.75),   # NH gate gave 0.723
    "maurer": (0.68, 0.74),   # weakest; ~0.71-0.73
}


def nse_median(path: str) -> tuple[float, int]:
    d = pd.read_csv(path, compression="gzip")
    def _nse(g):
        t, p = g["truth"].to_numpy(float), g["ymed"].to_numpy(float)
        m = np.isfinite(t) & np.isfinite(p)
        if m.sum() < 50:
            return np.nan
        t, p = t[m], p[m]
        den = ((t - t.mean()) ** 2).sum()
        return 1 - ((t - p) ** 2).sum() / den if den > 0 else np.nan
    per = d.groupby("station_id").apply(_nse, include_groups=False).dropna()
    return float(per.median()), int(len(per))


def main() -> int:
    dumps_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mblstm/gpu_dumps_s14"
    files = sorted(glob.glob(f"{dumps_dir}/camels531_*_nhlstm_s*.csv.gz"))
    files = [f for f in files if re.search(r"_s(1[123]|2[123]|3[123])\d?\.csv", f)
             or re.search(r"_s(111|222|333)\.csv", f)]
    if not files:
        print(f"no NH member dumps in {dumps_dir}")
        return 0
    print(f"{'member':40s} {'medNSE':>7s} {'basins':>7s}  verdict")
    print("-" * 70)
    for f in files:
        name = Path(f).name
        m = re.search(r"camels531_([a-z]+)_nhlstm_s(\d+)", name)
        forcing = m.group(1) if m else "?"
        med, n = nse_median(f)
        lo, hi = EXPECT.get(forcing, (0.65, 0.80))
        verdict = "PASS" if lo <= med <= hi else ("HIGH?" if med > hi else "LOW — CHECK")
        print(f"{name:40s} {med:7.4f} {n:7d}  [{lo:.2f}-{hi:.2f}] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
