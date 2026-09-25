#!/usr/bin/env python
"""ARBITRATE the 1-day precipitation labeling question against PHYSICS.

Daymet V4R1 precipitation matches CAMELS' CIDA-era Daymet only at lag +1
(r 0.94-0.99), i.e. the same data labeled a day apart. Which labeling is right?

USGS daily discharge is the arbiter: its dates are unambiguous, and rain must
CAUSE the streamflow rise, never follow it. So correlate precipitation against
the same-day flow increase dQ[t] = Q[t] - Q[t-1] at several lags.

  corr(P[t], dQ[t+k]) peaking at k = 0 or k > 0  -> precipitation is correctly
      labeled (rain then rise, or rain and rise same day)
  peaking at k < 0  -> the flow rises BEFORE the rain, which is impossible and
      means precipitation is labeled one day LATE (a look-ahead).

Uses only data already on disk.
"""
from __future__ import annotations
import gzip, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2")
CORPORA = {
    "FROZEN (CAMELS CIDA V2/V3)": ROOT / "gpu1080/corpora671/camels_corpus_daymet_v2",
    "MODERN POINT (V4R1)": ROOT / "data/modern_corpus/camels_corpus_daymet_modern",
    "MODERN AREAL (V4R1 rebuild)": ROOT / "data/modern_corpus/camels_corpus_daymet_modern_areal",
}
NBASIN = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LAGS = (-2, -1, 0, 1, 2)


def load(p):
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        P, Q = [], []
        for line in fh:
            f = line.strip().split(",")
            try:
                P.append(float(f[ci["precipitation_sum"]]))
            except (ValueError, IndexError):
                P.append(np.nan)
            try:
                q = float(f[ci["q_cfs"]])
                Q.append(q if q >= 0 else np.nan)   # mask -999999 sentinels
            except (ValueError, IndexError):
                Q.append(np.nan)
    return np.array(P), np.array(Q)


def lagcorr(a, b, lag):
    """corr(a[t], b[t+lag])"""
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 200 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    for label, d in CORPORA.items():
        files = sorted(d.glob("*.csv.gz"))
        if not files:
            print(f"\n{label}: NO FILES"); continue
        files = files[:NBASIN]
        acc = {k: [] for k in LAGS}
        used = 0
        for f in files:
            P, Q = load(f)
            if np.isfinite(Q).sum() < 500:
                continue
            dQ = np.full_like(Q, np.nan)
            dQ[1:] = Q[1:] - Q[:-1]
            # relative rise: keeps basins comparable regardless of size
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = dQ / np.where(Q > 0, Q, np.nan)
            ok = False
            for k in LAGS:
                r = lagcorr(P, rel, k)
                if not math.isnan(r):
                    acc[k].append(r); ok = True
            used += ok
        med = {k: float(np.median(v)) for k, v in acc.items() if v}
        if not med:
            print(f"\n{label}: insufficient data"); continue
        best = max(med, key=lambda k: med[k])
        print(f"\n=== {label}   (n={used} basins) ===")
        print("  corr(precip[t], relative dQ[t+k]):")
        print("   " + "  ".join(f"k{k:+d}={med[k]:+.4f}" for k in LAGS))
        verdict = ("CORRECT (rain causes same-day rise)" if best == 0 else
                   "rain LEADS the rise by {}d - plausible".format(best) if best > 0 else
                   "⚠️⚠️ FLOW RISES BEFORE THE RAIN - precip labeled LATE (look-ahead)")
        print(f"   peak at k={best:+d}  ->  {verdict}")


if __name__ == "__main__":
    main()
