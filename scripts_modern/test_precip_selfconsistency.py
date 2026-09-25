#!/usr/bin/env python
"""WHICH corpus has precipitation aligned to its OWN other variables?

We found precip agrees between Daymet V4R1 and CAMELS' CIDA-era Daymet only at
LAG +1, while temperature agrees at lag 0 — so one of them labels the daily
precipitation accumulation a day differently.

Internal-consistency test, needing no external truth: on a rainy day the sky is
cloudy, so **shortwave radiation drops** and **vapor pressure rises**, SAME DAY.
Whichever corpus shows its strongest precip-vs-srad anticorrelation at lag 0 has
precipitation aligned with its own radiation field. The other is offset.

Uses only data already on disk - no downloads.
"""
from __future__ import annotations
import gzip, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2")
CORPORA = {
    "FROZEN (CAMELS CIDA V2/V3, areal)": ROOT / "gpu1080/corpora671/camels_corpus_daymet_v2",
    "MODERN POINT (V4R1, single-pixel)": ROOT / "data/modern_corpus/camels_corpus_daymet_modern",
    "MODERN AREAL (V4R1, rebuilt)": ROOT / "data/modern_corpus/camels_corpus_daymet_modern_areal",
}
NBASIN = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def load(p):
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        rows = [l.strip().split(",") for l in fh if l.strip()]
    out = {}
    for c in ("precipitation_sum", "shortwave_radiation_sum", "vapor_pressure",
              "temperature_2m_max"):
        if c not in ci:
            return None
        v = []
        for r in rows:
            try:
                v.append(float(r[ci[c]]))
            except (ValueError, IndexError):
                v.append(np.nan)
        out[c] = np.array(v)
    return out


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
    if len(x) < 100 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    for label, d in CORPORA.items():
        files = sorted(d.glob("*.csv.gz"))[:NBASIN]
        if not files:
            print(f"\n{label}: NO FILES"); continue
        acc = {lag: [] for lag in (-2, -1, 0, 1, 2)}
        accv = {lag: [] for lag in (-2, -1, 0, 1, 2)}
        for f in files:
            c = load(f)
            if c is None:
                continue
            p = c["precipitation_sum"]
            s = c["shortwave_radiation_sum"]
            v = c["vapor_pressure"]
            for lag in acc:
                r = lagcorr(p, s, lag)
                if not math.isnan(r):
                    acc[lag].append(r)
                r2 = lagcorr(p, v, lag)
                if not math.isnan(r2):
                    accv[lag].append(r2)
        print(f"\n=== {label}   (n={len(files)} basins) ===")
        med = {l: float(np.median(x)) for l, x in acc.items() if x}
        medv = {l: float(np.median(x)) for l, x in accv.items() if x}
        if med:
            s = "  ".join(f"lag{l:+d}={med[l]:+.3f}" for l in sorted(med))
            best = min(med, key=lambda l: med[l])          # most NEGATIVE
            print(f"  precip vs srad : {s}")
            print(f"      strongest anticorrelation at lag {best:+d}"
                  + ("   <-- self-consistent" if best == 0
                     else "   <-- ⚠️ PRECIP OFFSET BY {:+d} DAY".format(best)))
        if medv:
            s = "  ".join(f"lag{l:+d}={medv[l]:+.3f}" for l in sorted(medv))
            best = max(medv, key=lambda l: medv[l])        # most POSITIVE
            print(f"  precip vs vp   : {s}")
            print(f"      strongest positive at lag {best:+d}"
                  + ("   <-- self-consistent" if best == 0 else "   <-- ⚠️ offset"))


if __name__ == "__main__":
    main()
