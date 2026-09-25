#!/usr/bin/env python
"""TRIANGULATE the one-day precip offset claim, using only on-disk data.

Claim under test: Daymet V4R1 labels precipitation a day earlier than CAMELS.
Risk: maybe CAMELS is the odd one out, having shifted precip during ITS
processing. If so, the claim's direction is wrong.

Test: CAMELS ships three forcings built from three DIFFERENT upstream products
(Daymet<-GHCN, NLDAS-2<-CPC, Maurer<-COOP). If all three agree with each other
at lag 0, and all three drive streamflow at k=0, then the CAMELS convention is
consistent across independent products and V4R1 is the outlier. If instead they
disagree among themselves, the whole framing is weaker than claimed.

No downloads: the frozen corpora all cover 1980-2014.
"""
from __future__ import annotations
import gzip, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2/gpu1080/corpora671")
FORC = {"daymet": ROOT / "camels_corpus_daymet_v2",
        "nldas": ROOT / "camels_corpus_nldas_v2",
        "maurer": ROOT / "camels_corpus_maurer_v2"}
NBASIN = int(sys.argv[1]) if len(sys.argv) > 1 else 50
LAGS = (-2, -1, 0, 1, 2)


def load(p):
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        d, P, Q = [], [], []
        for line in fh:
            f = line.strip().split(",")
            d.append(f[0])
            try:
                P.append(float(f[ci["precipitation_sum"]]))
            except (ValueError, IndexError):
                P.append(np.nan)
            try:
                q = float(f[ci["q_cfs"]])
                Q.append(q if q >= 0 else np.nan)
            except (ValueError, IndexError):
                Q.append(np.nan)
    return d, np.array(P), np.array(Q)


def lagcorr(a, b, lag):
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 300 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    common = None
    for name, d in FORC.items():
        ids = {p.name.split(".")[0] for p in d.glob("*.csv.gz")}
        common = ids if common is None else (common & ids)
    basins = sorted(common)[:NBASIN]
    print(f"triangulation on {len(basins)} basins, frozen corpora (1980-2014)\n")

    pairs = [("daymet", "nldas"), ("daymet", "maurer"), ("nldas", "maurer")]
    acc = {p: {l: [] for l in LAGS} for p in pairs}
    q_acc = {n: {l: [] for l in LAGS} for n in FORC}

    for b in basins:
        series = {}
        Qs = None
        ok = True
        for name, d in FORC.items():
            f = d / f"{b}.csv.gz"
            if not f.exists():
                ok = False; break
            dates, P, Q = load(f)
            series[name] = (dates, P)
            if Qs is None:
                Qs = (dates, Q)
        if not ok:
            continue
        # forcing-vs-forcing (align on the shorter record by date)
        for a, bb in pairs:
            da, Pa = series[a]; db, Pb = series[bb]
            idx = {x: i for i, x in enumerate(db)}
            keep = [(i, idx[x]) for i, x in enumerate(da) if x in idx]
            if len(keep) < 1000:
                continue
            va = np.array([Pa[i] for i, _ in keep])
            vb = np.array([Pb[j] for _, j in keep])
            for l in LAGS:
                r = lagcorr(va, vb, l)
                if not math.isnan(r):
                    acc[(a, bb)][l].append(r)
        # forcing-vs-streamflow
        dq, Q = Qs
        if np.isfinite(Q).sum() < 1000:
            continue
        dQ = np.full_like(Q, np.nan); dQ[1:] = Q[1:] - Q[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = dQ / np.where(Q > 0, Q, np.nan)
        for name in FORC:
            dn, Pn = series[name]
            idx = {x: i for i, x in enumerate(dq)}
            keep = [(i, idx[x]) for i, x in enumerate(dn) if x in idx]
            if len(keep) < 1000:
                continue
            vp = np.array([Pn[i] for i, _ in keep])
            vq = np.array([rel[j] for _, j in keep])
            for l in LAGS:
                r = lagcorr(vp, vq, l)
                if not math.isnan(r):
                    q_acc[name][l].append(r)

    print("=== FORCING vs FORCING (precip), median r by lag ===")
    allzero = True
    for p in pairs:
        med = {l: float(np.median(v)) for l, v in acc[p].items() if v}
        if not med:
            continue
        best = max(med, key=lambda l: med[l])
        allzero &= (best == 0)
        print(f"  {p[0]:>6} vs {p[1]:<6} " +
              "  ".join(f"lag{l:+d}={med[l]:+.3f}" for l in sorted(med)) +
              f"   -> best {best:+d}")

    print("\n=== FORCING vs relative dQ (physics), median r by lag ===")
    allk0 = True
    for name in FORC:
        med = {l: float(np.median(v)) for l, v in q_acc[name].items() if v}
        if not med:
            continue
        best = max(med, key=lambda l: med[l])
        allk0 &= (best == 0)
        print(f"  {name:>6}        " +
              "  ".join(f"k{l:+d}={med[l]:+.4f}" for l in sorted(med)) +
              f"   -> peak {best:+d}")

    print("\n--- VERDICT ---")
    print(f"  all three CAMELS forcings mutually aligned at lag 0 : {allzero}")
    print(f"  all three drive streamflow at k=0                   : {allk0}")
    if allzero and allk0:
        print("  => CAMELS convention is consistent across THREE independent")
        print("     upstream products; Daymet V4R1 is the outlier. Claim HOLDS.")
    else:
        print("  => ⚠️ CAMELS is NOT internally consistent - the one-day-offset")
        print("     framing is weaker than claimed and must be restated.")


if __name__ == "__main__":
    main()
