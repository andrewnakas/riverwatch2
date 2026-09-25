#!/usr/bin/env python
"""Is CAMELS' Maurer precipitation genuinely SHIFTED, or just SMEARED?

Pairwise correlation said Maurer sits at lag -1 vs Daymet in 27/30 basins, but
the physics test (precip vs same-day relative flow rise) said all three forcings
peak at k=0. Those cannot both mean "Maurer is a day late".

Distinguishing test, per basin rather than in aggregate:
  * SHIFTED  -> Maurer's precip-vs-dQ peak sits at a DIFFERENT lag than Daymet's
                in the SAME basin, consistently.
  * SMEARED  -> Maurer peaks at the same lag but with a LOWER peak and fatter
                shoulders (a broader interpolation kernel, not a time offset).

Only the first is worth GPU. Uses on-disk frozen corpora, no downloads.
"""
from __future__ import annotations
import gzip, math
from collections import Counter
from pathlib import Path
import numpy as np

R = Path("/home/nakas/riverwatch2/gpu1080/corpora671")
F = {"daymet": R / "camels_corpus_daymet_v2",
     "nldas": R / "camels_corpus_nldas_v2",
     "maurer": R / "camels_corpus_maurer_v2"}
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
    if len(x) < 500 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    common = None
    for n, d in F.items():
        s = {p.name.split(".")[0] for p in d.glob("*.csv.gz")}
        common = s if common is None else (common & s)
    basins = sorted(common)[:120]

    peak = {n: Counter() for n in F}
    peakval = {n: [] for n in F}
    shoulder = {n: [] for n in F}
    agree_with_daymet = Counter()

    for b in basins:
        series, Q0, dates0 = {}, None, None
        ok = True
        for n, d in F.items():
            p = d / f"{b}.csv.gz"
            if not p.exists():
                ok = False; break
            dates, P, Q = load(p)
            series[n] = (dates, P)
            if Q0 is None:
                Q0, dates0 = Q, dates
        if not ok or Q0 is None or np.isfinite(Q0).sum() < 1000:
            continue
        dQ = np.full_like(Q0, np.nan); dQ[1:] = Q0[1:] - Q0[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = dQ / np.where(Q0 > 0, Q0, np.nan)
        qidx = {d: i for i, d in enumerate(dates0)}

        best = {}
        for n in F:
            dn, Pn = series[n]
            keep = [(i, qidx[d]) for i, d in enumerate(dn) if d in qidx]
            if len(keep) < 1000:
                continue
            vp = np.array([Pn[i] for i, _ in keep])
            vq = np.array([rel[j] for _, j in keep])
            cs = {l: lagcorr(vp, vq, l) for l in LAGS}
            cs = {l: c for l, c in cs.items() if not math.isnan(c)}
            if not cs:
                continue
            bl = max(cs, key=lambda l: cs[l])
            best[n] = bl
            peak[n][bl] += 1
            peakval[n].append(cs[bl])
            sh = [cs[l] for l in cs if l != bl]
            shoulder[n].append(max(sh) / cs[bl] if sh and cs[bl] > 0 else np.nan)
        if "daymet" in best:
            for n in F:
                if n in best:
                    agree_with_daymet[(n, best[n] == best["daymet"])] += 1

    print(f"per-basin peak lag of corr(precip[t], relative dQ[t+k]),  n={len(peakval['daymet'])} basins\n")
    for n in F:
        c = peak[n]
        tot = sum(c.values())
        row = "  ".join(f"k{l:+d}={c.get(l,0):3d}" for l in LAGS)
        print(f"  {n:<7} {row}    peak-at-0: {c.get(0,0)/tot*100:5.1f}%")
    print()
    print("  median peak height (higher = sharper timing signal):")
    for n in F:
        v = [x for x in peakval[n] if not math.isnan(x)]
        s = [x for x in shoulder[n] if not math.isnan(x)]
        print(f"    {n:<7} peak r = {np.median(v):.4f}   "
              f"next-best/peak ratio = {np.median(s):.3f}")
    print()
    print("  agreement with daymet's peak lag, same basin:")
    for n in F:
        y = agree_with_daymet[(n, True)]; nn = agree_with_daymet[(n, False)]
        if y + nn:
            print(f"    {n:<7} same lag as daymet: {y}/{y+nn} ({y/(y+nn)*100:.0f}%)")

    print("\n--- INTERPRETATION ---")
    dm = peak["daymet"].get(0, 0) / max(sum(peak["daymet"].values()), 1)
    mm = peak["maurer"].get(0, 0) / max(sum(peak["maurer"].values()), 1)
    if mm < dm - 0.20:
        print("  ⇒ Maurer peaks at a different lag far more often than daymet:")
        print("    consistent with a genuine TIME SHIFT. Worth pricing further.")
    else:
        print("  ⇒ Maurer peaks at the SAME lag as daymet in most basins.")
        print("    The pairwise lag-1 result reflects a BROADER kernel (smearing),")
        print("    not a time offset. ⛔ Not a lever — do not spend GPU aligning it.")


if __name__ == "__main__":
    main()
