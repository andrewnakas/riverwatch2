#!/usr/bin/env python
"""REGISTRATION AUDIT of the FROZEN training corpus — the technique that found
a +0.229 NSE error in the modern corpus, applied to the data the 0.84 push uses.

A one-day date-label error on precipitation was worth +0.2292 NSE in the modern
window, and it was invisible to every mean/bias/correlation check because those
are shift-invariant. Only a lag scan finds them. So: lag-scan EVERY dynamic
channel of the frozen corpus against observed streamflow and check each peaks
where physics says it should.

Expected peaks (relative flow rise dQ[t+k] vs channel[t]):
  precipitation      -> k = 0   (rain drives the same-day rise)
  swe_delta          -> k = 0   (melt/accumulation is same-day)
  temperature        -> weak/seasonal, no sharp peak expected (reported, not judged)
  radiation, vp      -> weak, anti-correlated with storms
  soil moisture      -> k <= 0 (state precedes response)

Anything with a SHARP peak at k != 0 is a registration candidate and is worth
pricing, because the modern experiment showed what that costs.
No GPU, on-disk data only.
"""
from __future__ import annotations
import gzip, math, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2")
CORPUS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "gpu1080/corpora671/camels_corpus_fused_v2"
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 120
LAGS = (-2, -1, 0, 1, 2)


def load(p):
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        rows = [l.rstrip("\n").split(",") for l in fh if l.strip()]
    n = len(rows)
    out = {}
    for c, i in ci.items():
        if c == "date":
            continue
        v = np.empty(n)
        for k, r in enumerate(rows):
            try:
                v[k] = float(r[i])
            except (ValueError, IndexError):
                v[k] = np.nan
        out[c] = v
    return out


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
    files = sorted(CORPUS.glob("*.csv.gz"))[:NB]
    if not files:
        sys.exit(f"no files in {CORPUS}")
    print(f"registration audit: {CORPUS.name}, {len(files)} basins\n")

    peaks, strength = {}, {}
    for f in files:
        d = load(f)
        if "q_cfs" not in d:
            continue
        Q = d["q_cfs"].copy()
        Q[Q < 0] = np.nan
        if np.isfinite(Q).sum() < 1000:
            continue
        dQ = np.full_like(Q, np.nan)
        dQ[1:] = Q[1:] - Q[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = dQ / np.where(Q > 0, Q, np.nan)
        for c, v in d.items():
            if c == "q_cfs":
                continue
            cs = {l: lagcorr(v, rel, l) for l in LAGS}
            cs = {l: x for l, x in cs.items() if not math.isnan(x)}
            if not cs:
                continue
            # ⚠️ MUST be sign-aware, NOT abs(). After a storm the recession makes
            # dQ strongly NEGATIVE at k=+2, so |corr| picks k=+2 for a perfectly
            # registered precipitation channel — an earlier version of this audit
            # did exactly that and nearly produced a false "fusion bug" finding.
            # Precip/soil-moisture drive the rise (want the MAXIMUM); radiation is
            # suppressed by storms (want the MINIMUM).
            neg = ("shortwave" in c) or ("srad" in c)
            bl = min(cs, key=lambda l: cs[l]) if neg else max(cs, key=lambda l: cs[l])
            peaks.setdefault(c, Counter())[bl] += 1
            strength.setdefault(c, []).append(abs(cs[bl]))

    print(f"{'channel':<28} {'k=-2':>5}{'k=-1':>6}{'k=0':>6}{'k=+1':>6}{'k=+2':>6}"
          f"   {'@0%':>6} {'|r|':>7}   flag")
    print("-" * 88)
    rows = []
    for c in sorted(peaks):
        cnt = peaks[c]
        tot = sum(cnt.values())
        at0 = cnt.get(0, 0) / tot * 100
        med = float(np.median(strength[c]))
        flag = ""
        if med >= 0.10:                       # only judge channels with real signal
            if at0 < 60:
                dom = max(cnt, key=lambda l: cnt[l])
                flag = f"⚠️ peaks k={dom:+d} in {cnt[dom]/tot*100:.0f}%"
            elif at0 < 85:
                flag = "· partial"
        else:
            flag = "(weak signal — not judged)"
        print(f"{c:<28} " + "".join(f"{cnt.get(l,0):>6}" for l in LAGS) +
              f"   {at0:5.1f}% {med:7.3f}   {flag}")
        rows.append((c, at0, med, flag))

    print("\n--- READ ---")
    bad = [r for r in rows if r[3].startswith("⚠️")]
    if bad:
        print("  Registration candidates (sharp peak away from k=0):")
        for c, at0, med, flag in bad:
            print(f"    {c:<28} only {at0:.0f}% at k=0, {flag}")
        print("  ⇒ price these: a 1-day precip offset was worth +0.229 NSE.")
    else:
        print("  ✅ every channel with real signal peaks at k=0 in >=60% of basins.")
    print("  ⚠️ Channels with |r| < 0.10 carry too little timing signal to judge")
    print("     this way; absence of a flag there is NOT evidence of correctness.")


if __name__ == "__main__":
    main()
