#!/usr/bin/env python
"""MODERN-1 GATE 1 PART B: the streamflow referee, generalised to N products.

Same question and the SAME arithmetic as screen_v4_vs_camels_daymet.py, so the
numbers stay on one scale with the -0.0244 already measured for Daymet V4R1:

    peak_k corr( precip[t], relative dQ[t+k] )   for k in -2..+2

Observed streamflow is the referee, because USGS dates are unambiguous. Peak-over-lag
is used rather than a fixed lag, which makes the comparison REGISTRATION-INVARIANT:
it asks how much signal a forcing carries, not whether it is labelled the same way.
The peak lag is reported separately and is what sets each product's build-time shift.

Discharge and the day set come from the FROZEN corpus for every product, so the
only thing that varies between columns is the precipitation series itself.

BARS, fixed in PREREG_v2.md MODERN-1 before any of this ran:
    median peak r >= camels - 0.005   fund
    -0.02 .. -0.005                   build, but flagged
    < -0.02                           abandon
"""
from __future__ import annotations
import argparse, gzip, json, math, os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAGS = (-2, -1, 0, 1, 2)


def lagcorr(a, b, lag):                       # identical to the Daymet screen
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 400 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def load(path: Path, y0: int, y1: int, want_q: bool):
    if not path.exists():
        return None
    d, P, Q = [], [], []
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        if "precipitation_sum" not in ci:
            return None
        for line in fh:
            f = line.strip().split(",")
            y = int(f[0][:4])
            if y < y0 or y > y1:
                continue
            d.append(f[0])
            try:
                P.append(float(f[ci["precipitation_sum"]]))
            except (ValueError, IndexError):
                P.append(np.nan)
            if want_q:
                try:
                    q = float(f[ci["q_cfs"]])
                    Q.append(q if q >= 0 else np.nan)
                except (ValueError, IndexError):
                    Q.append(np.nan)
    return d, np.array(P), (np.array(Q) if want_q else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="gpu1080/corpora671/camels_corpus_daymet_v2",
                    help="supplies discharge, the day set, and the 'camels' column")
    ap.add_argument("--products", required=True,
                    help="label=dir,label=dir,... precipitation sources to score")
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2009)
    ap.add_argument("--basins", type=int, default=60)
    ap.add_argument("--cohort", default="data/m1_cohort.json")
    ap.add_argument("--out", default="benchmarks/m1_gate1_referee.json")
    a = ap.parse_args()
    os.chdir(ROOT)

    prods = [tuple(x.split("=", 1)) for x in a.products.split(",") if x]
    ids = json.loads(Path(a.cohort).read_text())["cohort"]
    rng = np.random.default_rng(0)            # fixed: the sample is not a free parameter
    sample = sorted(rng.choice(ids, min(a.basins, len(ids)), replace=False).tolist())

    per: dict[str, list] = {"camels": []}
    lag: dict[str, list] = {"camels": []}
    for label, _ in prods:
        per[label] = []; lag[label] = []
    n_used = 0

    for g in sample:
        fr = load(Path(a.frozen) / f"{g}.csv.gz", a.y0, a.y1, want_q=True)
        if fr is None:
            continue
        dates, Pc, Q = fr
        if np.isfinite(Q).sum() < 1000:
            continue
        dQ = np.full_like(Q, np.nan); dQ[1:] = Q[1:] - Q[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = dQ / np.where(Q > 0, Q, np.nan)

        cols = {"camels": Pc}
        ok = True
        for label, d in prods:
            got = load(Path(d) / f"{g}.csv.gz", a.y0, a.y1, want_q=False)
            if got is None:
                ok = False; break
            dd, Pp, _ = got
            m = dict(zip(dd, Pp))
            cols[label] = np.array([m.get(x, np.nan) for x in dates])
        if not ok:
            continue

        row = {}
        for label, P in cols.items():
            cs = {l: lagcorr(P, rel, l) for l in LAGS}
            cs = {l: c for l, c in cs.items() if not math.isnan(c)}
            if not cs:
                ok = False; break
            bl = max(cs, key=lambda l: cs[l])
            row[label] = (cs[bl], bl)
        if not ok:
            continue
        n_used += 1
        for label, (r, l) in row.items():
            per[label].append(r); lag[label].append(l)

    print(f"basins scored: {n_used} of {len(sample)} sampled   years {a.y0}-{a.y1}\n")
    base = float(np.median(per["camels"])) if per["camels"] else float("nan")
    print(f"{'product':14s} {'median peak r':>14s} {'mean':>8s} "
          f"{'delta vs camels':>16s} {'peak lag (mode)':>17s} {'verdict':>12s}")
    res = {}
    for label in ["camels"] + [p[0] for p in prods]:
        v = np.array(per[label])
        if not len(v):
            print(f"{label:14s} {'no data':>14s}"); continue
        med = float(np.median(v)); dl = med - base
        ls = np.array(lag[label])
        mode = int(np.bincount(ls - min(LAGS)).argmax() + min(LAGS))
        share = float((ls == mode).mean())
        verdict = ("reference" if label == "camels" else
                   "FUND" if dl >= -0.005 else
                   "FLAG" if dl >= -0.02 else "ABANDON")
        print(f"{label:14s} {med:14.4f} {v.mean():8.4f} {dl:+16.4f} "
              f"{mode:+d} ({share*100:.0f}%)".ljust(0).rjust(0) + f"{'':4s}{verdict:>12s}")
        res[label] = {"median_peak_r": med, "mean_peak_r": float(v.mean()),
                      "delta_vs_camels": dl, "modal_peak_lag": mode,
                      "modal_share": share, "n": int(len(v)), "verdict": verdict}
    Path(a.out).write_text(json.dumps(
        {"_note": "MODERN-1 Gate 1 Part B. Bars fixed in PREREG_v2.md before the run.",
         "years": [a.y0, a.y1], "n_basins": n_used, "results": res}, indent=1))
    print(f"\n-> {a.out}")
    print("\nA modal peak lag of 0 means the product already sits on the CAMELS "
          "convention.\nA non-zero mode is the build-time shift that product needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
