#!/usr/bin/env python
"""Compare the rebuilt AREAL corpus against the POINT corpus as the build runs.

Confirms on real output that the rebuild reproduces the expected orographic
signature (point reads drier as terrain steepens) rather than merely producing
files of the right shape.
"""
from __future__ import annotations
import gzip, json, math, statistics as st
from pathlib import Path

A = Path("data/modern_corpus/camels_corpus_daymet_modern_areal")
P = Path("data/modern_corpus/camels_corpus_daymet_modern")
attrs = json.load(open("data/camels_attrs.json"))
ids = json.load(open("data/camels_gauge_ids.json"))
b531 = set(ids["531"])


def rd(p):
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        return ci, {l.split(",", 1)[0]: l.strip().split(",") for l in fh if l.strip()}


rows = []
for fa in sorted(A.glob("*.csv.gz")):
    g = fa.name.split(".")[0]
    fp = P / f"{g}.csv.gz"
    if not fp.exists():
        continue
    ca, Ad = rd(fa)
    cp, Pd = rd(fp)
    com = sorted(set(Ad) & set(Pd))
    if len(com) < 3000:
        continue
    sl = attrs.get(g, {}).get("slope_mean")
    rec = {"g": g, "slope": float(sl) if sl is not None else float("nan"),
           "n": len(com), "in531": g in b531}
    for col in ("precipitation_sum", "temperature_2m_max"):
        a = [float(Ad[d][ca[col]]) for d in com]
        p = [float(Pd[d][cp[col]]) for d in com]
        ma, mp = sum(a) / len(a), sum(p) / len(p)
        rec[col + "_pct"] = (mp - ma) / ma * 100 if ma else float("nan")
        rec[col + "_abs"] = mp - ma
    rows.append(rec)

print(f"compared {len(rows)} rebuilt basins vs the point corpus "
      f"({sum(1 for r in rows if r['in531'])} in the 531)\n")
rows.sort(key=lambda r: r["slope"])
print(f"{'basin':>9} {'slope':>7} {'precip pt-areal %':>18} {'tmax pt-areal C':>17}")
for r in rows:
    print(f"{r['g']:>9} {r['slope']:7.1f} {r['precipitation_sum_pct']:18.2f} "
          f"{r['temperature_2m_max_abs']:17.2f}")

xs = [r["slope"] for r in rows if not math.isnan(r["slope"])]
ys = [r["precipitation_sum_pct"] for r in rows if not math.isnan(r["slope"])]
ts = [r["temperature_2m_max_abs"] for r in rows if not math.isnan(r["slope"])]
if len(xs) > 5:
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
    print(f"\ncorr(precip point-bias %, slope_mean) = {r:+.3f}   n={len(xs)}")
    print(f"median precip point-bias : {st.median(ys):+.2f}%")
    print(f"median tmax  point-bias  : {st.median(ts):+.2f} C")
    steep = [y for x, y in zip(xs, ys) if x > st.median(xs)]
    flat = [y for x, y in zip(xs, ys) if x <= st.median(xs)]
    if steep and flat:
        print(f"  flatter half: median {st.median(flat):+.2f}%   "
              f"steeper half: median {st.median(steep):+.2f}%")
