#!/usr/bin/env python
"""Test the PRE-REGISTERED prediction about the point-sampling defect.

Registered 2026-08-15 00:25, before any western basin had been built:

  "The median precip point-bias in HUC 10-14 will be substantially worse than
   the -5.42% seen so far, because those basins combine high relief with 1 km
   grid resolution. If it is NOT worse, the slope mechanism is wrong and the
   whole framing needs revisiting. Do not rationalize the result after the fact."

Reports east (HUC 01-09) vs west (HUC 10-14) so the comparison is the one that
was registered, not one chosen after seeing the data.
"""
from __future__ import annotations
import gzip, json, math, statistics as st
from pathlib import Path

A = Path("data/modern_corpus/camels_corpus_daymet_modern_areal")
P = Path("data/modern_corpus/camels_corpus_daymet_modern")
attrs = json.load(open("data/camels_attrs.json"))


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
    try:
        ca, Ad = rd(fa)
        cp, Pd = rd(fp)
    except Exception:
        continue
    com = sorted(set(Ad) & set(Pd))
    if len(com) < 3000:
        continue
    rec = {"g": g, "huc": int(g[:2]),
           "slope": float(attrs.get(g, {}).get("slope_mean") or "nan"),
           "elev": float(attrs.get(g, {}).get("elev_mean") or "nan")}
    for col in ("precipitation_sum", "temperature_2m_max"):
        a = [float(Ad[d][ca[col]]) for d in com]
        p = [float(Pd[d][cp[col]]) for d in com]
        ma, mp = sum(a) / len(a), sum(p) / len(p)
        rec[col + "_pct"] = (mp - ma) / ma * 100 if ma else float("nan")
        rec[col + "_abs"] = mp - ma
    rows.append(rec)

east = [r for r in rows if r["huc"] <= 9]
west = [r for r in rows if 10 <= r["huc"] <= 14]


def summarize(label, group):
    if not group:
        print(f"{label}: none yet")
        return None
    pp = [r["precipitation_sum_pct"] for r in group]
    tt = [r["temperature_2m_max_abs"] for r in group]
    sl = [r["slope"] for r in group if not math.isnan(r["slope"])]
    el = [r["elev"] for r in group if not math.isnan(r["elev"])]
    m = st.median(pp)
    print(f"{label}  (n={len(group)})")
    print(f"   median precip point-bias : {m:+.2f}%")
    print(f"   median tmax  point-bias  : {st.median(tt):+.2f} C")
    print(f"   median slope / elev      : {st.median(sl):.1f} / {st.median(el):.0f} m")
    print(f"   worst precip bias        : {min(pp):+.2f}%")
    return m


print("=== PRE-REGISTERED TEST: does the defect worsen in the mountainous West? ===\n")
me = summarize("EAST  (HUC 01-09)", east)
print()
mw = summarize("WEST  (HUC 10-14)", west)

if me is not None and mw is not None:
    print("\n--- VERDICT vs the registered prediction (east was -5.42% at n=145) ---")
    print(f"   east median {me:+.2f}%   west median {mw:+.2f}%   "
          f"difference {mw - me:+.2f} pp")
    if mw < me - 1.0:
        print("   => PREDICTION CONFIRMED: the West is substantially worse.")
    elif mw > me + 1.0:
        print("   => ⚠️ PREDICTION FAILED IN THE OPPOSITE DIRECTION — the West is")
        print("      BETTER. The slope mechanism does not generalise; revisit the framing.")
    else:
        print("   => ⚠️ PREDICTION NOT CONFIRMED: east and west are comparable")
        print("      (<1 pp apart). The regional claim was wrong even though the")
        print("      within-sample slope correlation held.")

    xs = [r["slope"] for r in rows if not math.isnan(r["slope"])]
    ys = [r["precipitation_sum_pct"] for r in rows if not math.isnan(r["slope"])]
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    print(f"\n   corr(precip bias %, slope) over ALL {len(xs)} built basins = "
          f"{sxy / math.sqrt(sxx * syy):+.3f}")
