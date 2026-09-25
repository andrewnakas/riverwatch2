#!/usr/bin/env python
"""Audit the modern Daymet corpus: coverage, date range, NaN/gaps, per-column sanity.

Measures state rather than assuming it. Reports the cohort that is actually usable
for a modern-window evaluation, which is the number that gates everything downstream.
"""
from __future__ import annotations
import gzip, sys, math
from pathlib import Path
from collections import Counter

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else
              "/home/nakas/riverwatch2/data/modern_corpus/camels_corpus_daymet_modern")
COLS = ["date","q_cfs","temperature_2m_mean","temperature_2m_max","temperature_2m_min",
        "precipitation_sum","shortwave_radiation_sum","vapor_pressure"]


def isnum(s):
    if s is None or s == "" or s.lower() in ("nan", "none", "null"):
        return False
    try:
        v = float(s)
        return not (math.isnan(v) or math.isinf(v))
    except ValueError:
        return False


def audit(path):
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        rows = [l.rstrip("\n").split(",") for l in fh if l.strip()]
    if not rows:
        return {"gauge": path.name.split(".")[0], "rows": 0, "empty": True}
    ci = {c: hdr.index(c) for c in hdr}
    n = len(rows)
    dates = [r[ci["date"]] for r in rows]
    good = {c: sum(1 for r in rows if isnum(r[ci[c]])) for c in hdr if c != "date"}
    # duplicate / out-of-order dates
    dup = n - len(set(dates))
    ooo = sum(1 for a, b in zip(dates, dates[1:]) if b <= a)
    # q sanity
    qs = [float(r[ci["q_cfs"]]) for r in rows if isnum(r[ci["q_cfs"]])]
    neg = sum(1 for q in qs if q < 0)
    zero = sum(1 for q in qs if q == 0)
    # longest run of missing q
    run = best = 0
    for r in rows:
        if isnum(r[ci["q_cfs"]]):
            run = 0
        else:
            run += 1
            best = max(best, run)
    # precip sanity (a stuck/zero precip column is a known silent-failure mode)
    ps = [float(r[ci["precipitation_sum"]]) for r in rows if isnum(r[ci["precipitation_sum"]])]
    return {
        "gauge": path.name.split(".")[0], "rows": n, "empty": False,
        "start": dates[0], "end": dates[-1], "dup": dup, "ooo": ooo,
        "q_good": good.get("q_cfs", 0), "q_frac": good.get("q_cfs", 0) / n,
        "q_neg": neg, "q_zero": zero, "q_maxgap": best,
        "q_min": min(qs) if qs else None, "q_max": max(qs) if qs else None,
        "p_good": good.get("precipitation_sum", 0),
        "p_sum": sum(ps) if ps else 0.0,
        "p_zerofrac": (sum(1 for p in ps if p == 0) / len(ps)) if ps else 1.0,
        "cols_bad": {c: n - g for c, g in good.items() if g < n},
    }


def main():
    files = sorted(CORPUS.glob("*.csv.gz"))
    print(f"corpus: {CORPUS}")
    print(f"files : {len(files)}")
    res = []
    bad_read = []
    for f in files:
        try:
            res.append(audit(f))
        except Exception as e:
            bad_read.append((f.name, repr(e)))
    if bad_read:
        print(f"\n!! UNREADABLE: {len(bad_read)}")
        for nm, e in bad_read[:10]:
            print("   ", nm, e)

    ok = [r for r in res if not r["empty"]]
    print(f"readable: {len(ok)}")

    print("\n=== date ranges ===")
    for k, v in Counter((r["start"], r["end"]) for r in ok).most_common(10):
        print(f"  {k[0]} .. {k[1]}   {v} basins")
    print("\n=== row counts ===")
    for k, v in Counter(r["rows"] for r in ok).most_common(10):
        print(f"  {k} rows   {v} basins")

    # streamflow coverage — the number that gates the modern evaluation cohort
    print("\n=== q_cfs coverage ===")
    for lo, hi, lbl in [(1.0, 1.01, "100%"), (0.99, 1.0, ">=99%"), (0.95, 0.99, ">=95%"),
                        (0.90, 0.95, ">=90%"), (0.50, 0.90, ">=50%"),
                        (0.0, 0.50, "<50%"), (-1, 0.0001, "ZERO")]:
        c = [r for r in ok if lo <= r["q_frac"] < hi]
        if c:
            print(f"  {lbl:>6}: {len(c):4d} basins")
    full = [r for r in ok if r["q_frac"] >= 0.99]
    print(f"\n  >=99% complete q: {len(full)} basins  <-- usable modern cohort")

    print("\n=== worst 15 by q coverage ===")
    for r in sorted(ok, key=lambda r: r["q_frac"])[:15]:
        print(f"  {r['gauge']}  q={r['q_frac']*100:5.1f}%  maxgap={r['q_maxgap']:5d}d  "
              f"{r['start']}..{r['end']}")

    print("\n=== anomalies ===")
    for lbl, pred in [
        ("duplicate dates", lambda r: r["dup"] > 0),
        ("non-monotonic dates", lambda r: r["ooo"] > 0),
        ("negative q", lambda r: r["q_neg"] > 0),
        ("all-zero precip", lambda r: r["p_sum"] == 0),
        ("precip >95% zeros", lambda r: r["p_zerofrac"] > 0.95),
        ("missing forcing values", lambda r: any(
            c != "q_cfs" for c in r["cols_bad"])),
    ]:
        hits = [r for r in ok if pred(r)]
        print(f"  {lbl:<28}: {len(hits)}")
        for r in hits[:5]:
            print(f"      {r['gauge']}  {r.get('cols_bad')}")

    print("\n=== q magnitude sanity (should look like cfs, not mm/day) ===")
    med_max = sorted(r["q_max"] for r in ok if r["q_max"] is not None)
    if med_max:
        print(f"  median of per-basin max q: {med_max[len(med_max)//2]:.1f} cfs")
        print(f"  min/max of per-basin max q: {med_max[0]:.2f} / {med_max[-1]:.1f}")


if __name__ == "__main__":
    main()
