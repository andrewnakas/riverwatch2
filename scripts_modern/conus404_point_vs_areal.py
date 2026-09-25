#!/usr/bin/env python
"""Measure the point-vs-areal sampling effect from data already on disk.

corpora671/ holds BOTH:
  camels_corpus_conus404_v2                   (areal, basin-mean)
  camels_corpus_conus404_POINTSAMPLED_DO_NOT_USE  (point at gauge lat/lon)

Same variable set, same basins, same period -> the difference is PURE sampling
method, with no climate confound and no network dependency. That gives an
unbiased read on how large the effect is for THIS basin set, which is what the
modern Daymet corpus (built point-sampled) inherits.
"""
from __future__ import annotations
import gzip, json, math
from pathlib import Path

ROOT = Path("/home/nakas/riverwatch2")
AREAL = ROOT / "gpu1080/corpora671/camels_corpus_conus404_v2"
POINT = ROOT / "gpu1080/corpora671/camels_corpus_conus404_POINTSAMPLED_DO_NOT_USE"
ATTRS = json.load(open(ROOT / "data/camels_attrs.json"))


def load(path):
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        rows = {}
        for line in fh:
            f = line.rstrip("\n").split(",")
            if len(f) < len(hdr):
                continue
            rows[f[ci["date"]]] = f
    return hdr, ci, rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def corr(xs, ys):
    if len(xs) < 10:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def main():
    a_files = {p.name.split(".")[0]: p for p in AREAL.glob("*.csv.gz")}
    p_files = {p.name.split(".")[0]: p for p in POINT.glob("*.csv.gz")}
    common = sorted(set(a_files) & set(p_files))
    print(f"areal corpus : {len(a_files)} basins  ({AREAL.name})")
    print(f"point corpus : {len(p_files)} basins  ({POINT.name})")
    print(f"common       : {len(common)}\n")
    if not common:
        return

    hdr, _, _ = load(a_files[common[0]])
    print("columns:", hdr, "\n")
    VARS = [c for c in hdr if c not in ("date", "q_cfs")]

    recs = []
    for g in common:
        try:
            ah, aci, ar = load(a_files[g])
            ph, pci, pr = load(p_files[g])
        except Exception:
            continue
        days = sorted(set(ar) & set(pr))
        if len(days) < 300:
            continue
        rec = {"gauge": g, "n": len(days)}
        for v in VARS:
            if v not in aci or v not in pci:
                continue
            av, pv = [], []
            for d in days:
                try:
                    x = float(ar[d][aci[v]]); y = float(pr[d][pci[v]])
                except ValueError:
                    continue
                if math.isnan(x) or math.isnan(y):
                    continue
                av.append(x); pv.append(y)
            if len(av) < 300:
                continue
            ma, mp = mean(av), mean(pv)
            # per-day correlation: does point tracking areal day-to-day?
            rec[v] = {"areal": ma, "point": mp, "diff": mp - ma,
                      "pct": (mp - ma) / ma * 100 if ma else float("nan"),
                      "r": corr(av, pv)}
        recs.append(rec)

    print(f"compared {len(recs)} basins, ~{recs[0]['n']} days each\n")
    print("=== POINT minus AREAL (pure sampling effect, no climate confound) ===")
    for v in VARS:
        vals = sorted(r[v]["diff"] for r in recs if v in r)
        pcts = sorted(r[v]["pct"] for r in recs if v in r and not math.isnan(r[v]["pct"]))
        rs = sorted(r[v]["r"] for r in recs if v in r and not math.isnan(r[v]["r"]))
        if not vals:
            continue
        n = len(vals)
        neg = sum(1 for d in vals if d < 0)
        print(f"\n{v}   (n={n})")
        print(f"   median diff  : {vals[n//2]:+.4f}"
              + (f"   ({pcts[len(pcts)//2]:+.2f}%)" if pcts else ""))
        print(f"   p10 / p90    : {vals[int(.1*n)]:+.4f} / {vals[int(.9*n)]:+.4f}")
        print(f"   point<areal  : {neg}/{n} ({neg/n*100:.0f}%)")
        if rs:
            print(f"   daily corr   : median r={rs[len(rs)//2]:.4f}  worst={rs[0]:.4f}")

    # terrain signature
    print("\n=== bias vs terrain ===")
    for v in VARS:
        if not any(v in r for r in recs):
            continue
        for key in ("elev_mean", "slope_mean"):
            xs, ys = [], []
            for r in recs:
                if v not in r or math.isnan(r[v]["pct"]):
                    continue
                try:
                    x = float(ATTRS.get(r["gauge"], {}).get(key))
                except (TypeError, ValueError):
                    continue
                if math.isnan(x):
                    continue
                xs.append(x); ys.append(r[v]["pct"])
            if len(xs) > 30:
                print(f"   corr(pct bias {v:<26}, {key:<11}) = {corr(xs, ys):+.3f}  n={len(xs)}")


if __name__ == "__main__":
    main()
