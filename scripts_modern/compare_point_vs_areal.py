#!/usr/bin/env python
"""Price the point-vs-areal risk in the modern Daymet corpus.

The frozen training corpus (corpora671/camels_corpus_daymet_v2) is basin-AREAL.
The modern corpus was built from the ORNL single-pixel API at one lat/lon per
basin, i.e. POINT. If the model trains on areal and is served point, that is a
distribution shift — the exact defect that got a sibling corpus renamed
"..._POINTSAMPLED_DO_NOT_USE".

Periods differ (frozen 1980-2014 vs modern 2015-2025) so some delta is real
climate. The point-sampling signature is a SYSTEMATIC bias, and one that grows
with terrain relief. We report the distribution and the correlation with
elevation/relief attributes so the two causes can be told apart.
"""
from __future__ import annotations
import gzip, json, math, sys
from pathlib import Path

ROOT = Path("/home/nakas/riverwatch2")
FROZEN = ROOT / "gpu1080/corpora671/camels_corpus_daymet_v2"
MODERN = ROOT / "data/modern_corpus/camels_corpus_daymet_modern"


def load_cols(path, cols, dmin=None, dmax=None):
    """Return {col: [values]} for rows within [dmin, dmax]."""
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        for c in cols:
            if c not in ci:
                return None
        out = {c: [] for c in cols}
        for line in fh:
            f = line.rstrip("\n").split(",")
            if len(f) < len(hdr):
                continue
            d = f[ci["date"]]
            if dmin and d < dmin:
                continue
            if dmax and d > dmax:
                continue
            for c in cols:
                v = f[ci[c]]
                try:
                    x = float(v)
                    if not math.isnan(x):
                        out[c].append(x)
                except ValueError:
                    pass
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]


def main():
    fro = {p.name.split(".")[0]: p for p in FROZEN.glob("*.csv.gz")}
    mod = {p.name.split(".")[0]: p for p in MODERN.glob("*.csv.gz")}
    common = sorted(set(fro) & set(mod))
    print(f"frozen basins : {len(fro)}  ({FROZEN.name})")
    print(f"modern basins : {len(mod)}")
    print(f"common        : {len(common)}\n")

    # frozen header check — confirm schema match
    with gzip.open(fro[common[0]], "rt") as fh:
        print("frozen header:", fh.readline().strip())
    with gzip.open(mod[common[0]], "rt") as fh:
        print("modern header:", fh.readline().strip())
    print()

    rows = []
    for g in common:
        a = load_cols(fro[g], VARS)
        b = load_cols(mod[g], VARS)
        if a is None or b is None:
            continue
        rec = {"gauge": g}
        for v in VARS:
            if not a[v] or not b[v]:
                rec[v] = None
                continue
            ma, mb = mean(a[v]), mean(b[v])
            rec[v] = (ma, mb, mb - ma, (mb - ma) / ma * 100 if ma else float("nan"))
        rows.append(rec)

    print(f"compared {len(rows)} basins (frozen full record vs modern 2015-2025)\n")
    for v in VARS:
        vals = [r[v] for r in rows if r[v]]
        if not vals:
            continue
        diffs = sorted(d for _, _, d, _ in vals)
        pcts = sorted(p for _, _, _, p in vals if not math.isnan(p))
        n = len(diffs)
        med = diffs[n // 2]
        medp = pcts[len(pcts) // 2] if pcts else float("nan")
        pos = sum(1 for d in diffs if d > 0)
        print(f"{v}")
        print(f"   median delta (modern-frozen): {med:+.4f}   ({medp:+.2f}%)")
        print(f"   p10 / p90 delta            : {diffs[int(.1*n)]:+.4f} / {diffs[int(.9*n)]:+.4f}")
        print(f"   basins with modern > frozen: {pos}/{n} ({pos/n*100:.1f}%)")

    # correlate precip bias with elevation & relief -> the point-sampling signature
    attrs_p = ROOT / "data/camels_attrs.json"
    if attrs_p.exists():
        attrs = json.load(open(attrs_p))
        print("\n=== precip bias vs terrain (point-sampling signature) ===")
        sample_key = next(iter(attrs.values()))
        cand = [k for k in sample_key
                if any(t in k.lower() for t in ("elev", "slope", "relief", "area"))]
        print("candidate terrain attrs:", cand)
        for key in cand:
            xs, ys = [], []
            for r in rows:
                if not r["precipitation_sum"]:
                    continue
                a = attrs.get(r["gauge"], {})
                try:
                    x = float(a.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isnan(x):
                    continue
                xs.append(x)
                ys.append(r["precipitation_sum"][3])   # pct diff
            if len(xs) < 30:
                continue
            mx, my = mean(xs), mean(ys)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sxx = sum((x - mx) ** 2 for x in xs)
            syy = sum((y - my) ** 2 for y in ys)
            r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
            print(f"   corr(pct precip bias, {key:<28}) = {r:+.3f}   n={len(xs)}")
    else:
        print("\n(no camels_attrs.json — skipped terrain correlation)")


if __name__ == "__main__":
    main()
