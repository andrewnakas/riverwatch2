#!/usr/bin/env python
"""DECISIVE test: point vs areal Daymet over the SAME period, same basins.

The 2015-2025-vs-1980-2014 comparison confounds sampling with real climate.
Here we fetch POINT Daymet (ORNL single-pixel, the exact call the modern corpus
used) for a period the FROZEN AREAL corpus already covers, for a sample of
basins stratified by slope. Any systematic difference is sampling, not climate.

Read-only w.r.t. the corpus: writes nothing but a JSON report to /tmp.
"""
from __future__ import annotations
import gzip, json, math, random, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path("/home/nakas/riverwatch2")
FROZEN = ROOT / "gpu1080/corpora671/camels_corpus_daymet_v2"
LATLON = json.load(open(ROOT / "data/modern_corpus/basin_latlon.json"))
ATTRS = json.load(open(ROOT / "data/camels_attrs.json"))
DAYMET = "https://daymet.ornl.gov/single-pixel/api/data"

START, END = "2008-01-01", "2010-12-31"   # inside the frozen corpus window
NBASIN = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def http_get(url, tries=4, timeout=90):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read().decode()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def fetch_point(lat, lon, start, end):
    """Identical variable handling to scripts_modern/extend_corpus_daymet.py."""
    import datetime
    url = (f"{DAYMET}?lat={lat}&lon={lon}"
           f"&vars=prcp,tmax,tmin,srad,vp,dayl&start={start}&end={end}")
    txt = http_get(url)
    if not txt or "prcp" not in txt:
        return {}
    lines = txt.splitlines()
    hi = next((i for i, l in enumerate(lines) if l.lower().startswith("year,yday")), None)
    if hi is None:
        return {}
    hdr = [h.strip() for h in lines[hi].split(",")]
    idx = {n: hdr.index(next(c for c in hdr if c.lower().startswith(n)))
           for n in ["year", "yday", "prcp", "tmax", "tmin", "srad", "vp", "dayl"]}
    out = {}
    for l in lines[hi + 1:]:
        f = l.split(",")
        if len(f) < len(hdr):
            continue
        try:
            yr = int(float(f[idx["year"]])); yd = int(float(f[idx["yday"]]))
            d = (datetime.date(yr, 1, 1) + datetime.timedelta(days=yd - 1)).isoformat()
            tmax = float(f[idx["tmax"]]); tmin = float(f[idx["tmin"]])
            out[d] = {
                "precipitation_sum": float(f[idx["prcp"]]),
                "temperature_2m_max": tmax,
                "temperature_2m_min": tmin,
                "temperature_2m_mean": (tmax + tmin) / 2,
                "vapor_pressure": float(f[idx["vp"]]),
            }
        except Exception:
            continue
    return out


def load_frozen(gauge, dmin, dmax):
    p = FROZEN / f"{gauge}.csv.gz"
    if not p.exists():
        return {}
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        out = {}
        for line in fh:
            f = line.rstrip("\n").split(",")
            if len(f) < len(hdr):
                continue
            d = f[ci["date"]]
            if d < dmin or d > dmax:
                continue
            rec = {}
            for c in ["precipitation_sum", "temperature_2m_mean",
                      "temperature_2m_max", "temperature_2m_min", "vapor_pressure"]:
                try:
                    rec[c] = float(f[ci[c]])
                except (ValueError, KeyError):
                    rec[c] = None
            out[d] = rec
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    # stratify by slope so the terrain relationship can be resolved
    cands = [g for g in LATLON if (FROZEN / f"{g}.csv.gz").exists()
             and g in ATTRS and ATTRS[g].get("slope_mean") is not None]
    cands.sort(key=lambda g: float(ATTRS[g]["slope_mean"]))
    step = max(1, len(cands) // NBASIN)
    sample = cands[::step][:NBASIN]
    print(f"testing {len(sample)} basins, {START}..{END}, stratified by slope_mean")
    print(f"slope range: {ATTRS[sample[0]]['slope_mean']} .. {ATTRS[sample[-1]]['slope_mean']}\n")

    VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
            "temperature_2m_min", "vapor_pressure"]
    rows = []
    for i, g in enumerate(sample, 1):
        lat, lon = LATLON[g]
        try:
            pt = fetch_point(lat, lon, START, END)
        except Exception as e:
            print(f"  [{i}] {g} FETCH-FAIL {e}")
            continue
        ar = load_frozen(g, START, END)
        common = sorted(set(pt) & set(ar))
        if len(common) < 300:
            print(f"  [{i}] {g} only {len(common)} common days - skip")
            continue
        rec = {"gauge": g, "n": len(common),
               "slope": float(ATTRS[g]["slope_mean"]),
               "elev": float(ATTRS[g].get("elev_mean") or "nan")}
        for v in VARS:
            pv = [pt[d][v] for d in common if pt[d].get(v) is not None
                  and ar[d].get(v) is not None]
            av = [ar[d][v] for d in common if pt[d].get(v) is not None
                  and ar[d].get(v) is not None]
            if not pv:
                continue
            mp, ma = mean(pv), mean(av)
            rec[v] = {"point": mp, "areal": ma, "diff": mp - ma,
                      "pct": (mp - ma) / ma * 100 if ma else float("nan")}
        rows.append(rec)
        if i % 10 == 0:
            print(f"  ...{i}/{len(sample)}")
        time.sleep(0.3)

    print(f"\n=== POINT minus AREAL, identical period {START}..{END}, n={len(rows)} basins ===")
    for v in VARS:
        vals = [r[v]["diff"] for r in rows if v in r]
        pcts = [r[v]["pct"] for r in rows if v in r and not math.isnan(r[v]["pct"])]
        if not vals:
            continue
        vals.sort(); pcts.sort()
        n = len(vals)
        neg = sum(1 for d in vals if d < 0)
        print(f"{v}")
        print(f"   median diff : {vals[n//2]:+.4f}   ({pcts[len(pcts)//2]:+.2f}%)")
        print(f"   p10 / p90   : {vals[int(.1*n)]:+.4f} / {vals[int(.9*n)]:+.4f}")
        print(f"   point < areal in {neg}/{n} basins ({neg/n*100:.0f}%)")

    # the signature: does the precip bias grow with slope?
    xs = [r["slope"] for r in rows if "precipitation_sum" in r]
    ys = [r["precipitation_sum"]["pct"] for r in rows if "precipitation_sum" in r]
    if len(xs) > 10:
        mx, my = mean(xs), mean(ys)
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
        print(f"\ncorr(precip pct bias, slope_mean) = {r:+.3f}  n={len(xs)}")
        lo = [y for x, y in zip(xs, ys) if x < sorted(xs)[len(xs)//3]]
        hi = [y for x, y in zip(xs, ys) if x > sorted(xs)[2*len(xs)//3]]
        print(f"  flattest third: median precip bias {sorted(lo)[len(lo)//2]:+.2f}%")
        print(f"  steepest third: median precip bias {sorted(hi)[len(hi)//2]:+.2f}%")

    out = Path("/tmp/point_vs_areal_sameperiod.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
