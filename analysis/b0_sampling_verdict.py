#!/usr/bin/env python
"""SCREEN B0 — point vs areal sampling, same product, same days, PAIRED.

`scripts_modern/screen_products_referee.py` already answers this question, but it
reports a DIFFERENCE OF MEDIANS, which has overstated an effect 10+ times in this
campaign (see benchmarks/EXPERIMENTS.md). This script reuses that screen's exact
arithmetic -- it imports `load` and `lagcorr` rather than reimplementing them, so
the numbers stay on the same scale as the -0.0244 already measured for Daymet
V4R1 -- and reports instead:

  * the PAIRED per-basin delta (areal - point) with a bootstrap CI over basins,
  * breadth (the fraction of basins where areal wins),
  * the delta stratified by `slope_mean` terciles.

The stratification is a FALSIFIER, not decoration. The proposed mechanism is
orographic averaging: a gauge-pixel sample misses the basin's wet high ground.
If the gain does not concentrate in the steep tercile then that mechanism is
wrong, and a positive number whose mechanism is wrong is a worse result than a
null -- it must be labelled unexplained rather than banked.

Both Daymet modern corpora carry a modal peak lag of +1 (the known V4R1 one-day
precip offset; the shift is applied later, at assembly into the m3 corpora). That
is harmless here: the screen is registration-invariant by construction (it takes
the peak over lags -2..+2) and both arms share the offset.
"""
from __future__ import annotations
import argparse, importlib.util, json, math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _referee():
    """Import the screen module directly so the arithmetic is literally shared."""
    p = ROOT / "scripts_modern" / "screen_products_referee.py"
    spec = importlib.util.spec_from_file_location("_referee", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # safe: its chdir is inside main()
    return mod


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, dtype=float)
    r = np.random.default_rng(seed)
    med = float(np.median(x))
    draws = np.median(r.choice(x, size=(n, len(x)), replace=True), axis=1)
    return med, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="data/modern_corpus/camels_corpus_nldas_modern_areal",
                    help="supplies discharge and the day set; only precip varies between arms")
    ap.add_argument("--point", default="data/modern_corpus/camels_corpus_daymet_modern")
    ap.add_argument("--areal", default="data/modern_corpus/camels_corpus_daymet_modern_areal")
    ap.add_argument("--y0", type=int, default=2015)
    ap.add_argument("--y1", type=int, default=2025)
    ap.add_argument("--cohort", default="data/m1_cohort.json")
    ap.add_argument("--attrs", default="data/camels_attrs.json")
    ap.add_argument("--out", default="benchmarks/b0_sampling_verdict.json")
    a = ap.parse_args()

    import os
    os.chdir(ROOT)
    ref = _referee()
    LAGS = ref.LAGS

    ids = json.loads(Path(a.cohort).read_text())["cohort"]
    attrs = json.loads(Path(a.attrs).read_text())

    rows = []
    for g in ids:
        fr = ref.load(Path(a.frozen) / f"{g}.csv.gz", a.y0, a.y1, want_q=True)
        if fr is None:
            continue
        dates, _Pc, Q = fr
        if np.isfinite(Q).sum() < 1000:
            continue
        dQ = np.full_like(Q, np.nan); dQ[1:] = Q[1:] - Q[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = dQ / np.where(Q > 0, Q, np.nan)

        got = {}
        for label, d in (("point", a.point), ("areal", a.areal)):
            g2 = ref.load(Path(d) / f"{g}.csv.gz", a.y0, a.y1, want_q=False)
            if g2 is None:
                got = None; break
            dd, Pp, _ = g2
            m = dict(zip(dd, Pp))
            P = np.array([m.get(x, np.nan) for x in dates])
            cs = {l: ref.lagcorr(P, rel, l) for l in LAGS}
            cs = {l: c for l, c in cs.items() if not math.isnan(c)}
            if not cs:
                got = None; break
            bl = max(cs, key=lambda l: cs[l])
            got[label] = (cs[bl], bl)
        if not got:
            continue

        at = attrs.get(g, {})
        rows.append({
            "id": g,
            "r_point": got["point"][0], "lag_point": got["point"][1],
            "r_areal": got["areal"][0], "lag_areal": got["areal"][1],
            "delta": got["areal"][0] - got["point"][0],
            "slope_mean": at.get("slope_mean"),
            "elev_mean": at.get("elev_mean"),
        })

    d = np.array([r["delta"] for r in rows])
    med, lo, hi = boot_ci(d)
    breadth = float((d > 0).mean())

    # Difference of medians, for contrast with the paired number.
    dom = float(np.median([r["r_areal"] for r in rows]) - np.median([r["r_point"] for r in rows]))

    # Falsifier: the mechanism is orographic, so sort by slope.
    sl = [(r["slope_mean"], r["delta"]) for r in rows
          if r["slope_mean"] is not None and np.isfinite(r["slope_mean"])]
    sl.sort()
    terciles = {}
    if len(sl) >= 30:
        k = len(sl) // 3
        for name, chunk in (("flat", sl[:k]), ("mid", sl[k:2 * k]), ("steep", sl[2 * k:])):
            dd = np.array([c[1] for c in chunk])
            m2, l2, h2 = boot_ci(dd, n=2000)
            terciles[name] = {
                "n": len(chunk),
                "slope_range": [chunk[0][0], chunk[-1][0]],
                "paired_median_delta": m2, "ci95": [l2, h2],
                "breadth": float((dd > 0).mean()),
            }
        sp = np.array([c[0] for c in sl]); dl = np.array([c[1] for c in sl])
        rs = float(np.corrcoef(sp.argsort().argsort(), dl.argsort().argsort())[0, 1])
    else:
        rs = float("nan")

    out = {
        "_note": ("SCREEN B0: paired point-vs-areal, same product/version/days. Imports "
                  "scripts_modern/screen_products_referee.py arithmetic. The headline is the "
                  "PAIRED median, not the difference of medians."),
        "frame": {"years": [a.y0, a.y1], "n_basins": len(rows),
                  "frozen": a.frozen, "point": a.point, "areal": a.areal},
        "paired_median_delta": med, "ci95": [lo, hi], "breadth": breadth,
        "difference_of_medians": dom,
        "median_r_point": float(np.median([r["r_point"] for r in rows])),
        "median_r_areal": float(np.median([r["r_areal"] for r in rows])),
        "modal_lag_point": int(np.bincount([r["lag_point"] + 2 for r in rows]).argmax()) - 2,
        "modal_lag_areal": int(np.bincount([r["lag_areal"] + 2 for r in rows]).argmax()) - 2,
        "slope_terciles": terciles,
        "spearman_delta_vs_slope": rs,
        "per_basin": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"basins paired: {len(rows)}   years {a.y0}-{a.y1}")
    print(f"median peak r   point {out['median_r_point']:.4f}   areal {out['median_r_areal']:.4f}")
    print(f"difference of medians        {dom:+.6f}   <- the statistic NOT to quote")
    print(f"PAIRED median delta          {med:+.6f}  [{lo:+.6f}, {hi:+.6f}]  breadth {breadth:.3f}")
    print(f"spearman(delta, slope_mean)  {rs:+.4f}   (mechanism falsifier: want > +0.15)")
    for n2 in ("flat", "mid", "steep"):
        if n2 in terciles:
            t = terciles[n2]
            print(f"  {n2:6s} n={t['n']:3d} slope {t['slope_range'][0]:7.2f}..{t['slope_range'][1]:7.2f}"
                  f"  delta {t['paired_median_delta']:+.6f} [{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}]"
                  f"  breadth {t['breadth']:.3f}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
