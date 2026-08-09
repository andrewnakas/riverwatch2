#!/usr/bin/env python3
"""Is the flashy-basin HEADROOM real, or an artifact of the error model?

THE OPEN QUESTION
-----------------
headroom_vs_ceiling.py says ~104 flashy basins hold all the remaining headroom
(median NSE 0.527). But those are exactly the basins the literature says have
the LEAST reliable rating curves (Westerberg & McMillan 2015: worst ratings
occur where "peak flows occur seldom and last only a few hours"; Kiang 2018:
41-200% where the rating is extrapolated). If their true sigma is much larger
than the 0.35 we assumed, their real ceiling collapses and the headroom is
illusory.

So: at what sigma does each cohort's headroom go to zero? That "break-even
sigma" is the decision-relevant number, because it converts an unknown (the
true gauge error in flashy basins) into a checkable claim (is flashy-basin
high-flow sigma above or below X?).

Also reports what fraction of CURRENT ensemble error would have to be gauge
noise for the headroom to vanish -- a second framing of the same question.
"""
import argparse
import glob

import numpy as np
import pandas as pd


def nse(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    ok = np.isfinite(t) & np.isfinite(p)
    if ok.sum() < 10:
        return np.nan
    t, p = t[ok], p[ok]
    den = ((t - t.mean()) ** 2).sum()
    return np.nan if den <= 0 else 1.0 - ((p - t) ** 2).sum() / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="multi,daymet,nldas,maurer,multi5")
    a = ap.parse_args()

    keep = set(a.members.split(","))
    files = [f for f in glob.glob("gpu1080/dumps/camels531ls_*_nhlstm_s*.csv.gz")
             if "_TRAIN_" not in f and "QUARANTINE" not in f
             and f.split("camels531ls_")[1].split("_nhlstm_")[0] in keep]
    if not files:
        raise SystemExit("no dumps matched")
    print(f"{len(files)} dumps, members: {sorted(keep)}\n")

    acc = None
    for f in files:
        d = pd.read_csv(f, usecols=["station_id", "t0", "h", "truth", "ymed"])
        d = d[d.h == 1][["station_id", "t0", "truth", "ymed"]]
        acc = d if acc is None else acc.merge(
            d[["station_id", "t0", "ymed"]], on=["station_id", "t0"],
            suffixes=("", f"_{len(acc.columns)}"))
    pred = [c for c in acc.columns if c.startswith("ymed")]
    acc["ens"] = acc[pred].mean(axis=1)

    rows = []
    for sid, g in acc.groupby("station_id"):
        q = g["truth"].to_numpy(float)
        ok = np.isfinite(q)
        if ok.sum() < 50:
            continue
        q = q[ok]
        M = (q ** 2).mean()
        V = ((q - q.mean()) ** 2).mean()
        if V <= 0:
            continue
        n = nse(g["truth"], g["ens"])
        if not np.isfinite(n):
            continue
        # ceiling(sigma) = 1 - (M/V)(e^{s^2}-1);  set == n  ->  break-even sigma
        val = (1 - n) * V / M + 1
        s_be = np.sqrt(np.log(val)) if val > 1 else 0.0
        flash = np.percentile(q[q > 0], 99) / np.median(q[q > 0]) if (q > 0).sum() > 20 else np.nan
        rows.append((sid, n, M / V, s_be, flash))

    df = pd.DataFrame(rows, columns=["sid", "nse", "MV", "s_be", "flash"]).dropna()
    thr = df.flash.median()
    df["cohort"] = np.where(df.flash > thr, "flashy", "steady")

    print("BREAK-EVEN sigma: the relative gauge error at which this basin's")
    print("ceiling equals its CURRENT score. Above it, we are already at/past")
    print("the ceiling; below it, real headroom remains.\n")
    print(f"{'cohort':<10}{'n':>5}{'med NSE':>10}{'med M/V':>10}{'med break-even sigma':>22}")
    print("-" * 58)
    for c in ("steady", "flashy"):
        s = df[df.cohort == c]
        print(f"{c:<10}{len(s):>5}{s.nse.median():>10.3f}{s.MV.median():>10.2f}"
              f"{s.s_be.median():>22.3f}")
    print(f"{'ALL':<10}{len(df):>5}{df.nse.median():>10.3f}{df.MV.median():>10.2f}"
          f"{df.s_be.median():>22.3f}")

    print("\nLiterature anchors for high-flow relative sigma:")
    print("  Coxon 2015 / Aerts 2024 well-gauged .......... 0.12-0.13")
    print("  Westerberg & McMillan 2015 Q0.1 half-width ... 0.20-0.23")
    print("  our 'literature' flashy assumption ........... 0.35")
    print("  Kiang 2018 EXTRAPOLATED ratings .............. 0.41-2.00")

    for c in ("steady", "flashy"):
        s = df[df.cohort == c]
        for lab, sig in (("well-gauged 0.13", 0.13), ("W&M 0.22", 0.22),
                         ("our flashy 0.35", 0.35), ("Kiang low 0.41", 0.41)):
            frac = (s.s_be <= sig).mean()
            print(f"  {c:<7} if sigma={lab:<18} -> {100*frac:5.1f}% of basins are AT/PAST ceiling")
        print()

    print("READ THIS AS: the flashy cohort's headroom survives only if its true")
    print("high-flow gauge error is BELOW its break-even sigma. That is the")
    print("single empirical question that decides whether the remaining")
    print("headroom is bankable -- and it is answerable from USGS rating-curve")
    print("metadata (how many CAMELS gauges have peaks above the highest")
    print("direct measurement), which no one has published.")


if __name__ == "__main__":
    main()
