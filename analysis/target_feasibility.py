#!/usr/bin/env python3
"""Is no-q 0.845 still reachable once the gauge-error ceiling is respected?

We are at 0.8347 (7 streams + multi5 + inverse-MSE weighting) and the target is
0.845 -- a gap of +0.0103. But 29% of basins already score AT or ABOVE their own
gauge-error ceiling (headroom-where-effort-pays), so the gap cannot be closed by
improving those.

This asks the concrete question: if a perfect model captured ALL the remaining
recoverable headroom -- i.e. every basin rose to its own ceiling and no further
-- what would the ensemble median be? That is the CEILING ON THE TARGET, not on
a single basin, and it says whether 0.845 is reachable at all.

Also reports how much of the +0.0103 gap each cohort could contribute, which
says WHERE to spend the remaining effort.
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


def sigma_series(q, s_low, s_high):
    pos = q[q > 0]
    if pos.size < 10:
        return np.full(q.shape, (s_low + s_high) / 2)
    lo, hi = np.log(np.percentile(pos, 5)), np.log(np.percentile(pos, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(q.shape, (s_low + s_high) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.log(np.where(q > 0, q, np.nan))
    f = np.clip((lq - lo) / (hi - lo), 0.0, 1.0)
    return s_low + np.where(np.isfinite(f), f, 0.5) * (s_high - s_low)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="multi,daymet,nldas,maurer,multi5")
    ap.add_argument("--glob", default=None,
                    help="explicit dump glob (e.g. the with-q dumps); "
                         "bypasses --members")
    ap.add_argument("--target", type=float, default=0.845)
    ap.add_argument("--s-low", type=float, default=0.30)
    ap.add_argument("--s-high", type=float, default=0.18)
    ap.add_argument("--s-high-flashy", type=float, default=0.35)
    a = ap.parse_args()

    if a.glob:
        files = [f for f in glob.glob(a.glob)
                 if "_TRAIN_" not in f and "QUARANTINE" not in f]
    else:
        keep = set(a.members.split(","))
        files = [f for f in glob.glob("gpu1080/dumps/camels531ls_*_nhlstm_s*.csv.gz")
                 if "_TRAIN_" not in f and "QUARANTINE" not in f
                 and f.split("camels531ls_")[1].split("_nhlstm_")[0] in keep]
    if not files:
        raise SystemExit("no dumps matched")
    print(f"{len(files)} dumps\n")

    acc = None
    for f in files:
        d = pd.read_csv(f, usecols=["station_id", "t0", "h", "truth", "ymed"])
        d = d[d.h == 1][["station_id", "t0", "truth", "ymed"]]
        acc = d if acc is None else acc.merge(
            d[["station_id", "t0", "ymed"]], on=["station_id", "t0"],
            suffixes=("", f"_{len(acc.columns)}"))
    pred = [c for c in acc.columns if c.startswith("ymed")]
    acc["ens"] = acc[pred].mean(axis=1)

    flash = acc.groupby("station_id")["truth"].apply(
        lambda s: np.percentile(s[s > 0], 99) / np.median(s[s > 0])
        if (s > 0).sum() > 20 else np.nan)
    thr = flash.median()

    rows = []
    for sid, g in acc.groupby("station_id"):
        q = g["truth"].to_numpy(float)
        ok = np.isfinite(q)
        if ok.sum() < 50:
            continue
        qq = q[ok]
        V = ((qq - qq.mean()) ** 2).mean()
        if V <= 0:
            continue
        M = (qq ** 2).mean()
        s_hi = a.s_high_flashy if flash.get(sid, 0) > thr else a.s_high
        s2 = np.nanmean(sigma_series(q, a.s_low, s_hi) ** 2)
        ceil = 1 - (M / V) * (np.exp(s2) - 1)
        n = nse(g["truth"], g["ens"])
        if np.isfinite(n):
            rows.append((sid, n, ceil, flash.get(sid, np.nan)))

    df = pd.DataFrame(rows, columns=["sid", "nse", "ceiling", "flash"]).dropna()
    df["cohort"] = np.where(df.flash > thr, "flashy", "steady")
    cur = df.nse.median()

    # If every basin rose to its own ceiling (and none fell), what is the median?
    df["capped"] = np.maximum(df.nse, np.minimum(df.ceiling, 1.0))
    best = df.capped.median()

    print(f"current ensemble median NSE ......... {cur:.4f}")
    print(f"target ............................. {a.target:.4f}  "
          f"(gap {a.target - cur:+.4f})")
    print(f"ALL basins raised to their ceiling .. {best:.4f}  "
          f"(headroom {best - cur:+.4f})")
    verdict = "REACHABLE" if best >= a.target else "NOT REACHABLE"
    print(f"\n  => target is {verdict} under this error model")
    if best >= a.target:
        print(f"     it needs {100*(a.target-cur)/(best-cur):.0f}% of all "
              f"recoverable headroom")

    print("\nWhere the recoverable headroom lives:")
    print(f"{'cohort':<10}{'n':>5}{'med NSE':>10}{'med ceiling':>13}"
          f"{'sum gain':>11}")
    print("-" * 49)
    tot = (df.capped - df.nse).sum()
    for c in ("steady", "flashy"):
        s = df[df.cohort == c]
        gain = (s.capped - s.nse).sum()
        print(f"{c:<10}{len(s):>5}{s.nse.median():>10.3f}"
              f"{s.ceiling.median():>13.3f}{100*gain/tot:>10.0f}%")

    at = (df.ceiling <= df.nse).sum()
    print(f"\n{at}/{len(df)} basins ({100*at/len(df):.0f}%) are already at or "
          f"above their ceiling and contribute nothing.")
    print("\nNOTE: our own derivation; scenario-dependent. The extrapolation")
    print("check (1.7% preliminary) suggests the OPTIMISTIC scenario may be")
    print("closer to truth for CAMELS, which would raise every ceiling here.")


if __name__ == "__main__":
    main()
