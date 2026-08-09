#!/usr/bin/env python3
"""with-q: is there ANY member-side work left, or is it all ceiling?

62% of with-q basins already score at/above their gauge-error ceiling
(withq-095-needs-89pct-of-headroom). Before spending more GPU on with-q members,
answer three concrete questions:

1. ORACLE vs CEILING. Per-basin best-member selection was measured at 0.9271.
   How much of the ceiling-limited headroom does perfect member SELECTION
   already capture? If the oracle ~= the ceiling, member work is closed and only
   the ceiling matters.

2. WHERE the non-saturated basins are. If the remaining headroom is concentrated
   in few basins, a specialist member is justified; if it is spread thin, it is
   not.

3. Does the CURRENT ensemble already beat every individual member on the
   non-saturated basins? If a single member wins there, that member is the one
   to replicate.

Everything here reads existing dumps -- no GPU.
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
    ap.add_argument("--glob",
                    default="data/mblstm/gpu_dumps_s14/camels531_*_withq5_full531.csv.gz")
    ap.add_argument("--s-low", type=float, default=0.25)
    ap.add_argument("--s-high", type=float, default=0.13)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    names = [f.split("camels531_")[1].split("_withq5")[0] for f in files]
    print(f"{len(files)} members: {names}\n")

    acc = None
    for f, n in zip(files, names):
        d = pd.read_csv(f, usecols=["station_id", "t0", "h", "truth", "ymed"])
        d = d[d.h == 1][["station_id", "t0", "truth", "ymed"]].rename(
            columns={"ymed": n})
        acc = d if acc is None else acc.merge(
            d[["station_id", "t0", n]], on=["station_id", "t0"])
    acc["ens"] = acc[names].mean(axis=1)

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
        s2 = np.nanmean(sigma_series(q, a.s_low, a.s_high) ** 2)
        ceil = 1 - (M / V) * (np.exp(s2) - 1)
        e = nse(g["truth"], g["ens"])
        per = {n: nse(g["truth"], g[n]) for n in names}
        best = max(per.values())
        rows.append((sid, e, best, ceil, max(per, key=per.get)))

    df = pd.DataFrame(rows, columns=["sid", "ens", "best_member",
                                     "ceiling", "which"]).dropna()
    df["oracle"] = np.maximum(df.ens, df.best_member)

    print("=== 1. ORACLE vs CEILING (optimistic scenario) ===")
    print(f"  ensemble median ......... {df.ens.median():.4f}")
    print(f"  per-basin ORACLE ........ {df.oracle.median():.4f}"
          f"  (+{df.oracle.median()-df.ens.median():.4f})")
    print(f"  gauge CEILING ........... {np.minimum(df.ceiling,1).median():.4f}")
    capped = np.maximum(df.ens, np.minimum(df.ceiling, 1.0))
    tot_head = (capped - df.ens).sum()
    orac_head = (df.oracle - df.ens).sum()
    if tot_head > 0:
        print(f"\n  perfect member SELECTION captures "
              f"{100*orac_head/tot_head:.1f}% of ceiling-limited headroom")
        if orac_head / tot_head < 0.15:
            print("  => member SELECTION is closed; the ceiling dominates.")

    print("\n=== 2. WHERE the non-saturated basins are ===")
    ns = df[df.ceiling > df.ens]
    print(f"  {len(ns)}/{len(df)} basins are below their ceiling")
    if len(ns):
        gaps = (np.minimum(ns.ceiling, 1.0) - ns.ens).sort_values(ascending=False)
        for k in (10, 25, 50, 100):
            if k <= len(gaps):
                print(f"    top {k:>3} basins hold "
                      f"{100*gaps.head(k).sum()/gaps.sum():.0f}% of the headroom")

    print("\n=== 3. Best single member on the NON-SATURATED basins ===")
    if len(ns):
        sub = acc[acc.station_id.isin(set(ns.sid))]
        per = {n: np.nanmedian([nse(g["truth"], g[n])
                                for _, g in sub.groupby("station_id")])
               for n in names}
        ense = np.nanmedian([nse(g["truth"], g["ens"])
                             for _, g in sub.groupby("station_id")])
        for n, v in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f"    {n:<10} {v:.4f}")
        print(f"    {'ENSEMBLE':<10} {ense:.4f}")
        w = max(per, key=per.get)
        if per[w] > ense:
            print(f"\n  ⚠️ {w} BEATS the ensemble on these basins "
                  f"({per[w]-ense:+.4f}) -- worth replicating.")
        else:
            print(f"\n  ensemble beats every member here "
                  f"(+{ense-max(per.values()):.4f}) -- no single member to chase.")


if __name__ == "__main__":
    main()
