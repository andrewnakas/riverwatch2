#!/usr/bin/env python3
"""Per-basin HEADROOM: how far is each basin from its own gauge-error ceiling?

The aggregate ceiling (0.8956 central) hides the decision-relevant question:
WHICH basins still have room to improve, and which are already at the limit of
what a noisy gauge can reward?

For each basin:
    ceiling_i = 1 - (M_i/V_i)*(exp(sigma_i^2)-1)      [verified analytically]
    headroom_i = ceiling_i - actual_NSE_i

Basins with headroom <= 0 are already scoring AT OR ABOVE what a perfect model
could expect against their own gauge -- further modelling effort there is
chasing noise, and any apparent gain is fitting gauge error.

Reports the ensemble prediction, not a single member, since that is what the
campaign's headline number uses.
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
    ap.add_argument("--glob", default="gpu1080/dumps/camels531ls_*_nhlstm_s*.csv.gz")
    ap.add_argument("--members", default=None,
                    help="comma-separated member names; restricts the ensemble")
    ap.add_argument("--s-low", type=float, default=0.30)
    ap.add_argument("--s-high", type=float, default=0.18)
    ap.add_argument("--s-high-flashy", type=float, default=0.35)
    a = ap.parse_args()

    files = [f for f in glob.glob(a.glob)
             if "_TRAIN_" not in f and "QUARANTINE" not in f]
    if a.members:
        keep = set(a.members.split(","))
        files = [f for f in files
                 if f.split("camels531ls_")[1].split("_nhlstm_")[0] in keep]
    if not files:
        raise SystemExit("no dumps matched -- refusing to score an empty ensemble")
    print(f"averaging {len(files)} member dumps into an ensemble prediction")
    acc = None
    for f in files:
        d = pd.read_csv(f, usecols=["station_id", "t0", "h", "truth", "ymed"])
        d = d[d.h == 1][["station_id", "t0", "truth", "ymed"]]
        acc = d if acc is None else acc.merge(
            d[["station_id", "t0", "ymed"]], on=["station_id", "t0"],
            suffixes=("", f"_{len(acc.columns)}"))
    pred_cols = [c for c in acc.columns if c.startswith("ymed")]
    acc["ens"] = acc[pred_cols].mean(axis=1)
    print(f"ensemble rows {len(acc):,}, basins {acc.station_id.nunique()}, "
          f"streams {len(pred_cols)}\n")

    flash = acc.groupby("station_id")["truth"].apply(
        lambda s: np.percentile(s[s > 0], 99) / np.median(s[s > 0])
        if (s > 0).sum() > 20 else np.nan)
    thr = flash.median()

    rows = []
    for sid, g in acc.groupby("station_id"):
        q = g["truth"].to_numpy(float)
        s_hi = a.s_high_flashy if flash.get(sid, 0) > thr else a.s_high
        sig = sigma_series(q, a.s_low, s_hi)
        pos = np.isfinite(q)
        M = (q[pos] ** 2).mean()
        V = ((q[pos] - q[pos].mean()) ** 2).mean()
        s2 = np.nanmean(sig ** 2)
        ceil = 1 - (M / V) * (np.exp(s2) - 1) if V > 0 else np.nan
        rows.append((sid, nse(g["truth"], g["ens"]), ceil,
                     flash.get(sid, np.nan)))

    df = pd.DataFrame(rows, columns=["sid", "nse", "ceiling", "flash"]).dropna()
    df["headroom"] = df["ceiling"] - df["nse"]

    print(f"median actual NSE   {df.nse.median():.4f}")
    print(f"median ceiling      {df.ceiling.median():.4f}")
    print(f"median headroom     {df.headroom.median():+.4f}\n")

    at = (df.headroom <= 0).sum()
    print(f"basins already AT/ABOVE their gauge-error ceiling: "
          f"{at}/{len(df)} ({100*at/len(df):.1f}%)")
    print("  -> further modelling effort on these is chasing gauge noise\n")

    for lo, hi, lab in [(-9, 0, "at/above ceiling"), (0, .05, "headroom <0.05"),
                        (.05, .2, "headroom 0.05-0.2"), (.2, 9, "headroom >0.2")]:
        sel = df[(df.headroom > lo) & (df.headroom <= hi)]
        if len(sel):
            print(f"  {lab:<20} n={len(sel):>3}  median NSE {sel.nse.median():.3f}"
                  f"  median flashiness {sel.flash.median():.1f}")

    print("\nNOTE: our own derivation. Scenario choice drives these numbers; the")
    print("mix of well-gauged vs extrapolated ratings in CAMELS-531 is UNKNOWN.")


if __name__ == "__main__":
    main()
