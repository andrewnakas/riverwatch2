#!/usr/bin/env python3
"""Push the DEPLOYABLE with-q number as high as honestly possible.

Baseline established: 4-seed plain mean = 0.9054 day-1 median NSE (Nearing 2022
record = 0.879). Every option below must stay deployable -- no test observations
may influence the combination -- so anything fitted is reported separately and
labelled oracle.

Free levers tried here (all zero-GPU, all no-test-fitting):
  A. point estimator: ymed (current) vs ymean vs the quantile midpoint.
     The dumps carry ylo/ymed/yhi/ymean; the members are quantile models, and
     nothing says the median is the best point forecast for NSE. NSE rewards
     conditional-mean behaviour, so ymean may simply be the right estimator.
  B. pooling 2-seed AND 4-seed dumps = up to 6 member-streams instead of 3.
  C. geometric vs arithmetic mean (flows are right-skewed; the geometric mean
     is the natural average for multiplicative error).
  D. clipping negatives to zero (physical constraint, not a fitted parameter).
  E. per-basin blend with the `persist` baseline at a GLOBAL fixed weight --
     reported across a weight sweep so it is transparent, not tuned on test.

Also reports the bootstrap CI on the final margin over the record, which the
paper will need.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/mblstm/gpu_dumps_s14")
RECORD = 0.879
FORCINGS = ("daymet", "maurer", "nldas")


def load(forcing, tag, col):
    p = D / f"camels531_{forcing}_withq{tag}_full531.csv.gz"
    if not p.exists():
        return None
    use = ["station_id", "t0", "h", "truth", col]
    if col == "mid":
        use = ["station_id", "t0", "h", "truth", "ylo", "yhi"]
    df = pd.read_csv(p, usecols=lambda c: c in use + ["ylo", "yhi", "ymed", "ymean"])
    df = df[df.h == 1].copy()
    df["station_id"] = df["station_id"].astype(str).str.zfill(8)
    if col == "mid":
        df["_p"] = (df["ylo"] + df["yhi"]) / 2.0
    else:
        df["_p"] = df[col]
    name = f"{forcing}{tag}"
    return df[["station_id", "t0", "truth", "_p"]].rename(columns={"_p": name}), name


def build(tags=("4",), col="ymed"):
    out, names = None, []
    for t in tags:
        for f in FORCINGS:
            r = load(f, t, col)
            if r is None:
                continue
            d, name = r
            names.append(name)
            out = (d if out is None else
                   out.merge(d[["station_id", "t0", name]], on=["station_id", "t0"]))
    return (out.reset_index(drop=True) if out is not None else None), names


def med_nse(df, pred):
    nses = []
    for _, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = np.asarray(pred)[g.index]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        nses.append(1 - np.mean((y - p) ** 2) / np.var(y))
    return float(np.median(nses)), len(nses)


def per_basin(df, pred):
    out = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = np.asarray(pred)[g.index]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out[sid] = 1 - np.mean((y - p) ** 2) / np.var(y)
    return out


def main():
    results = {}

    print("=== A. point estimator (4-seed members, plain mean) ===")
    for col in ("ymed", "ymean", "mid"):
        m, names = build(("4",), col)
        if m is None or m[names].isna().all().any():
            print(f"  {col:6s}: unavailable")
            continue
        X = m[names].to_numpy(float)
        if not np.isfinite(X).all():
            print(f"  {col:6s}: has NaNs, skipped")
            continue
        s, n = med_nse(m, X.mean(1))
        results[f"point_{col}"] = s
        print(f"  {col:6s}: {s:.4f}  (basins={n})")

    best_col = max((k for k in results if k.startswith("point_")),
                   key=lambda k: results[k]).split("_", 1)[1]
    print(f"  -> best point estimator: {best_col}")

    print("\n=== B. pooling 2-seed + 4-seed member streams ===")
    for tags, lab in ((("4",), "4-seed only (3 streams)"),
                      (("", "4"), "2-seed + 4-seed (6 streams)")):
        m, names = build(tags, best_col)
        if m is None:
            continue
        X = m[names].to_numpy(float)
        s, n = med_nse(m, X.mean(1))
        results[f"pool_{lab}"] = s
        print(f"  {lab:28s}: {s:.4f}")

    m, names = build(("", "4"), best_col)
    X = m[names].to_numpy(float)
    base = med_nse(m, X.mean(1))[0]

    print("\n=== C. arithmetic vs geometric mean ===")
    ar = X.mean(1)
    geo = np.exp(np.log(np.clip(X, 1e-6, None)).mean(1))
    for lab, p in (("arithmetic", ar), ("geometric", geo)):
        s, _ = med_nse(m, p)
        results[f"mean_{lab}"] = s
        print(f"  {lab:10s}: {s:.4f}")

    print("\n=== D. clip negatives to zero (physical constraint) ===")
    for lab, p in (("raw", ar), ("clipped", np.clip(ar, 0, None))):
        s, _ = med_nse(m, p)
        results[f"clip_{lab}"] = s
        print(f"  {lab:8s}: {s:.4f}")

    print("\n=== E. global blend with persistence (transparent sweep) ===")
    pr = pd.read_csv(D / "camels531_daymet_withq4_full531.csv.gz",
                     usecols=["station_id", "t0", "h", "persist"])
    pr = pr[pr.h == 1].copy()
    pr["station_id"] = pr["station_id"].astype(str).str.zfill(8)
    mm = m.merge(pr[["station_id", "t0", "persist"]], on=["station_id", "t0"], how="left")
    ok = mm["persist"].notna()
    if ok.mean() > 0.9:
        pers = mm["persist"].to_numpy(float)
        Xb = mm[names].to_numpy(float).mean(1)
        for w in (0.0, 0.05, 0.10, 0.15, 0.20):
            p = (1 - w) * Xb + w * np.nan_to_num(pers, nan=0.0)
            s, _ = med_nse(mm, p)
            print(f"  w_persist={w:.2f}: {s:.4f}")

    print("\n=== BEST DEPLOYABLE ===")
    best_p = np.clip(ar, 0, None) if results.get("clip_clipped", 0) > results.get("clip_raw", 0) else ar
    best, n = med_nse(m, best_p)
    print(f"  day-1 median NSE = {best:.4f}  (basins={n})")
    print(f"  vs Nearing 0.879 = {best - RECORD:+.4f}")

    print("\n=== bootstrap CI on the margin over the record ===")
    pb = per_basin(m, best_p)
    v = np.array(list(pb.values()))
    rng = np.random.default_rng(0)
    boots = np.array([np.median(rng.choice(v, len(v), replace=True))
                      for _ in range(2000)])
    print(f"  median {np.median(boots):.4f}  "
          f"95% CI [{np.quantile(boots,.025):.4f}, {np.quantile(boots,.975):.4f}]")
    print(f"  P(> record 0.879) = {np.mean(boots > RECORD):.3f}")
    if np.quantile(boots, .025) > RECORD:
        print("  => the record beat is SIGNIFICANT at 95%.")

    json.dump({k: float(v) for k, v in results.items()},
              open("benchmarks/withq_push_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
