#!/usr/bin/env python3
"""LEDGER 54 — the SEED-DEPTH CURVE of a single member (zero GPU).

Question: can ONE member reach 0.89 by seed averaging alone?

`nhar` reads 0.8714 at one seed and 0.886153 at three — a +0.0148 jump, far
larger than the MB-LSTM members' 1->5 gain of ~+0.002. This enumerates EVERY
k-subset of a member's seeds, medians the per-basin NSE, and fits the standard
variance-reduction law

    median(k) = a - b/k

(a = the infinite-seed asymptote, b/k = the residual seed noise), then
extrapolates. ⚠️ The fit is an EXTRAPOLATION, not a measurement: LEDGER 50 got
this wrong in the other direction by mapping a solo gain through a weight law
([[the-ensemble-sensitivity-map]] §3), so the number here is a PREDICTION to be
scored against seeds 504/505, not a result.

Seed averaging matches app/mblstm.py exactly: average the physical quantile
slots, THEN sort, THEN clip at 0, then read out (ylo+yhi)/2.

usage: l54_seed_depth.py --member nhar --frame test1 [--seeds 501,502,503]
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
DD = "data/mblstm/l51_dumps"
IDS = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
MIN_ROWS = 20


def load_seeds(member, frame, seeds=None):
    """Return (key, truth, {seed: (lo, md, hi)}) on the member's common grid."""
    files = sorted(glob.glob(f"{DD}/camels531_l51_{member}_s*_{frame}.csv.gz"))
    out, key, truth = {}, None, None
    for f in files:
        s = int(f.split("_s")[-1].split("_")[0])
        if seeds is not None and s not in seeds:
            continue
        side = f.replace(".csv.gz", ".json")
        assert os.path.exists(side), f"{f}: no sidecar — refusing"
        meta = json.load(open(side))
        assert meta.get("train_start") == "1999-10-01" and meta.get("train_end") == "2008-09-30", meta
        d = pd.read_csv(f, dtype={"station_id": str})
        d["station_id"] = d.station_id.str.zfill(8)
        d = d[(d.h == 1) & d.station_id.isin(IDS)].sort_values(["station_id", "t0"])
        k = (d.station_id + "|" + d.t0).to_numpy()
        if key is None:
            key, truth = k, d.truth.to_numpy(np.float64)
        else:
            assert np.array_equal(key, k), f"{f}: grid differs from the first seed"
        out[s] = (d.ylo.to_numpy(np.float64), d.ymed.to_numpy(np.float64), d.yhi.to_numpy(np.float64))
    assert out, f"no dumps for {member!r} {frame!r}"
    return key, truth, out


def median_nse(pred, truth, basin_idx):
    v = []
    for ii in basin_idx.values():
        y, p = truth[ii], pred[ii]
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < MIN_ROWS:
            continue
        y, p = y[m], p[m]
        if np.var(y) < 1e-9:
            continue
        v.append(1 - np.mean((y - p) ** 2) / np.var(y))
    return float(np.median(v)), len(v)


def readout(slots):
    q = np.sort(np.stack(slots, axis=1), axis=1)
    q = np.clip(q, 0.0, None)
    return (q[:, 0] + q[:, 2]) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--seeds", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seeds = set(int(x) for x in a.seeds.split(",") if x) or None

    key, truth, per = load_seeds(a.member, a.frame, seeds)
    sids = np.array([k.split("|")[0] for k in key])
    basin_idx = pd.Series(np.arange(len(sids))).groupby(sids).apply(lambda s: s.to_numpy()).to_dict()
    S = sorted(per)
    print(f"member={a.member} frame={a.frame} seeds={S} rows={len(key):,} basins={len(basin_idx)}\n")

    rows = []
    for k in range(1, len(S) + 1):
        vals = []
        for combo in itertools.combinations(S, k):
            slots = [np.mean([per[s][j] for s in combo], axis=0) for j in range(3)]
            med, n = median_nse(readout(slots), truth, basin_idx)
            vals.append(med)
        rows.append({"k": k, "n_subsets": len(vals), "mean": float(np.mean(vals)),
                     "min": float(np.min(vals)), "max": float(np.max(vals))})
        print(f"  k={k}  subsets={len(vals):2d}  mean {np.mean(vals):.6f}  "
              f"[{np.min(vals):.6f}, {np.max(vals):.6f}]")

    res = {"member": a.member, "frame": a.frame, "seeds": S, "rows": int(len(key)),
           "basins": len(basin_idx), "curve": rows}
    if len(rows) >= 2:
        ks = np.array([r["k"] for r in rows], float)
        ys = np.array([r["mean"] for r in rows], float)
        A = np.vstack([np.ones_like(ks), -1.0 / ks]).T
        (a_hat, b_hat), *_ = np.linalg.lstsq(A, ys, rcond=None)
        pred = {int(k): float(a_hat - b_hat / k) for k in (1, 2, 3, 4, 5, 7, 10, 15, 20)}
        resid = ys - (a_hat - b_hat / ks)
        res["fit"] = {"a_infinite_seeds": float(a_hat), "b": float(b_hat),
                      "max_abs_residual": float(np.abs(resid).max()), "extrapolation": pred}
        print(f"\nFIT  median(k) = a - b/k   a={a_hat:.6f}  b={b_hat:.6f}  "
              f"max|resid|={np.abs(resid).max():.2e}")
        for k, v in pred.items():
            flag = "  <- measured" if k <= len(S) else ""
            print(f"   k={k:<3} {v:.6f}{flag}")
        print(f"\n  asymptote (infinite seeds): {a_hat:.6f}")
        for tgt in (0.89, 0.90):
            if a_hat <= tgt:
                print(f"  0.{int(tgt*1000)}: UNREACHABLE by seed averaging alone "
                      f"(asymptote is {a_hat - tgt:+.6f} short)")
            else:
                kneed = b_hat / (a_hat - tgt)
                print(f"  {tgt:.2f}: needs k >= {kneed:.1f} seeds")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
