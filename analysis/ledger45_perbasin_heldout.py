#!/usr/bin/env python3
"""LEDGER 45 / T6 — THE LAST UNBOUNDED COMBINATION CLASS: PER-BASIN, HELD-OUT.

⛔ NO-SHIP CLAUSE (pre-registered): every ORACLE_* arm fits on HELD-OUT rows.
   Illegal by construction, bounds only, never shipped, and no legal rule may be
   selected from any number here.

WHY THIS IS NOT ALREADY CLOSED
------------------------------
T3 bounded GLOBAL linear combination held-out: an oracle fit on the test rows
buys -0.000028 at h==1. But the largest oracle this campaign ever measured is
PER-BASIN weighting: +0.0070 with 83.6% breadth -- and it was measured
TRAIN-SIDE (fit 1980-90, val 1990-95), which ledger 44 proved is IN-SAMPLE for
these networks. Its held-out value has never been measured.

Ledger 44 showed the in-sample bias runs in three directions depending on
mechanism, so it cannot be extrapolated. Per-basin weighting is a
VARIANCE-heavy estimator (9 params per basin on ~345 test rows), and for
ensembling the in-sample surface UNDERSTATES -- but for selection it OVERSTATES.
Only measurement settles it.

⭐ And T3b taught the second half of the question: an oracle can move the median
without helping most basins. So every arm reports BREADTH, not just the delta.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger45_weightest2 import (build_base, cols_of, frames, w_prod, sc_nse,
                                 per_basin_nse, boot_ci, RECORD, THETA)

RNG = np.random.default_rng(20260829)


def per_basin_scores(sc, cols, wfun):
    """wfun(gdf, cols) -> weight vector, applied per basin."""
    out = {}
    for sid, g in sc.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        P = g[cols].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(P).all(axis=1)
        if m.sum() < 20 or np.var(y[m]) < 1e-9:
            continue
        w = wfun(P[m], y[m])
        pred = P[m] @ w
        out[sid] = 1.0 - np.mean((y[m] - pred) ** 2) / np.var(y[m])
    return pd.Series(out)


def w_best_member(P, y):
    mse = np.mean((P - y[:, None]) ** 2, axis=0)
    w = np.zeros(P.shape[1]); w[int(np.argmin(mse))] = 1.0
    return w


def w_invmse_pb(P, y, theta=THETA, lam=0.0):
    mse = np.mean((P - y[:, None]) ** 2, axis=0)
    raw = mse ** (-theta); raw /= raw.sum()
    k = P.shape[1]
    w = lam * np.ones(k) / k + (1 - lam) * raw
    return w / w.sum()


def w_lsq_pb(P, y, ridge=1e-8):
    A = P[:, :-1] - P[:, -1:]
    b = y - P[:, -1]
    wr = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)
    return np.append(wr, 1.0 - wr.sum())


def rep(tag, base, cand, store):
    i = base.index.intersection(cand.index)
    d = (cand[i] - base[i]).to_numpy()
    lo, hi = boot_ci(d)
    r = {"median": float(cand[i].median()), "base": float(base[i].median()),
         "diff_of_medians": float(cand[i].median() - base[i].median()),
         "paired": float(np.median(d)), "breadth": float((d > 0).mean()),
         "boot95": [lo, hi], "n": int(len(i))}
    store[tag] = r
    flag = "" if r["breadth"] > 0.5 else "   <-- NARROW (T3b shape)"
    print(f"  {tag:<34} paired {r['paired']:+.6f}  dom {r['diff_of_medians']:+.6f}"
          f"  breadth {r['breadth']:.3f}{flag}")
    return r


def main():
    BTR, BTE = build_base()
    ship, _ = cols_of(BTR)
    out = {"record": RECORD, "n_streams": len(ship),
           "no_ship_clause": "ORACLE arms fit on held-out rows; bounds only."}

    for fr in ("h1", "all"):
        fit, sc = frames(BTR, BTE, fr)
        base = sc_nse(sc, ship, w_prod(fit, ship))
        st = out.setdefault(fr, {})
        st["_baseline"] = float(base.median())
        rows_pb = int(np.median(sc.groupby("station_id").size()))
        st["_median_rows_per_basin"] = rows_pb
        print(f"\n[{fr}] baseline {float(base.median()):.6f}   "
              f"median test rows/basin {rows_pb}")

        print("  -- ORACLE per-basin, fit on the SAME held-out rows (upper bound) --")
        rep("ORACLE_pb_best_member", base,
            per_basin_scores(sc, ship, w_best_member), st)
        rep("ORACLE_pb_invmse", base,
            per_basin_scores(sc, ship, lambda P, y: w_invmse_pb(P, y)), st)
        rep("ORACLE_pb_lsq", base, per_basin_scores(sc, ship, w_lsq_pb), st)

        # honest split: fit per-basin weights on the FIRST half of the held-out
        # window, score the SECOND half. Still illegal (uses test), but it prices
        # the ESTIMATION cost the pure oracle hides.
        mid = sc["date"].min() + (sc["date"].max() - sc["date"].min()) / 2
        h1f, h2f = sc[sc.date <= mid], sc[sc.date > mid]
        base2 = sc_nse(h2f, ship, w_prod(fit, ship))
        print("  -- SPLIT-HALF: per-basin weights fit on test-half-1, scored on half-2 --")
        for tag, f in (("splithalf_pb_best_member", w_best_member),
                       ("splithalf_pb_invmse", lambda P, y: w_invmse_pb(P, y)),
                       ("splithalf_pb_lsq", w_lsq_pb)):
            wmap = {}
            for sid, g in h1f.groupby("station_id"):
                y = g["truth"].to_numpy(float); P = g[ship].to_numpy(float)
                m = np.isfinite(y) & np.isfinite(P).all(axis=1)
                if m.sum() >= 20:
                    wmap[sid] = f(P[m], y[m])
            sc2 = h2f[h2f.station_id.isin(wmap)].copy()
            preds = np.empty(len(sc2))
            P2 = sc2[ship].to_numpy(float)
            codes, uniq = pd.factorize(sc2["station_id"].to_numpy())
            W = np.array([wmap[u] for u in uniq])
            preds = np.einsum("ij,ij->i", P2, W[codes])
            sc2["_e"] = preds
            rep(tag, base2, per_basin_nse(sc2, "_e"), st)

    p = "benchmarks/ledger45_perbasin_heldout.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p} md5 {hashlib.md5(open(p,'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
