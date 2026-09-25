#!/usr/bin/env python3
"""LEDGER 45 / TRACK 3 — WEIGHT ESTIMATION: THE INTERIOR DOOR.

⛔⛔ NO-SHIP / NO-TUNE CLAUSE — pre-registered BEFORE any number was computed:
    Sections marked ORACLE fit weights on HELD-OUT rows. They are ILLEGAL by
    construction and bound headroom ONLY. No oracle may be shipped, and NO
    hyper-parameter of a legal estimator may be selected from any held-out
    number. Every legal estimator here is PARAMETER-FREE by construction,
    because ledger 44 proved a regularizer cannot be tuned in-sample.

TWO FRAMES, KEPT SEPARATE (the mistake caught on the first run)
--------------------------------------------------------------
Joining the 11 pool members INNER-joins rows away (2.75M -> 2.36M): `aorc`
alone truncates the test window. So the shipped-9 score on the POOL frame is
NOT the record. Therefore:
  * BASE frame  = the 9 shipped streams only  -> the ANCHOR must reproduce
    0.8362893021622821 exactly, and it is the frame on which a RECORD-RELEVANT
    change to the shipped ensemble (E1) must be judged.
  * POOL frame  = base INNER-JOIN the 11 members -> pool comparisons live here,
    always PAIRED within the frame.

E1 IS A SPECIFICATION FIX, NOT A TUNING KNOB
--------------------------------------------
Production fits weights on POOLED RAW MSE, so basins with large flows dominate
the fit. The METRIC is a MEDIAN OF PER-BASIN NSE, where each basin is
normalised by its own variance and counts once. E1 weights fit rows by
1/var(y_basin), aligning the fitted objective with the scored functional. It is
legal (train rows only) and parameter-free.
"""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
import gate_eval as G                                               # noqa: E402
from phase_lever import load_allh, VAL_START, VAL_END               # noqa: E402
from rescore9 import glob_seeds                                     # noqa: E402
from ledger44_gate_transfer import (build_streams, merge, per_basin_nse,
                                    TEST_END, THETA, LAM)           # noqa: E402
from ledger44_breadth_allleads import KEPT, add                     # noqa: E402

RECORD = 0.8362893021622821
CACHE = Path("/tmp/l45_cache"); CACHE.mkdir(exist_ok=True)


def build_base():
    f1, f2 = CACHE / "base_tr.pkl", CACHE / "base_te.pkl"
    if f1.exists() and f2.exists():
        return pd.read_pickle(f1), pd.read_pickle(f2)
    print("building BASE streams (9, all leads)...", file=sys.stderr)
    TR, TE = merge(build_streams(True)), merge(build_streams(False))
    TR.to_pickle(f1); TE.to_pickle(f2)
    return TR, TE


def build_pool(TR, TE, pool):
    f1, f2 = CACHE / f"pool{len(pool)}_tr.pkl", CACHE / f"pool{len(pool)}_te.pkl"
    if f1.exists() and f2.exists():
        return pd.read_pickle(f1), pd.read_pickle(f2)
    for t in pool:
        TR = add(TR, t, True); TE = add(TE, t, False)
    TR.to_pickle(f1); TE.to_pickle(f2)
    return TR, TE


def cols_of(df):
    ship = [c for c in df.columns if c not in ("station_id", "date", "h", "truth")
            and not c.startswith("pool_")]
    return ship, ship + [c for c in df.columns if c.startswith("pool_")]


# ---------------------------------------------- sufficient statistics
def basin_stats(fit, cols):
    """Per-basin SSE[b,c], N[b], VAR[b]. A basin-block bootstrap is exactly a
    resample of THESE rows, so one draw costs O(531x20), not O(2.4M x 20)."""
    y = fit["truth"].to_numpy(float)
    P = fit[cols].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(P).all(axis=1)
    codes, uniq = pd.factorize(fit["station_id"].to_numpy())
    nb = len(uniq); c_ok = codes[ok]
    e2 = (P[ok] - y[ok, None]) ** 2
    SSE = np.empty((nb, len(cols)))
    for j in range(len(cols)):
        SSE[:, j] = np.bincount(c_ok, weights=e2[:, j], minlength=nb)
    N = np.bincount(c_ok, minlength=nb).astype(float)
    VAR = pd.Series(y[ok]).groupby(c_ok).var().reindex(np.arange(nb)).to_numpy()
    return SSE, N, VAR


def mse_stats(SSE, N, VAR, sel=None, varnorm=False):
    S, n, v = (SSE, N, VAR) if sel is None else (SSE[sel], N[sel], VAR[sel])
    if varnorm:
        g = np.isfinite(v) & (v > 1e-9)
        S, n = S[g] / v[g][:, None], n[g] / v[g]
    return S.sum(axis=0) / n.sum()


def w_from_mse(mse, theta=THETA, lam=LAM):
    raw = mse ** (-theta); raw = raw / raw.sum(); k = len(mse)
    w = lam * np.ones(k) / k + (1 - lam) * raw
    return w / w.sum()


def w_prod(fit, cols):                 # production: pooled raw MSE
    return w_from_mse(mse_stats(*basin_stats(fit, cols)))


def w_e1(fit, cols):                   # E1: variance-normalised (metric-aligned)
    return w_from_mse(mse_stats(*basin_stats(fit, cols), varnorm=True))


def w_e2(fit, cols, n_boot=400, seed=20260828, varnorm=False):
    """E2: James-Stein shrinkage of the MSE ESTIMATES, factor ESTIMATED from
    their measured sampling variance -- parameter-free."""
    S, N, V = basin_stats(fit, cols)
    lm = np.log(mse_stats(S, N, V, varnorm=varnorm)); nb = len(N)
    rng = np.random.default_rng(seed)
    B = np.array([np.log(mse_stats(S, N, V, sel=rng.integers(0, nb, nb), varnorm=varnorm))
                  for _ in range(n_boot)])
    Vv = float(np.var(B, axis=0, ddof=1).mean()); k = len(cols)
    spread = float(np.sum((lm - lm.mean()) ** 2))
    js = max(0.0, 1.0 - (k - 2) * Vv / spread) if spread > 0 else 0.0
    return w_from_mse(np.exp(lm.mean() + js * (lm - lm.mean()))), \
        {"js_factor": js, "logmse_sampling_var": Vv, "logmse_spread": spread}


def w_lsq(fit, cols, varnorm=False, ridge=1e-6):
    """ORACLE-CLASS: best sum-to-1 linear combination. varnorm targets NSE."""
    y = fit["truth"].to_numpy(float); P = fit[cols].to_numpy(float)
    m = np.isfinite(y) & np.isfinite(P).all(axis=1)
    y, P = y[m], P[m]
    if varnorm:
        v = fit.groupby("station_id")["truth"].transform("var").to_numpy(float)[m]
        sw = np.sqrt(1.0 / np.where(np.isfinite(v) & (v > 1e-9), v, np.inf))
    else:
        sw = np.ones(len(y))
    Pw, yw = P * sw[:, None], y * sw
    A, b = Pw[:, :-1] - Pw[:, -1:], yw - Pw[:, -1]
    wr = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)
    return np.append(wr, 1.0 - wr.sum())


def sc_nse(sc, cols, w):
    d = sc.copy(); d["_e"] = (d[cols].to_numpy(float) * w).sum(axis=1)
    return per_basin_nse(d, "_e")


def frames(TR, TE, frame):
    f = (lambda d: d[d.h == 1]) if frame == "h1" else (lambda d: d)
    sc = f(TE); return f(TR), sc[sc.date <= TEST_END]


def boot_ci(d, n=4000, seed=20260828):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    bs = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rep(tag, base, cand, store):
    i = base.index.intersection(cand.index)
    d = (cand[i] - base[i]).to_numpy(); lo, hi = boot_ci(d)
    r = {"median": float(cand[i].median()), "base_median": float(base[i].median()),
         "paired": float(np.median(d)), "breadth": float((d > 0).mean()),
         "boot95": [lo, hi], "n": int(len(i)),
         "ci_excl_0": bool(lo > 0 or hi < 0), "clears_0.003": bool(lo >= 0.003)}
    store[tag] = r
    print(f"  {tag:<32} med {r['median']:.6f} paired {r['paired']:+.6f} "
          f"[{lo:+.6f},{hi:+.6f}] br {r['breadth']:.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--out", default="benchmarks/ledger45_weightest.json")
    args = ap.parse_args()

    BTR, BTE = build_base()
    out = {"no_ship_clause": "ORACLE rows fit weights on held-out data; illegal by "
           "construction, headroom bound only. No oracle shipped; no legal "
           "estimator parameter selected from a held-out number.",
           "record": RECORD, "theta": THETA, "lam": LAM}

    # ================================================= ANCHOR (BASE frame)
    print("=" * 78 + "\nANCHOR — shipped 9, production rule, h==1, BASE frame\n" + "=" * 78)
    ship0, _ = cols_of(BTR)
    f0, s0 = frames(BTR, BTE, "h1")
    a = sc_nse(s0, ship0, w_prod(f0, ship0)); am = float(a.median())
    ok = abs(am - RECORD) < 1e-9
    print(f"  got    {am:.16f}\n  record {RECORD:.16f}   diff {am-RECORD:+.2e}   "
          f"{'REPRODUCES' if ok else '*** MISMATCH ***'}")
    # the sufficient-statistic fast path must be identical to the direct one
    S, N, V = basin_stats(f0, ship0)
    y = f0["truth"].to_numpy(float); P = f0[ship0].to_numpy(float)
    direct = np.nanmean((P - y[:, None]) ** 2, axis=0)
    rd = float(np.max(np.abs(mse_stats(S, N, V) - direct) / np.abs(direct)))
    print(f"  stats-path vs direct MSE: max rel diff {rd:.2e}")
    out["anchor"] = {"median": am, "diff": am - RECORD, "reproduces": ok,
                     "stats_path_max_rel_diff": rd,
                     "base_rows": int(len(BTR)), "test_rows_h1": int(len(s0))}
    if not ok or rd > 1e-9:
        print("  ⚠️ anchor or stats path failed — downstream numbers are suspect.")

    # ============ RECORD-RELEVANT: E1 on the SHIPPED 9, full BASE frame ======
    print("\n" + "=" * 78)
    print("A — E1 (variance-normalised fit) ON THE SHIPPED 9, BASE FRAME")
    print("    This is the only test here that could move the RECORD itself.")
    print("=" * 78)
    for fr in ("h1", "all"):
        fit, sc = frames(BTR, BTE, fr)
        base = sc_nse(sc, ship0, w_prod(fit, ship0))
        st = out.setdefault(f"base_{fr}", {})
        st["_baseline_median"] = float(base.median())
        print(f"  [{fr}] baseline {float(base.median()):.6f}  "
              f"fit rows {len(fit):,}  score rows {len(sc):,}")
        rep("ship9_E1_varnorm", base, sc_nse(sc, ship0, w_e1(fit, ship0)), st)
        w2, i2 = w_e2(fit, ship0, n_boot=args.nboot); st["ship9_E2_info"] = i2
        rep("ship9_E2_js", base, sc_nse(sc, ship0, w2), st)
        w2v, i2v = w_e2(fit, ship0, n_boot=args.nboot, varnorm=True)
        st["ship9_E2var_info"] = i2v
        rep("ship9_E1+E2_js_varnorm", base, sc_nse(sc, ship0, w2v), st)
        print("  -- ORACLE (illegal; bounds only) --")
        rep("ORACLE_ship9_invmse_on_test", base, sc_nse(sc, ship0, w_prod(sc, ship0)), st)
        rep("ORACLE_ship9_lsq_on_test", base, sc_nse(sc, ship0, w_lsq(sc, ship0)), st)
        rep("ORACLE_ship9_lsqvar_on_test", base,
            sc_nse(sc, ship0, w_lsq(sc, ship0, varnorm=True)), st)

    # ====================================== POOL frame (breadth question) ====
    pool = list(KEPT)
    PTR, PTE = build_pool(BTR, BTE, pool)
    ship, allc = cols_of(PTR)
    lost = 1 - len(PTR) / len(BTR)
    print("\n" + "=" * 78)
    print(f"B — THE 19-STREAM POOL, POOL FRAME ({len(allc)} streams; "
          f"inner joins dropped {lost*100:.1f}% of base rows)")
    print("    ⚠️ baseline here is NOT the record — different frame. Paired only.")
    print("=" * 78)
    out["pool"] = {"members": pool, "n_streams": len(allc),
                   "pool_rows": int(len(PTR)), "frac_rows_lost_vs_base": lost}

    for fr in ("h1", "all"):
        fit, sc = frames(PTR, PTE, fr)
        fit_all, _ = frames(PTR, PTE, "all")
        base = sc_nse(sc, ship, w_prod(fit, ship))
        st = out.setdefault(f"pool_{fr}", {})
        st["_baseline_median"] = float(base.median())
        print(f"\n  [{fr}] baseline(ship9,pool frame) {float(base.median()):.6f}  "
              f"fit {len(fit):,}  score {len(sc):,}")
        print("  -- LEGAL --")
        rep("pool_production", base, sc_nse(sc, allc, w_prod(fit, allc)), st)
        rep("pool_E1_varnorm", base, sc_nse(sc, allc, w_e1(fit, allc)), st)
        wp, ip = w_e2(fit, allc, n_boot=args.nboot); st["pool_E2_info"] = ip
        rep("pool_E2_js", base, sc_nse(sc, allc, wp), st)
        wpv, ipv = w_e2(fit, allc, n_boot=args.nboot, varnorm=True)
        st["pool_E2var_info"] = ipv
        rep("pool_E1+E2_js_varnorm", base, sc_nse(sc, allc, wpv), st)
        rep("ship9_E1_varnorm", base, sc_nse(sc, ship, w_e1(fit, ship)), st)
        if fr == "h1":     # E3: mix toward the lower-variance all-leads estimate
            wa, wh = w_prod(fit_all, allc), w_prod(fit, allc)
            for al in (0.25, 0.5, 0.75, 1.0):
                rep(f"pool_E3_framemix_a{al}", base,
                    sc_nse(sc, allc, (1 - al) * wh + al * wa), st)
        print("  -- ORACLE (illegal; bounds only) --")
        rep("ORACLE_pool_invmse_on_test", base, sc_nse(sc, allc, w_prod(sc, allc)), st)
        rep("ORACLE_pool_lsq_on_test", base, sc_nse(sc, allc, w_lsq(sc, allc)), st)
        rep("ORACLE_pool_lsqvar_on_test", base,
            sc_nse(sc, allc, w_lsq(sc, allc, varnorm=True)), st)

    # ============================== S1: how noisy IS the weight vector? =======
    print("\n" + "=" * 78 + "\nS1 — SAMPLING NOISE OF THE WEIGHT VECTOR (legal, train rows)\n" + "=" * 78)
    s1 = {}
    for fr in ("h1", "all"):
        fit, _ = frames(PTR, PTE, fr)
        S, N, V = basin_stats(fit, allc); nb = len(N)
        rng = np.random.default_rng(20260828)
        W = np.array([w_from_mse(mse_stats(S, N, V, sel=rng.integers(0, nb, nb)))
                      for _ in range(args.nboot)])
        wm, ws = W.mean(axis=0), W.std(axis=0, ddof=1)
        rel = float(np.mean(ws / np.maximum(wm, 1e-12)))
        s1[fr] = {"fit_rows": int(len(fit)), "mean_rel_sd": rel,
                  "max_rel_sd": float(np.max(ws / np.maximum(wm, 1e-12)))}
        print(f"  [{fr}] fit rows {len(fit):>9,}  mean rel-SD of {len(allc)} weights "
              f"{rel:.4f}  max {s1[fr]['max_rel_sd']:.4f}")
    s1["h1_over_all_ratio"] = s1["h1"]["mean_rel_sd"] / s1["all"]["mean_rel_sd"]
    print(f"  => h==1 weight noise is {s1['h1_over_all_ratio']:.2f}x all-leads")
    out["S1_weight_noise"] = s1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}  md5 "
          f"{hashlib.md5(open(args.out,'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
