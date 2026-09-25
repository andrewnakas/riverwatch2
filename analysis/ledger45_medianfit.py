#!/usr/bin/env python3
"""LEDGER 45 / T3b — MAXIMISE THE MEDIAN DIRECTLY.

⛔ NO-SHIP: ORACLE_* arms fit on held-out rows. Bounds only.

T3 showed every MSE-based oracle buys ~0 at h==1, and that objectives targeting a
MEAN drag the MEDIAN down. So none of them bounds what a linear combination can
do. This maximises the scored functional itself.

EXACT: MSE_b(w) = w'A_b w - 2 b_b'w + c_b ; NSE_b = 1 - MSE_b/var_b.
Precompute (A_b, b_b, c_b, var_b) once -> one median evaluation is O(531 k^2).
"""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger45_weightest2 import (build_base, build_pool, cols_of, frames, w_prod,
                                sc_nse, per_basin_nse, boot_ci, rep, RECORD, KEPT)

CACHE = Path("/tmp/l45_cache")


def quad_forms(df, cols):
    """Per-basin (A_b, b_b, c_b, var_b, id). Exact, no approximation."""
    y = df["truth"].to_numpy(float); P = df[cols].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(P).all(axis=1)
    y, P = y[ok], P[ok]
    codes, uniq = pd.factorize(df["station_id"].to_numpy()[ok])
    k = len(cols); A, B, C, V, ids = [], [], [], [], []
    for i, sid in enumerate(uniq):
        m = codes == i
        if m.sum() < 20:
            continue
        Pb, yb = P[m], y[m]
        v = yb.var()
        if not np.isfinite(v) or v < 1e-9:
            continue
        A.append(Pb.T @ Pb / len(yb)); B.append(Pb.T @ yb / len(yb))
        C.append(float((yb ** 2).mean())); V.append(float(v)); ids.append(sid)
    return (np.array(A), np.array(B), np.array(C), np.array(V), np.array(ids))


def nse_vec(w, Q):
    A, B, C, V, _ = Q
    mse = np.einsum("i,bij,j->b", w, A, w) - 2 * (B @ w) + C
    return 1.0 - mse / V


def med(w, Q):
    return float(np.median(nse_vec(w, Q)))


def simplex(z):
    e = np.exp(z - z.max()); return e / e.sum()


def maximise_median(Q, k, w0, n_restart=12, seed=20260828):
    """Nelder-Mead on a softmax parameterisation, multi-start, best kept."""
    rng = np.random.default_rng(seed)
    best_w, best_v = w0.copy(), med(w0, Q)
    starts = [np.log(np.clip(w0, 1e-6, None))]
    starts += [np.log(np.clip(w0, 1e-6, None)) + rng.normal(0, s, k)
               for s in np.linspace(0.3, 2.0, n_restart - 1)]
    for z0 in starts:
        r = minimize(lambda z: -med(simplex(z), Q), z0, method="Nelder-Mead",
                     options={"maxiter": 20000, "maxfev": 20000,
                              "xatol": 1e-8, "fatol": 1e-10})
        w = simplex(r.x); v = med(w, Q)
        if v > best_v:
            best_w, best_v = w, v
    return best_w, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/ledger45_medianfit.json")
    args = ap.parse_args()
    BTR, BTE = build_base()
    out = {"no_ship_clause": "ORACLE_median is fit on held-out rows: a bound, "
           "never shipped; no legal rule tuned from it.", "record": RECORD}

    for label, (TR, TE, use_pool) in {
            "BASE": (BTR, BTE, False),
            "POOL": (None, None, True)}.items():
        if use_pool:
            TR, TE = build_pool(BTR, BTE, list(KEPT))
        ship, allc = cols_of(TR)
        cols = ship if label == "BASE" else allc
        for fr in ("h1", "all"):
            fit, sc = frames(TR, TE, fr)
            k = len(cols)
            Qtr, Qte = quad_forms(fit, cols), quad_forms(sc, cols)
            w0 = w_prod(fit, cols)
            base = sc_nse(sc, cols, w0)
            st = out.setdefault(f"{label}_{fr}", {})
            st["_baseline_median"] = float(base.median())
            st["_n_streams"] = k
            # exactness check: quad-form median must equal the scored median
            qm, sm = med(w0, Qte), float(base.median())
            st["quadform_check_absdiff"] = abs(qm - sm)
            print(f"\n[{label} {fr}] k={k} baseline {sm:.6f}  "
                  f"quad-form check |Δ|={abs(qm-sm):.2e}")
            if abs(qm - sm) > 1e-9:
                print("  ⚠️ quad-form disagrees with the scorer — arm is invalid.")

            wo, vo = maximise_median(Qte, k, w0)          # ORACLE
            st["oracle_median_trainside_value"] = vo
            rep("ORACLE_median_on_test", base, sc_nse(sc, cols, wo), st)
            wl, vl = maximise_median(Qtr, k, w0)          # LEGAL
            st["legal_median_train_value"] = vl
            st["legal_median_train_gain"] = vl - med(w0, Qtr)
            print(f"  LEGAL train-side median {vl:.6f} "
                  f"(gain {vl-med(w0,Qtr):+.6f}) -> held-out:")
            rep("LEGAL_median_fit_on_train", base, sc_nse(sc, cols, wl), st)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out} md5 "
          f"{hashlib.md5(open(args.out,'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
