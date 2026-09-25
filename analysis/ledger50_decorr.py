#!/usr/bin/env python3
"""LEDGER 50 — price DECORRELATION AT FIXED SKILL, in the same currency as G0b/G0c.

G0b/G0c pull a member toward truth: they price a member becoming BETTER. But the
campaign's one successful member (multi5) worked by PERTURBATION/decorrelation, not
skill, and "decorrelation" has been refuted five times as a value PREDICTOR — always
because architectures bought decorrelation by FITTING WORSE. That is a statement about
the achievable trade-off, never about the value of decorrelation at FIXED skill.

This isolates it. For stream X, per basin, rotate its error vector toward the subspace
orthogonal to the other members' errors, preserving its norm exactly:

    e' = cos(t)*e_X + sin(t)*|e_X| * u,     u ⊥ {e_X, e_others}, |u| = 1

so per-basin MSE (hence solo NSE) is UNCHANGED to floating point, while corr(e', e_j)
falls by cos(t). Then refit the production weights on TRAIN and rescore the record frame.

⛔ BOUNDS/PLANNING ONLY, like G0b/G0c: reads held-out rows to characterise the ensemble's
sensitivity. Nothing is shipped from it and no candidate is selected on test performance.
⚠️ It is an UPPER BOUND on decorrelation's value: a real member cannot choose its error
direction freely, and the rotation is applied to TRAIN and TEST with independent noise,
so the weight fit sees a consistently-decorrelated member (the optimistic case).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger44_gate_transfer import (per_basin_nse, invmse_weights, TEST_END)  # noqa: E402
from ledger45_weightest import boot_ci                                        # noqa: E402

RECORD = 0.8362893021622821
FROZEN = Path.home() / "riverwatch2" / "frozen_base"
META = ("station_id", "date", "h", "truth")


def rotate_errors(df, target, others, theta, rng):
    """Per basin: rotate `target`'s error toward the orthogonal complement of
    {its own error} ∪ {the others' errors}, preserving its norm."""
    out = df[target].to_numpy(float).copy()
    y = df["truth"].to_numpy(float)
    e_t = out - y
    E_o = df[others].to_numpy(float) - y[:, None]
    codes, _ = pd.factorize(df["station_id"], sort=False)
    for b in range(codes.max() + 1):
        m = codes == b
        et = e_t[m]
        n = et.shape[0]
        if n < 5:
            continue
        B = np.column_stack([et, E_o[m]])          # (n, 1+k) span to avoid
        u = rng.standard_normal(n)
        # Gram-Schmidt: remove every component lying in the span of B
        Q, _ = np.linalg.qr(B)
        u = u - Q @ (Q.T @ u)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u /= nu
        e_new = np.cos(theta) * et + np.sin(theta) * np.linalg.norm(et) * u
        out[m] = y[m] + e_new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="lstm_multi")
    ap.add_argument("--thetas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", default="benchmarks/ledger50_decorr.json")
    a = ap.parse_args()
    TR = pd.read_pickle(FROZEN / "base_tr.pkl"); TE = pd.read_pickle(FROZEN / "base_te.pkl")
    cols = [c for c in TR.columns if c not in META]
    fit0 = TR[TR.h == 1].reset_index(drop=True)
    sc0 = TE[(TE.h == 1) & (TE.date <= TEST_END)].reset_index(drop=True)
    w0 = invmse_weights(fit0, cols)
    base = per_basin_nse(sc0.assign(_e=(sc0[cols].to_numpy(float) * w0).sum(1)), "_e")
    print(f"ANCHOR {base.median():.16f}  diff {base.median()-RECORD:+.2e}")
    others = [c for c in cols if c != a.stream]
    res = {"stream": a.stream, "anchor": float(base.median()), "record": RECORD, "rows": {}}
    for th in a.thetas:
        rng = np.random.default_rng(a.seed)
        f, s = fit0.copy(), sc0.copy()
        f[a.stream] = rotate_errors(f, a.stream, others, th, rng)
        s[a.stream] = rotate_errors(s, a.stream, others, th, rng)
        w = invmse_weights(f, cols)
        s["_e"] = (s[cols].to_numpy(float) * w).sum(1)
        cand = per_basin_nse(s, "_e")
        idx = base.index.intersection(cand.index)
        d = (cand[idx] - base[idx]).to_numpy()
        lo, hi = boot_ci(d)
        solo_b = float(per_basin_nse(sc0, a.stream).median())
        solo_c = float(per_basin_nse(s, a.stream).median())
        ec = float(np.mean([np.corrcoef(s[a.stream] - s["truth"], s[o] - s["truth"])[0, 1]
                            for o in others]))
        ec0 = float(np.mean([np.corrcoef(sc0[a.stream] - sc0["truth"], sc0[o] - sc0["truth"])[0, 1]
                             for o in others]))
        row = {"theta": th, "cos": float(np.cos(th)), "paired": float(np.median(d)),
               "breadth": float((d > 0).mean()), "boot95": [lo, hi],
               "solo_before": solo_b, "solo_after": solo_c, "solo_shift": solo_c - solo_b,
               "mean_err_corr_before": ec0, "mean_err_corr_after": ec,
               "weight_before": float(w0[cols.index(a.stream)]),
               "weight_after": float(w[cols.index(a.stream)])}
        res["rows"][f"theta_{th:.2f}"] = row
        print(f"  theta={th:.2f} cos={np.cos(th):.3f} | solo {solo_b:.4f}->{solo_c:.4f} "
              f"({solo_c-solo_b:+.4f}) | err-corr {ec0:.4f}->{ec:.4f} | w {w0[cols.index(a.stream)]:.4f}"
              f"->{w[cols.index(a.stream)]:.4f} | ensemble {np.median(d):+.6f} "
              f"[{lo:+.6f},{hi:+.6f}] breadth {(d>0).mean():.3f}")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
