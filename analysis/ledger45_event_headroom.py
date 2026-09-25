#!/usr/bin/env python3
"""LEDGER 45 / T7 — HOW MUCH OF THE EVENT-DAY ERROR IS *MODEL* ERROR?

T4 measured per-gauge sigma from 263,524 USGS gaugings and overturned the
campaign's assumed shape: high flow is CLEANER (0.207) than low flow (0.239),
where the assumed triple had high flow as the NOISIEST band (0.35). Every peak
lever was rejected while the campaign believed peaks were mostly gauge noise.

So price it. Per basin, per flow band, on the HELD-OUT test rows:
    E_band = mean( (ens - y)^2 )                      total squared error
    N_band = mean( y^2 * (exp(sigma_band^2) - 1) )    irreducible gauge variance
                                                      (multiplicative lognormal)
    irreducible_fraction = N_band / E_band
A small fraction at high flow means most peak error is MODEL error -> headroom.
A fraction near 1 means peaks really are noise-limited and idea (2) is dead.

Reported for MEASURED sigma and for the OLD ASSUMED triple, so the change in
belief is visible as a number rather than asserted.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

H = Path.home() / "riverwatch2"
sys.path.insert(0, str(H))
from ledger45_weightest2 import build_base, cols_of, frames, w_prod, RECORD

SIG = json.load(open(H / "analysis/ledger45_gauge_sigma.json"))["stations"]
BANDS = ("low", "mid", "high")
ASSUMED = {"low": 0.30, "mid": 0.18, "high": 0.35}


def main():
    BTR, BTE = build_base()
    ship, _ = cols_of(BTR)
    fit, sc = frames(BTR, BTE, "h1")
    w = w_prod(fit, ship)
    sc = sc.copy()
    sc["_ens"] = (sc[ship].to_numpy(float) * w).sum(axis=1)

    rows = []
    for sid, g in sc.groupby("station_id"):
        y = g["truth"].to_numpy(float); p = g["_ens"].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(p) & (y > 0)
        if m.sum() < 100:
            continue
        y, p = y[m], p[m]
        s = SIG.get(sid, {}).get("sigma_empirical") or {}
        if any(s.get(b) is None for b in BANDS):
            continue
        q1, q2 = np.percentile(y, [33.333, 66.667])
        idx = {"low": y <= q1, "mid": (y > q1) & (y <= q2), "high": y > q2}
        r = {"sid": sid}
        for b in BANDS:
            k = idx[b]
            if k.sum() < 20:
                r = None; break
            E = float(np.mean((p[k] - y[k]) ** 2))
            sm_, sa_ = float(s[b]), ASSUMED[b]
            Nm = float(np.mean(y[k] ** 2 * (np.exp(sm_ ** 2) - 1)))
            Na = float(np.mean(y[k] ** 2 * (np.exp(sa_ ** 2) - 1)))
            if E <= 0:
                r = None; break
            r[f"{b}_E"] = E
            r[f"{b}_frac_measured"] = Nm / E
            r[f"{b}_frac_assumed"] = Na / E
            r[f"{b}_share_of_total_err"] = E * k.sum()
        if r:
            rows.append(r)

    D = pd.DataFrame(rows)
    print(f"basins with measured sigma and full band coverage: {len(D)}")
    tot = sum(D[f"{b}_share_of_total_err"].sum() for b in BANDS)
    out = {"record": RECORD, "n_basins": int(len(D)), "bands": {}}
    print(f"\n{'band':<6} {'irreducible frac':>17} {'(assumed)':>11} "
          f"{'% of ALL sq err':>16}")
    for b in BANDS:
        fm = float(D[f"{b}_frac_measured"].median())
        fa = float(D[f"{b}_frac_assumed"].median())
        share = float(D[f"{b}_share_of_total_err"].sum() / tot)
        out["bands"][b] = {"irreducible_fraction_measured": fm,
                           "irreducible_fraction_assumed": fa,
                           "share_of_total_squared_error": share,
                           "p25": float(D[f"{b}_frac_measured"].quantile(.25)),
                           "p75": float(D[f"{b}_frac_measured"].quantile(.75))}
        print(f"{b:<6} {fm:>17.3f} {fa:>11.3f} {share*100:>15.1f}%")
    hi = out["bands"]["high"]
    print(f"\nHIGH-FLOW: measured irreducible fraction {hi['irreducible_fraction_measured']:.3f} "
          f"(IQR {hi['p25']:.3f}-{hi['p75']:.3f})")
    print(f"  => {(1-hi['irreducible_fraction_measured'])*100:.1f}% of high-flow "
          f"squared error is MODEL error, and high flow carries "
          f"{hi['share_of_total_squared_error']*100:.1f}% of all squared error")
    print(f"  under the OLD assumed sigma it looked {hi['irreducible_fraction_assumed']:.3f} "
          f"irreducible")
    p = "benchmarks/ledger45_event_headroom.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p} md5 {hashlib.md5(open(p,'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
