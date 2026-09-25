#!/usr/bin/env python3
"""LEDGER 45 / T4b — MEASURED sigma: (1) re-derive the ceiling, (2) the
pre-GPU KILL TEST for a heteroscedastic label-noise loss.

Uses analysis/ledger45_gauge_sigma.json (263,524 real USGS gaugings) in place of
the ASSUMED triples the ceiling has always rested on.

CEILING — the CORRECTED flow-weighted form (the plain-mean form understated every
ceiling by 0.03-0.04):
    num  = E[y^2 (exp(s(y)^2)-1)]
    den  = E[y^2 exp(s(y)^2)] - (E[y])^2      # Var(OBS), not Var(truth)
    ceil = 1 - num/den

KILL TEST — a heteroscedastic loss weights sample i by 1/sigma_i^2. Decompose the
variance of log sigma into BETWEEN-BASIN and WITHIN-BASIN(between flow-band)
parts:
  * if the WITHIN-basin (band) part ~ 0, the loss cannot distinguish samples
    inside a basin, so it degenerates to a PER-BASIN reweighting -- and per-basin
    weighting is already CLOSED (oracle +0.0070, every deployable rule negative).
  * only a large WITHIN-basin part would be a genuinely new lever.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger45_weightest2 import build_base, cols_of, frames, w_prod, sc_nse, RECORD

SIG = json.load(open(Path.home() / "riverwatch2/analysis/ledger45_gauge_sigma.json"))
ST = SIG["stations"]
BANDS = ("low", "mid", "high")


def sigma_of(sid, kind="sigma_empirical"):
    s = ST.get(sid, {}).get(kind) or {}
    v = [s.get(b) for b in BANDS]
    return None if any(x is None or not np.isfinite(x) or x <= 0 for x in v) else np.array(v, float)


def basin_ceiling(y, sig3, terciles):
    """Corrected flow-weighted ceiling under multiplicative lognormal error."""
    s = np.where(y <= terciles[0], sig3[0], np.where(y <= terciles[1], sig3[1], sig3[2]))
    e = np.exp(s ** 2)
    num = np.mean(y ** 2 * (e - 1.0))
    den = np.mean(y ** 2 * e) - np.mean(y) ** 2
    return 1.0 - num / den if den > 0 else np.nan


def main():
    BTR, BTE = build_base()
    ship, _ = cols_of(BTR)
    _, sc = frames(BTR, BTE, "h1")
    nse = sc_nse(sc, ship, w_prod(frames(BTR, BTE, "h1")[0], ship))
    out = {"record": RECORD, "sigma_source": "analysis/ledger45_gauge_sigma.json",
           "n_gaugings": SIG["summary"]["n_gaugings_total_all_time"]}

    # ---------------- 1. CEILING, measured vs assumed
    print("=" * 74 + "\n1. CEILING FROM MEASURED sigma (corrected flow-weighted form)\n" + "=" * 74)
    SCEN = {"MEASURED_empirical": None,
            "assumed_central": np.array([0.30, 0.18, 0.35]),
            "assumed_optimistic": np.array([0.25, 0.13, 0.13])}
    res = {}
    for name, fixed in SCEN.items():
        ceils, heads, used = {}, [], 0
        for sid, g in sc.groupby("station_id"):
            y = g["truth"].to_numpy(float); y = y[np.isfinite(y) & (y > 0)]
            if len(y) < 100:
                continue
            s3 = fixed if fixed is not None else sigma_of(sid)
            if s3 is None:
                continue
            t = np.percentile(y, [33.333, 66.667])
            c = basin_ceiling(y, s3, t)
            if np.isfinite(c):
                ceils[sid] = c; used += 1
                if sid in nse.index:
                    heads.append(c - nse[sid])
        cs = pd.Series(ceils)
        r = {"n_basins": used, "median_ceiling": float(cs.median()),
             "p10": float(cs.quantile(.10)), "p90": float(cs.quantile(.90)),
             "median_headroom_vs_record": float(np.median(heads)),
             "frac_basins_above_ceiling": float(np.mean(
                 [nse[s] > ceils[s] for s in ceils if s in nse.index]))}
        res[name] = r
        print(f"  {name:<20} median ceiling {r['median_ceiling']:.4f} "
              f"[p10 {r['p10']:.4f}, p90 {r['p90']:.4f}]  n={used}")
        print(f"  {'':<20} median headroom vs our per-basin NSE "
              f"{r['median_headroom_vs_record']:+.4f}   "
              f"basins already ABOVE ceiling {r['frac_basins_above_ceiling']*100:.1f}%")
    out["ceiling"] = res

    # ---------------- 2. KILL TEST for the heteroscedastic loss
    print("\n" + "=" * 74)
    print("2. KILL TEST — is a label-noise loss anything OTHER than per-basin weighting?")
    print("=" * 74)
    rows = []
    for sid in ST:
        s3 = sigma_of(sid)
        if s3 is not None:
            rows.append((sid, *np.log(s3)))
    L = pd.DataFrame(rows, columns=["sid", "low", "mid", "high"]).set_index("sid")
    bm = L.mean(axis=1)                       # per-basin mean log sigma
    within = L.sub(bm, axis=0)                # band deviation inside each basin
    v_between = float(bm.var(ddof=1))
    v_within = float(within.to_numpy().ravel().var(ddof=1))
    frac_w = v_within / (v_between + v_within)
    print(f"  n basins with measured sigma: {len(L)}")
    print(f"  var(log sigma) BETWEEN basins {v_between:.4f}")
    print(f"  var(log sigma) WITHIN basin (across flow bands) {v_within:.4f}")
    print(f"  => within-basin share {frac_w*100:.1f}%")
    print(f"  band medians of log-sigma deviation: "
          + ", ".join(f"{b} {within[b].median():+.4f}" for b in BANDS))
    sw = float(np.exp(2 * within.to_numpy().std()))
    print(f"  implied within-basin loss-weight spread (1/sigma^2): ~{sw:.2f}x")
    out["killtest"] = {"n_basins": int(len(L)), "var_between": v_between,
                       "var_within": v_within, "within_share": frac_w,
                       "band_median_dev": {b: float(within[b].median()) for b in BANDS},
                       "implied_within_weight_spread": sw}

    p = "benchmarks/ledger45_sigma_ceiling.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p} md5 {hashlib.md5(open(p,'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
