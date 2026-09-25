#!/usr/bin/env python3
"""LEDGER 45 / T2e — IS THE CONJUNCTION SCREEN COMPARING UNMATCHED SAMPLE SIZES?

T2d found CCI SM reads 3.94x against the screen's CONSTANT noise floor (0.0531)
but only 2.27x against an EMPIRICAL noise anchor computed on the SAME rows.

Hypothesis: in ledger45_satellite_screen.conjunction_screen, A_noise is never
NaN, so its dropna keeps ALL residual rows (~370/basin), while a sparse channel
like CCI SM keeps only the days it is observed (~76/basin). |pcorr| under the
null scales ~1/sqrt(n), so a SPARSE channel is compared against a DENSER null and
is flattered by ~sqrt(370/76) ~ 2.2x.

This measures the row counts directly. No hypothesis is adopted without them.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

H = Path.home() / "riverwatch2"
sys.path.insert(0, str(H)); sys.path.insert(0, str(H / "analysis"))
import ledger45_satellite_screen as S                       # noqa: E402
from noq_harness import build_9stream_train, fit_global_invmse, score, split  # noqa: E402
from rescore9 import PREREG                                 # noqa: E402
from ivt_landing import partial_corr                        # noqa: E402


def main():
    gids = sorted(json.load(open(H / "data/camels_gauge_ids.json"))["531"])
    forc = S.load_forcing(gids, ["prcp_daymet"])
    prods = sys.argv[1:] or ["cci_combined", "globsnow"]
    PROD = {p: S.load_product(p)[0] for p in prods}

    tr, cols = build_9stream_train()
    fit, v0 = split(tr)
    w, _, _ = fit_global_invmse(fit, cols)
    base = score(v0, cols, w)
    print(f"ANCHOR {float(base.median()):.6f} (prereg {PREREG})")
    v = v0.copy()
    v["resid"] = (v[cols].to_numpy(float) * w).sum(axis=1) - v.truth
    v["date"] = pd.to_datetime(v.date)

    rec = {"n_rows_used": {}, "matched_null": {}}
    counts = {"A_noise": []}
    counts.update({p: [] for p in prods})
    ratio = {p: [] for p in prods}
    rng0 = np.random.default_rng(20260829)
    for g in gids:
        if g not in forc:
            continue
        r = v[v.station_id == g][["date", "resid"]]
        if len(r) < 60:
            continue
        idx = PROD[prods[0]].index
        ch = pd.DataFrame(index=idx)
        for p in prods:
            ch[p] = PROD[p][g].reindex(idx)
        ch["prcp"] = forc[g]["prcp_daymet"].reindex(idx)
        m = r.merge(ch.reset_index().rename(columns={"index": "date"}),
                    on="date", how="inner")
        m["A_noise"] = rng0.standard_normal(len(m))
        for c in ["A_noise"] + prods:
            sub = m[[c, "resid", "prcp"]].dropna()
            if len(sub) >= 60:
                counts[c].append(len(sub))
        # matched-n null for each sparse channel, on ITS OWN rows
        for c in prods:
            sub = m[[c, "resid", "prcp"]].dropna()
            if len(sub) < 60:
                continue
            ar = np.abs(sub.resid.values)
            pc = partial_corr(sub[c].values, ar, sub.prcp.values)
            nulls = [abs(partial_corr(rng0.standard_normal(len(sub)), ar,
                                      sub.prcp.values)) for _ in range(25)]
            nl = float(np.median(nulls))
            if np.isfinite(pc) and nl > 0:
                ratio[c].append(abs(pc) / nl)

    print("\n--- rows actually used per basin (median) ---")
    for c, arr in counts.items():
        if arr:
            print(f"  {c:<10} n_basins={len(arr):>4}  median rows={int(np.median(arr)):>5}")
            rec["n_rows_used"][c] = {"n_basins": len(arr),
                                     "median_rows": int(np.median(arr))}
    if counts["A_noise"] and counts[prods[0]]:
        infl = float(np.sqrt(np.median(counts["A_noise"]) / np.median(counts[prods[0]])))
        rec["predicted_inflation_sqrt_n_ratio"] = infl
        print(f"\n  => predicted flattery of the sparse channel: "
              f"sqrt({int(np.median(counts['A_noise']))}/"
              f"{int(np.median(counts[prods[0]]))}) = {infl:.2f}x")

    print("\n--- MATCHED-n ratio (channel |pcorr| / its own same-n null) ---")
    for c, arr in ratio.items():
        if arr:
            a = np.array(arr)
            rec["matched_null"][c] = {"median_ratio": float(np.median(a)),
                                      "n_basins": int(len(a))}
            print(f"  {c:<10} median ratio {np.median(a):.2f}x  (n={len(a)} basins)"
                  f"   {'PASS' if np.median(a) >= 3.0 else 'FAIL'} vs 3x bar")

    p = H / ("benchmarks/ledger45_screen_nmatch_" + "_".join(prods) + ".json")
    p.write_text(json.dumps(rec, indent=1, default=float))
    print(f"\nwrote {p} md5 {hashlib.md5(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
