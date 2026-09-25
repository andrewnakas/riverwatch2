#!/usr/bin/env python3
"""Ledger 45 track 2 stage 2, SUPPLEMENT: the snow-conditioned specification of
the conjunction screen for GlobSnow SWE.

The pooled screen in ledger45_satellite_screen.py averages over all 531 basins
and all seasons; a SWE channel can only carry information in snowy basins during
the cold season, so the pooled number dilutes it. This runs the identical screen
(same anchors, same partial_corr, same near-median band) on three progressively
tighter specifications, and then asks the median-straddle question of whichever
cohort survives -- because a channel that only helps snowy basins must still
straddle the 531-median to move the metric.

Specs:
  S1 all basins / all days        (= the pooled number, reproduced as a check)
  S2 snowy basins (frac_snow>=0.3), all days
  S3 snowy basins, cold season (Nov-Apr) only
Channels: C_swe (raw), C_dswe (day-to-day SWE increment == melt/accum rate).
Zero GPU. Writes into benchmarks/ledger45_satellite_screen.json key snow_conditioned.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(os.path.expanduser("~/riverwatch2")); sys.path.insert(0, str(ROOT))
SAT = ROOT / "data/satellite"; TS = ROOT / "gpu1080/nh_data_multi/time_series"
OUT = ROOT / "benchmarks/ledger45_satellite_screen.json"
NOISE, BAND = 0.0531, 0.01
from noq_harness import build_9stream_train, fit_global_invmse, score, split
from rescore9 import PREREG
from ivt_landing import partial_corr

def main():
    z = np.load(SAT / "ledger45_globsnow_daily.npz")
    dates = pd.to_datetime(z["dates"].astype(str), format="%Y%m%d")
    gids = [str(g) for g in z["gids"]]
    val = pd.DataFrame(z["value"], index=dates, columns=gids)
    attrs = json.load(open(ROOT / "data/camels_attrs.json"))
    fsnow = {g: attrs.get(g, {}).get("frac_snow", np.nan) for g in gids}

    tr, cols = build_9stream_train(); fit, va = split(tr)
    w, _, _ = fit_global_invmse(fit, cols); base = score(va, cols, w)
    anchor = float(base.median()); print(f"ANCHOR {anchor:.6f}")
    assert abs(anchor - PREREG) < 1e-5, "anchor drift"
    v = va.copy(); v["resid"] = (v[cols].to_numpy(float) * w).sum(axis=1) - v.truth
    v["date"] = pd.to_datetime(v.date)
    med = base.median(); near = set(base[(base - med).abs() <= BAND].index)

    specs = {"S1_all": lambda g: True,
             "S2_snowy": lambda g: fsnow.get(g, 0) >= 0.3,
             "S3_snowy_cold": lambda g: fsnow.get(g, 0) >= 0.3}
    acc = {s: {} for s in specs}
    for g in gids:
        f = TS / f"{g}.nc"
        if not f.exists(): continue
        d = xr.open_dataset(f)
        fr = pd.DataFrame({c: d[c].values.astype(float) for c in
                           ["prcp_daymet", "tmax_daymet", "tmin_daymet"]},
                          index=pd.to_datetime(d.date.values)); d.close()
        x = val[g].reindex(fr.index)
        dx = x.diff(); dx[x.isna() | x.shift(1).isna()] = np.nan
        ch = pd.DataFrame({"C_swe": x, "C_dswe": dx, "prcp": fr.prcp_daymet},
                          index=fr.index)
        ch["cold"] = ((fr.tmax_daymet + fr.tmin_daymet) / 2) < 5.0
        rows = v[v.station_id == g][["date", "resid"]]
        if len(rows) < 60: continue
        m = rows.merge(ch.reset_index().rename(columns={"index": "date"}),
                       on="date", how="inner")
        for sname, keep in specs.items():
            if not keep(g): continue
            mm = m[m.cold] if sname == "S3_snowy_cold" else m
            for c in ("C_swe", "C_dswe"):
                sub = mm[[c, "resid", "prcp"]].dropna()
                if len(sub) < 60 or sub[c].std() == 0: continue
                pc = partial_corr(sub[c].values, np.abs(sub.resid.values), sub.prcp.values)
                if np.isfinite(pc): acc[sname].setdefault(c, []).append((g, abs(pc)))
    out = {}
    print(f"\n{'spec':<16}{'chan':<9}{'n':>5}{'near|pcorr|':>13}{'xnoise':>8}{'n_near':>8}  verdict")
    for sname in specs:
        for c, vals in sorted(acc[sname].items()):
            a = pd.DataFrame(vals, columns=["gid", "p"])
            nv = a[a.gid.isin(near)].p
            mn = float(nv.median()) if len(nv) >= 10 else float("nan")
            xn = mn / NOISE
            verd = ("PASS >=3x noise" if xn >= 3 else
                    "marginal 2-3x" if xn >= 2 else "DEAD below 2x noise")
            out[f"{sname}|{c}"] = {"n_basins": len(a), "near_pcorr": mn,
                                   "x_noise": xn, "n_near": int(len(nv)),
                                   "all_pcorr": float(a.p.median()), "verdict": verd}
            print(f"{sname:<16}{c:<9}{len(a):>5}{mn:>13.4f}{xn:>8.2f}{len(nv):>8}  {verd}")
    # straddle of the snowy cohort
    nse = pd.Series(json.load(open(ROOT / "analysis/noq_test_result.json"))["per_basin_nse"], dtype=float)
    snowy = [g for g in gids if fsnow.get(g, 0) >= 0.3 and val[g].notna().sum() > 500]
    u = nse[nse.index.isin(snowy)]
    st = {"n_snowy_usable": len(u), "above": int((u > med).sum()), "below": int((u < med).sum()),
          "snowy_median_nse": float(u.median()), "median_531": float(nse.median()),
          "near_band_snowy": int(((u - float(nse.median())).abs() <= BAND).sum()),
          "near_band_531": int(((nse - nse.median()).abs() <= BAND).sum())}
    print(f"\nsnowy cohort n={st['n_snowy_usable']} above/below 531-median {st['above']}/{st['below']}"
          f" near-band {st['near_band_snowy']}/{st['near_band_531']}"
          f" snowy-median NSE {st['snowy_median_nse']:.4f} vs {st['median_531']:.4f}")
    res = json.load(open(OUT)) if OUT.exists() else {}
    res["snow_conditioned"] = {"screen": out, "snowy_straddle": st,
                               "cold_def": "tmean<5C", "snowy_def": "frac_snow>=0.3"}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("md5", hashlib.md5(OUT.read_bytes()).hexdigest())

main()
