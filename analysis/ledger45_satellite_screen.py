#!/usr/bin/env python3
"""LEDGER 45 TRACK 2 STAGE 2 — the pre-registered break screen for the two
satellite candidates (GlobSnow v3.0 SWE, ESA CCI SM COMBINED v07.1 standard +
break-adjusted), on the full 1980-2008 basin-averaged series built by
ledger45_sat_extract.py (real polygon areal averaging).

Four parts, each can kill the candidate; every number is reported either way.
  A. BREAK TEST at documented sensor transitions: step in the VALUE level
     (relative to a Daymet P/T reference built on the IDENTICAL valid-sample
     mask, so sampling composition cancels) AND step in the VALID-FRACTION
     (calendar-reindexed: a missing file day is vfrac 0 -- the availability
     cadence itself is under test).
  B. SIGN-AWARE LAG SCAN +/-5 d vs discharge (q_mm) and, sharper, the channel
     INCREMENT vs Daymet precip (the +0.229-NSE one-day-offset detector).
  C. THE CONJUNCTION SCREEN (standing pre-GPU filter, pet_screen.py method,
     same anchors): near-median |pcorr(channel, |resid| | precip)| >= 3x noise
     AND R^2 < 0.9 on the existing 15 inputs (+harmonics). An extended-X R^2
     (adds precip + rolling precip/degree-day sums the LSTM state could form)
     is reported as diagnostic only.
  D. MEDIAN-STRADDLE: does the usable cohort straddle the 531-median NSE?

Usage: ledger45_satellite_screen.py [globsnow] [cci_combined] [cci_adjusted]
       (any subset; results MERGE into benchmarks/ledger45_satellite_screen.json)
Run with ~/riverwatch2/gpu1080/.venv/bin/python. Zero GPU.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, str(ROOT))
SAT = ROOT / "data/satellite"
TS = ROOT / "gpu1080/nh_data_multi/time_series"
OUT = ROOT / "benchmarks/ledger45_satellite_screen.json"
NOISE, SM_ANCHOR = 0.0531, 0.2839          # pet_screen / ivt anchors
MEDIAN_BAND = 0.01                          # ivt_landing.BAND

BREAKS = {
    "globsnow": [("1987-09", "SMMR->SSM/I F08"), ("1992-01", "F08->F11"),
                 ("1995-05", "F11->F13")],
    "cci_combined": [("1987-08", "documented"), ("1991-08", "documented"),
                     ("1998-01", "documented(1998)"), ("2002-07", "documented"),
                     ("2007-01", "documented(2007)"),
                     ("2007-10", "diagnostic ASCAT era")],
}
BREAKS["cci_adjusted"] = BREAKS["cci_combined"]


def load_product(prod):
    z = np.load(SAT / f"ledger45_{prod}_daily.npz", allow_pickle=False)
    dates = pd.to_datetime(z["dates"].astype(str), format="%Y%m%d")
    gids = [str(g) for g in z["gids"]]
    val = pd.DataFrame(z["value"], index=dates, columns=gids)
    vf = pd.DataFrame(z["vfrac"], index=dates, columns=gids)
    full = pd.date_range("1980-01-01", "2008-12-31", freq="D")
    val = val.reindex(full)                 # missing file day -> NaN values
    vf = vf.reindex(full, fill_value=0.0)   # missing file day -> vfrac 0
    return val, vf


def load_forcing(gids, cols):
    out = {}
    for g in gids:
        f = TS / f"{g}.nc"
        if not f.exists():
            continue
        d = xr.open_dataset(f)
        df = pd.DataFrame({c: d[c].values.astype(float) for c in cols},
                          index=pd.to_datetime(d.date.values))
        d.close()
        out[g] = df
    return out


def ar1_step_z(series, break_ts, half=24):
    """Step at break_ts on a monthly series: mean(after)-mean(before) over
    +/-half months, z-scored with an AR(1) effective-sample correction."""
    s = series.dropna()
    pre = s[(s.index >= break_ts - pd.DateOffset(months=half)) & (s.index < break_ts)]
    post = s[(s.index >= break_ts) & (s.index < break_ts + pd.DateOffset(months=half))]
    if len(pre) < 12 or len(post) < 12:
        return dict(step=np.nan, z=np.nan, n_pre=len(pre), n_post=len(post))
    step = float(post.mean() - pre.mean())
    resid = np.concatenate([pre.values - pre.mean(), post.values - post.mean()])
    sd = resid.std(ddof=2)
    r1 = float(pd.Series(resid).autocorr(1)) if sd > 0 else 0.0
    r1 = min(max(r1, -0.99), 0.99)
    f = (1 - r1) / (1 + r1)                 # n_eff = n * f
    se = sd * np.sqrt(1 / (len(pre) * f) + 1 / (len(post) * f))
    return dict(step=step, z=float(step / se) if se > 0 else np.nan,
                sd=float(sd), r1=r1, n_pre=len(pre), n_post=len(post))


def monthly_masked_anomaly(val, ref):
    """Monthly cross-basin mean anomaly of `val`, and of each ref channel
    sampled on the IDENTICAL (basin, day) valid mask."""
    mask = np.isfinite(val.values)
    mon = val.index.month
    out = {}
    frames = {"chan": val}
    frames.update(ref)
    for name, df in frames.items():
        a = df.copy()
        arr = np.where(mask, a.values, np.nan)
        a = pd.DataFrame(arr, index=val.index, columns=val.columns)
        clim = a.groupby(mon).transform("mean")        # basin x month climatology
        anom = (a - clim).mean(axis=1, skipna=True)     # daily cross-basin mean
        out[name] = anom.resample("MS").mean()
    return out


def break_tests(prod, val, vf, forc, gids):
    res = []
    ref = {
        "p": pd.DataFrame({g: forc[g]["prcp_daymet"] for g in gids}).reindex(val.index),
        "t": pd.DataFrame({g: (forc[g]["tmax_daymet"] + forc[g]["tmin_daymet"]) / 2
                           for g in gids}).reindex(val.index),
    }
    m = monthly_masked_anomaly(val, ref)
    # residualize channel anomaly on the same-mask reference anomalies
    d = pd.DataFrame(m).dropna()
    X = np.column_stack([np.ones(len(d)), d.p.values, d.t.values])
    beta, *_ = np.linalg.lstsq(X, d.chan.values, rcond=None)
    resid = pd.Series(d.chan.values - X @ beta, index=d.index)
    # valid-fraction: calendar daily mean -> monthly -> deseasonalized
    vfm = vf.mean(axis=1).resample("MS").mean()
    vfm = vfm - vfm.groupby(vfm.index.month).transform("mean")
    for ym, label in BREAKS[prod]:
        bts = pd.Timestamp(ym + "-01")
        rv = ar1_step_z(resid, bts)
        rf = ar1_step_z(vfm, bts)
        res.append({"break": ym, "label": label,
                    "value_step": rv, "vfrac_step": rf})
        print(f"  {prod} {ym} ({label}): value step {rv['step'] if rv['step']==rv['step'] else float('nan'):+.4f} z={rv['z']:+.2f}"
              f" | vfrac step {rf['step']:+.4f} z={rf['z']:+.2f}")
    return res, resid


def lag_scan(prod, val, forc, gids):
    lags = range(-5, 6)
    rq = {l: [] for l in lags}
    rp = {l: [] for l in lags}
    for g in gids:
        x = val[g]
        if x.notna().sum() < 500 or x.std(skipna=True) == 0:
            continue
        fr = forc[g].reindex(val.index)
        q = fr["q_mm"]
        p = fr["prcp_daymet"]
        # increment on consecutive valid days only
        dx = x.diff()
        dx[x.isna() | x.shift(1).isna()] = np.nan
        if prod == "globsnow":  # accumulation signal: cold days only
            cold = ((fr["tmax_daymet"] + fr["tmin_daymet"]) / 2) < 2.0
            dxp = dx.where(cold)
        else:
            dxp = dx
        for l in lags:
            xs = x.shift(l)
            c = xs.corr(q)
            if np.isfinite(c):
                rq[l].append(c)
            c2 = dxp.shift(l).corr(p)
            if np.isfinite(c2):
                rp[l].append(c2)
    med_q = {l: float(np.median(rq[l])) if rq[l] else np.nan for l in lags}
    med_p = {l: float(np.median(rp[l])) if rp[l] else np.nan for l in lags}
    bq = max(med_q, key=lambda l: abs(med_q[l]) if med_q[l] == med_q[l] else -1)
    bp = max(med_p, key=lambda l: med_p[l] if med_p[l] == med_p[l] else -9)
    print(f"  lag scan vs q_mm: " + " ".join(f"{l}:{med_q[l]:+.3f}" for l in lags))
    print(f"  lag scan d(chan) vs precip: " + " ".join(f"{l}:{med_p[l]:+.3f}" for l in lags))
    print(f"  best |r| lag vs q: {bq}; best increment-vs-precip lag: {bp}")
    return {"vs_q_median_r": {str(l): med_q[l] for l in lags},
            "incr_vs_precip_median_r": {str(l): med_p[l] for l in lags},
            "best_lag_vs_q": int(bq), "best_lag_incr_vs_precip": int(bp),
            "n_basins": len(rq[0])}


def build_X(fr, extended=False):
    doy = fr.index.dayofyear.values.astype(float)
    X = pd.DataFrame({
        "tmax": fr["tmax_daymet"], "tmin": fr["tmin_daymet"],
        "srad": fr["srad_daymet"], "vp": fr["vp_daymet"],
        "tmax_n": fr["tmax_nldas"], "srad_n": fr["srad_nldas"],
        "vp_n": fr["vp_nldas"],
        "s1": np.sin(2 * np.pi * doy / 365.25), "c1": np.cos(2 * np.pi * doy / 365.25),
        "s2": np.sin(4 * np.pi * doy / 365.25), "c2": np.cos(4 * np.pi * doy / 365.25),
    }, index=fr.index)
    if extended:
        p = fr["prcp_daymet"]
        tmean = (fr["tmax_daymet"] + fr["tmin_daymet"]) / 2
        X["prcp"] = p
        X["prcp_n"] = fr["prcp_nldas"]
        X["prcp_m"] = fr["prcp_maurer"]
        X["p7"] = p.rolling(7, min_periods=1).sum()
        X["p30"] = p.rolling(30, min_periods=1).sum()
        X["p90"] = p.rolling(90, min_periods=1).sum()
        X["coldp90"] = (p * (tmean < 0)).rolling(90, min_periods=1).sum()
        X["ddm90"] = tmean.clip(lower=0).rolling(90, min_periods=1).sum()
    return X


def r2_of(y, X):
    d = X.copy()
    d["y"] = np.asarray(y, float)   # y must already be row-aligned with X
    d = d.dropna()
    if len(d) < 500 or d.y.std() == 0:
        return np.nan
    A = np.column_stack([np.ones(len(d)), d.drop(columns="y").to_numpy(float)])
    beta, *_ = np.linalg.lstsq(A, d.y.to_numpy(float), rcond=None)
    pred = A @ beta
    ssr = ((d.y.to_numpy(float) - pred) ** 2).sum()
    sst = ((d.y.to_numpy(float) - d.y.mean()) ** 2).sum()
    return float(1 - ssr / sst) if sst > 0 else np.nan


def conjunction_screen(channels, forc):
    """pet_screen.py method verbatim: anchor-checked train-side val resid,
    near-median cohort, pcorr(channel, |resid| | precip)."""
    from noq_harness import build_9stream_train, fit_global_invmse, score, split
    from rescore9 import PREREG
    from ivt_landing import partial_corr

    tr, cols = build_9stream_train()
    fit, val = split(tr)
    w, _, _ = fit_global_invmse(fit, cols)
    base = score(val, cols, w)
    anchor = float(base.median())
    print(f"  ANCHOR {anchor:.6f} (prereg {PREREG})")
    if abs(anchor - PREREG) > 1e-5:
        raise SystemExit("anchor drift — screen numbers would be invalid")
    v = val.copy()
    v["resid"] = (v[cols].to_numpy(float) * w).sum(axis=1) - v.truth
    v["date"] = pd.to_datetime(v.date)
    med = base.median()
    near = set(base[(base - med).abs() <= MEDIAN_BAND].index)

    acc, red, red_ext = {}, {}, {}
    matched = {}
    nb = 0
    for g, ch in channels.items():
        fr = forc[g]
        rows = v[v.station_id == g][["date", "resid"]]
        if len(rows) < 60:
            continue
        ch = ch.copy()
        ch["prcp"] = fr["prcp_daymet"].reindex(ch.index)
        m = rows.merge(ch.reset_index().rename(columns={"index": "date"}),
                       on="date", how="inner")
        nb += 1
        rng = np.random.default_rng(abs(hash(g)) % 2 ** 32)
        m["A_noise"] = rng.standard_normal(len(m))
        X1 = build_X(fr, extended=False)
        X2 = build_X(fr, extended=True)
        for c in [c for c in m.columns if c.startswith(("C_", "A_"))]:
            sub = m[[c, "resid", "prcp"]].dropna()
            if len(sub) < 60 or sub[c].std() == 0:
                continue
            pc = partial_corr(sub[c].values, np.abs(sub.resid.values), sub.prcp.values)
            if np.isfinite(pc):
                acc.setdefault(c, []).append((g, abs(pc)))
            # LEDGER 45 T2e FIX: the null must be drawn on THIS channel's own
            # rows at ITS n. A_noise is never NaN, so the legacy constant floor
            # judged sparse channels against a denser null and flattered them by
            # ~sqrt(n_dense/n_sparse) -- that produced the CCI SM 3.94x "pass",
            # which is 2.33x against a matched null. The bug creates FALSE
            # POSITIVES only; dense channels were effectively matched already.
            if c.startswith("C_"):
                _ar = np.abs(sub.resid.values)
                _p = sub.prcp.values
                _nl = [abs(partial_corr(rng.standard_normal(len(sub)), _ar, _p))
                       for _ in range(25)]
                _nl = [x for x in _nl if np.isfinite(x)]
                if _nl:
                    matched.setdefault(c, []).append((g, float(np.median(_nl))))
            if c.startswith("C_"):
                y = ch[c]
                r2 = r2_of(y.values, X1.reindex(y.index))
                if np.isfinite(r2):
                    red.setdefault(c, []).append(r2)
                r2e = r2_of(y.values, X2.reindex(y.index))
                # y.values row-aligned via reindex(y.index) above
                if np.isfinite(r2e):
                    red_ext.setdefault(c, []).append(r2e)
    print(f"  screened {nb} basins with channel data")
    rows_out = []
    for name, vals in acc.items():
        a = pd.DataFrame(vals, columns=["gid", "p"])
        nv = a[a.gid.isin(near)].p
        m_near = float(nv.median()) if len(nv) >= 10 else np.nan
        m_all = float(a.p.median())
        r2 = float(np.median(red[name])) if name in red else np.nan
        r2e = float(np.median(red_ext[name])) if name in red_ext else np.nan
        mn = pd.DataFrame(matched.get(name, []), columns=["gid", "n"])
        null_near = np.nan
        if len(mn):
            sel = mn[mn.gid.isin(near)].n
            null_near = float(sel.median()) if len(sel) >= 10 else float(mn.n.median())
        x_matched = (m_near / null_near) if (np.isfinite(m_near)
                                             and np.isfinite(null_near)
                                             and null_near > 0) else np.nan
        if name.startswith("A_"):
            verd = "ANCHOR"
        elif not np.isfinite(m_near):
            verd = "near-median cohort too thin (<10 basins) — reported all-basin"
        elif np.isfinite(r2) and r2 >= 0.9:
            verd = "DEAD — reconstructible from existing inputs"
        elif not np.isfinite(x_matched):
            verd = "INVALID — no matched null computed; no verdict"
        elif x_matched >= 3.0:
            verd = "PASS conjunction (matched null) — worth building"
        else:
            verd = "DEAD — %.2fx its own matched null (bar 3x)" % x_matched
        rows_out.append({"channel": name, "near_pcorr": m_near,
                         "all_pcorr": m_all, "x_noise": m_near / NOISE if np.isfinite(m_near) else np.nan,
                         "n_basins": len(a), "n_near": int(len(nv)),
                         "r2_existing": r2, "r2_extended": r2e,
                         "matched_null": null_near, "x_matched_null": x_matched,
                         "verdict": verd})
        print(f"  {name:<14} near|pcorr|={m_near if m_near==m_near else float('nan'):.4f}"
              f" ({(m_near/NOISE) if m_near==m_near else float('nan'):.2f}x noise, n_near={len(nv)})"
              f" all={m_all:.4f} R2={r2:.4f} R2ext={r2e:.4f}  {verd}")
    return rows_out


def straddle(usable_gids):
    nse = json.load(open(ROOT / "analysis/noq_test_result.json"))["per_basin_nse"]
    allv = pd.Series(nse, dtype=float)
    med = float(allv.median())
    u = allv[allv.index.isin(usable_gids)]
    near_all = int(((allv - med).abs() <= MEDIAN_BAND).sum())
    near_u = int(((u - med).abs() <= MEDIAN_BAND).sum())
    out = {"n_usable": int(len(u)), "n_above_median": int((u > med).sum()),
           "n_below_median": int((u < med).sum()),
           "usable_median_nse": float(u.median()) if len(u) else np.nan,
           "median_531": med, "near_band_531": near_all,
           "near_band_usable": near_u}
    print(f"  usable {out['n_usable']}: above/below median {out['n_above_median']}/{out['n_below_median']},"
          f" near-band {near_u}/{near_all}, usable-median NSE {out['usable_median_nse']:.4f} vs {med:.4f}")
    return out


def main():
    prods = sys.argv[1:] or ["globsnow"]
    result = json.load(open(OUT)) if OUT.exists() else {}
    gids_all = sorted(json.load(open(ROOT / "data/camels_gauge_ids.json"))["531"])
    cols = ["prcp_daymet", "prcp_nldas", "prcp_maurer", "tmax_daymet",
            "tmin_daymet", "srad_daymet", "vp_daymet", "tmax_nldas",
            "srad_nldas", "vp_nldas", "q_mm"]
    print("loading forcings ...", flush=True)
    forc = load_forcing(gids_all, cols)

    channels_by_basin = {}
    for prod in prods:
        print(f"\n=== {prod} ===")
        val, vf = load_product(prod)
        r = {}
        # usable cohort
        if prod == "globsnow":
            winter = val[val.index.month.isin([11, 12, 1, 2, 3, 4])]
            wvf = vf[vf.index.month.isin([11, 12, 1, 2, 3, 4])]
            usable = [g for g in gids_all
                      if wvf[g].mean() >= 0.10 and winter[g].std(skipna=True) > 0]
        else:
            usable = [g for g in gids_all if (vf[g] > 0).mean() >= 0.10
                      and val[g].std(skipna=True) > 0]
        r["usable_cohort_n"] = len(usable)
        print(f"  usable cohort: {len(usable)}/531")
        print("  A. break tests")
        r["break_tests"], _ = break_tests(prod, val[usable], vf, forc, usable)
        print("  B. lag scan")
        r["lag_scan"] = lag_scan(prod, val, forc, [g for g in usable])
        print("  D. median straddle")
        r["median_straddle"] = straddle(usable)
        result[prod] = r
        # collect channels for the conjunction screen
        cname = {"globsnow": "C_swe", "cci_combined": "C_sm_comb",
                 "cci_adjusted": "C_sm_adj"}[prod]
        for g in gids_all:
            df = channels_by_basin.setdefault(g, pd.DataFrame(index=val.index))
            df[cname + "_raw"] = val[g]
            if prod == "globsnow":
                fill = val[g].copy()
                # off-season / unretrieved days -> 0 (SWE prior), only for
                # basins the sensor can see at all
                if g in usable:
                    df[cname + "_fill0"] = fill.fillna(0.0)

    print("\n=== C. conjunction screen (standing pre-GPU filter) ===")
    scr = conjunction_screen(channels_by_basin, forc)
    result.setdefault("conjunction_screen", [])
    have = {r["channel"] for r in scr}
    result["conjunction_screen"] = [r for r in result["conjunction_screen"]
                                    if r["channel"] not in have] + scr
    result["anchors"] = {"noise_floor": NOISE, "sm_l3": SM_ANCHOR,
                         "pass_rule": "near|pcorr| >= 3x ITS OWN MATCHED null AND R2 < 0.9 "
                                      "(ledger 45 T2e fix; the legacy constant "
                                      "floor is kept in x_noise for "
                                      "reproducibility only)"}
    OUT.write_text(json.dumps(result, indent=1, default=float))
    md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f"\nwrote {OUT} md5 {md5}")


if __name__ == "__main__":
    main()
