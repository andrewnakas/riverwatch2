#!/usr/bin/env python3
"""REGISTRATION + SKILL SCREEN for the areal GFS lead-1 forcing, before any decay number exists.

A one-day precip label offset was worth +0.229 NSE in this campaign and was invisible to every
mean/bias check, so a new forcing product is not believed until it passes the same two tests the
four corpus products passed:

 1. LAG SCAN vs the corpus's OBSERVED precipitation on the same basins and dates. A 1-day-ahead
    forecast of day D must correlate best with the OBSERVED day D -- peak at k=0. A peak at +/-1
    means the forecast-day bookkeeping is off by a day, which would masquerade as forecast error.
 2. MAGNITUDE. Forecast precipitation should be close to unbiased with a modest wet/dry ratio;
    a 300 mm/day basin maximum is what the earlier double-counting bug produced.

Reference is camels_corpus_gridmetfix3_m3 (the corrected corpus). The GFS file for init date D
carries the forecast for day D itself (lead_day 1), matching load_gfs_windows()'s convention that
a 00z run on D has lead 1 = calendar day D.
"""
import glob, json, os
import numpy as np, pandas as pd

GFS = "data/mblstm/gfs_fcst_areal"
REF = "data/modern_corpus/v3/camels_corpus_gridmetfix3_m3"
VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]
KS = (-2, -1, 0, 1, 2)

files = sorted(glob.glob(f"{GFS}/*.csv.gz"))
print(f"{len(files)} GFS init dates")
assert files, "no GFS files"

# one long frame: (station, date) -> gfs value
rows = []
for f in files:
    d = os.path.basename(f).split(".")[0]
    g = pd.read_csv(f, dtype={"station_id": str})
    g["date"] = d
    rows.append(g)
G = pd.concat(rows, ignore_index=True)
G["station_id"] = G.station_id.str.zfill(8)
print(f"{len(G):,} (basin, date) forecast rows, {G.station_id.nunique()} basins")

cohort = set(str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"])
G = G[G.station_id.isin(cohort)]
print(f"{len(G):,} rows after the cohort filter")

# pull the observed values at the same (basin, date), and at +/- offsets, from the reference corpus
obs = {}
for sid, grp in G.groupby("station_id"):
    p = f"{REF}/{sid}.csv.gz"
    if not os.path.exists(p):
        continue
    c = pd.read_csv(p, dtype={"date": str}).set_index("date")
    obs[sid] = c
print(f"reference corpora loaded for {len(obs)} basins")

print(f"\nLAG SCAN -- corr(GFS forecast for day D, OBSERVED day D+k), pooled over basins\n")
print(f"{'variable':<26}" + "".join(f"{k:>9}" for k in KS) + "   best k")
verdict = {}
for v in VARS:
    r = {}
    for k in KS:
        a, b = [], []
        for sid, grp in G.groupby("station_id"):
            c = obs.get(sid)
            if c is None or v not in c.columns:
                continue
            dts = pd.to_datetime(grp["date"]) + pd.Timedelta(days=int(k))
            key = dts.dt.strftime("%Y-%m-%d")
            ov = key.map(c[v])
            m = np.isfinite(grp[v].to_numpy(float)) & np.isfinite(ov.to_numpy(float))
            if m.sum() < 5:
                continue
            a.append(grp[v].to_numpy(float)[m]); b.append(ov.to_numpy(float)[m])
        if not a:
            r[k] = np.nan; continue
        A, Bv = np.concatenate(a), np.concatenate(b)
        r[k] = float(np.corrcoef(A, Bv)[0, 1])
    fin = {k: x for k, x in r.items() if np.isfinite(x)}
    best = max(fin, key=fin.get) if fin else None
    flag = "" if best == 0 else "   <- OFFSET"
    print(f"  {v:<24}" + "".join(f"{r[k]:9.3f}" for k in KS) + f"{best:>9}{flag}")
    verdict[v] = {"best_k": best, "r": r}

print(f"\nMAGNITUDE at k=0\n")
print(f"{'variable':<26}{'GFS mean':>11}{'obs mean':>11}{'ratio':>9}{'GFS max':>11}{'obs max':>11}")
mag = {}
for v in VARS:
    a, b = [], []
    for sid, grp in G.groupby("station_id"):
        c = obs.get(sid)
        if c is None or v not in c.columns: continue
        ov = grp["date"].map(c[v])
        m = np.isfinite(grp[v].to_numpy(float)) & np.isfinite(ov.to_numpy(float))
        if m.sum(): a.append(grp[v].to_numpy(float)[m]); b.append(ov.to_numpy(float)[m])
    A, Bv = np.concatenate(a), np.concatenate(b)
    mag[v] = {"gfs_mean": float(A.mean()), "obs_mean": float(Bv.mean()),
              "ratio": float(A.mean() / Bv.mean()) if Bv.mean() else np.nan,
              "gfs_max": float(A.max()), "obs_max": float(Bv.max())}
    print(f"  {v:<24}{A.mean():11.3f}{Bv.mean():11.3f}{mag[v]['ratio']:9.3f}"
          f"{A.max():11.2f}{Bv.max():11.2f}")

pk = verdict["precipitation_sum"]["best_k"]
pr = verdict["precipitation_sum"]["r"]
rat = mag["precipitation_sum"]["ratio"]
print("\n" + "=" * 78)
ok = (pk == 0) and (0.7 <= rat <= 1.4) and pr.get(0, 0) >= 0.5
if ok:
    print(f"✅ SCREEN PASSES: precipitation peaks at k=0 (r={pr[0]:.3f}), bias ratio {rat:.3f}.")
    print("   The forecast-day bookkeeping is right and the magnitudes are sane, so a day-1 decay")
    print("   delta computed from this build is interpretable.")
else:
    print(f"⛔ SCREEN FAILS: precip best k={pk} (r at 0 = {pr.get(0)}), ratio {rat:.3f}.")
    print("   Do NOT report any decay number from this build until this is resolved.")
json.dump({"n_dates": len(files), "lag": {k: {"best_k": v["best_k"]} for k, v in verdict.items()},
           "magnitude": mag, "screen_passes": bool(ok)},
          open("benchmarks/m1_gfs_areal_screen.json", "w"), indent=2)
print("-> benchmarks/m1_gfs_areal_screen.json")
