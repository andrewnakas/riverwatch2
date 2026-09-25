#!/usr/bin/env python3
"""Ledger 45 track 2 stage 2, SUPPLEMENT: TRAIN-vs-TEST level shift and gradual
drift in the satellite channels.

The break tests in ledger45_satellite_screen.py look for STEPS at documented
sensor dates. They cannot see a gradual instrument drift -- and for this
project's split (train 1980-10..1995-09, test 1995-10..2008-12) a drift is
arguably worse than a step: the network would be trained on one channel level
and scored on another, which is exactly how a channel comes to encode the DATE.

For each product this reports, on the P/T-residualized monthly anomaly of the
basin-mean series (same construction as the break test):
  - OLS trend per decade + its AR(1)-corrected t
  - raw mean in the train window vs the test window, and the shift in SD units
  - the SAME two quantities for a Daymet reference (cold-season precip and
    tmean), which is homogeneous by construction -- if the reference shows no
    drift and the satellite channel does, the drift is instrumental.
Zero GPU. Merges into benchmarks/ledger45_satellite_screen.json key drift.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(os.path.expanduser("~/riverwatch2")); sys.path.insert(0, str(ROOT))
SAT = ROOT / "data/satellite"; TS = ROOT / "gpu1080/nh_data_multi/time_series"
OUT = ROOT / "benchmarks/ledger45_satellite_screen.json"
TRAIN = ("1980-10-01", "1995-09-30"); TEST = ("1995-10-01", "2008-12-31")

def ar1_t(y, x):
    b = np.polyfit(x, y, 1); r = y - np.polyval(b, x)
    r1 = float(pd.Series(r).autocorr(1)); r1 = min(max(r1, -0.99), 0.99)
    neff = len(y) * (1 - r1) / (1 + r1)
    se = np.sqrt(((r ** 2).sum() / max(neff - 2, 1)) / ((x - x.mean()) ** 2).sum() * len(y) / max(neff, 1))
    return float(b[0]), float(b[0] / se) if se > 0 else np.nan, float(r1), float(neff)

def stats(series, label):
    s = series.dropna()
    xi = (s.index - s.index[0]).days.values / 3652.5   # decades
    slope, t, r1, neff = ar1_t(s.values, xi)
    tr = s[(s.index >= TRAIN[0]) & (s.index <= TRAIN[1])]
    te = s[(s.index >= TEST[0]) & (s.index <= TEST[1])]
    sd = s.std()
    return {"label": label, "trend_per_decade": slope, "t_ar1": t, "ar1": r1,
            "n_eff": neff, "train_mean": float(tr.mean()), "test_mean": float(te.mean()),
            "shift": float(te.mean() - tr.mean()),
            "shift_in_sd": float((te.mean() - tr.mean()) / sd) if sd > 0 else np.nan,
            "sd": float(sd), "n_months": len(s)}

def main():
    prods = sys.argv[1:] or ["globsnow"]
    gids_all = sorted(json.load(open(ROOT / "data/camels_gauge_ids.json"))["531"])
    forc = {}
    for g in gids_all:
        f = TS / f"{g}.nc"
        if not f.exists(): continue
        d = xr.open_dataset(f)
        forc[g] = pd.DataFrame({"p": d["prcp_daymet"].values.astype(float),
                                "t": ((d["tmax_daymet"].values + d["tmin_daymet"].values) / 2).astype(float)},
                               index=pd.to_datetime(d.date.values)); d.close()
    res = json.load(open(OUT)) if OUT.exists() else {}
    res.setdefault("drift", {})
    for prod in prods:
        z = np.load(SAT / f"ledger45_{prod}_daily.npz")
        dates = pd.to_datetime(z["dates"].astype(str), format="%Y%m%d")
        gids = [str(g) for g in z["gids"]]
        val = pd.DataFrame(z["value"], index=dates, columns=gids).reindex(
            pd.date_range("1980-01-01", "2008-12-31", freq="D"))
        P = pd.DataFrame({g: forc[g]["p"] for g in gids if g in forc}).reindex(val.index)
        T = pd.DataFrame({g: forc[g]["t"] for g in gids if g in forc}).reindex(val.index)
        val = val[P.columns]
        mask = np.isfinite(val.values); mon = val.index.month
        def anom(df):
            a = pd.DataFrame(np.where(mask, df.values, np.nan), index=val.index, columns=df.columns)
            return (a - a.groupby(mon).transform("mean")).mean(axis=1, skipna=True).resample("MS").mean()
        ac, ap, at = anom(val), anom(P), anom(T)
        d = pd.DataFrame({"c": ac, "p": ap, "t": at}).dropna()
        X = np.column_stack([np.ones(len(d)), d.p.values, d.t.values])
        beta, *_ = np.linalg.lstsq(X, d.c.values, rcond=None)
        resid = pd.Series(d.c.values - X @ beta, index=d.index)
        out = {"channel_residualized": stats(resid, "channel resid on P,T"),
               "channel_raw_anomaly": stats(ac, "channel anomaly"),
               "ref_precip_anomaly": stats(ap, "daymet precip (same mask)"),
               "ref_tmean_anomaly": stats(at, "daymet tmean (same mask)")}
        print(f"\n=== {prod} ===")
        print(f"{'series':<28}{'trend/dec':>11}{'t_AR1':>8}{'train':>9}{'test':>9}{'shift':>9}{'shift/SD':>10}")
        for k, s in out.items():
            print(f"{s['label']:<28}{s['trend_per_decade']:>11.4f}{s['t_ar1']:>8.2f}"
                  f"{s['train_mean']:>9.3f}{s['test_mean']:>9.3f}{s['shift']:>9.3f}{s['shift_in_sd']:>10.2f}")
        # raw (non-anomaly) basin-mean level, train vs test
        raw = val.mean(axis=1)
        out["raw_level"] = {"train_mean": float(raw[(raw.index >= TRAIN[0]) & (raw.index <= TRAIN[1])].mean()),
                            "test_mean": float(raw[(raw.index >= TEST[0]) & (raw.index <= TEST[1])].mean())}
        print(f"raw basin-mean level: train {out['raw_level']['train_mean']:.4f} "
              f"test {out['raw_level']['test_mean']:.4f} "
              f"({100*(out['raw_level']['test_mean']/out['raw_level']['train_mean']-1):+.1f}%)")
        # cadence control: the file cadence changes across the record (GlobSnow
        # roughly doubles at the 1987/88 SMMR->SSM/I transition), so restrict to
        # a fixed set of peak months -- month composition can no longer confound.
        peak = [12, 1, 2, 3] if prod == "globsnow" else [5, 6, 7, 8, 9]
        cs = raw[raw.index.month.isin(peak)]
        ctr = float(cs[(cs.index >= TRAIN[0]) & (cs.index <= TRAIN[1])].mean())
        cte = float(cs[(cs.index >= TEST[0]) & (cs.index <= TEST[1])].mean())
        out["fixed_month_level"] = {"months": peak, "train_mean": ctr, "test_mean": cte,
                                    "pct_change": 100 * (cte / ctr - 1) if ctr else np.nan}
        print(f"fixed-month {peak} level: train {ctr:.4f} test {cte:.4f} "
              f"({100*(cte/ctr-1):+.1f}%)  [cadence-composition control]")
        res["drift"][prod] = out
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("\nmd5", hashlib.md5(OUT.read_bytes()).hexdigest())

main()
