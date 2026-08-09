#!/usr/bin/env python3
"""Phase C: the Li/Song 0.83 grand ensemble = plain arithmetic mean of 7
seed-averaged streamflow streams, day-1 (h=1) median NSE over the 531 basins,
test window 1995-2010.

7 streams:
  3 LSTM   : camels531ls_{daymet,nldas,maurer}_nhlstm_s{111,222,333}  (mean of 3 seeds)
  3 dHBV   : dhbv dumps  {daymet,nldas,maurer}  s{111,222,333}         (mean of 3 seeds)
  1 LSTMmulti: camels531ls_multi_nhlstm_s{111,222,3334}                (mean of 3 seeds)

Merge key: (station_id, target_date) at h==1. The LSTM dumps label t0 = the
predicted (target) date. The dHBV dumps (our --dump-day1) label t0 one day
EARLIER (corpus date indexing offset, verified empirically: same truth values
one row apart). So dHBV t0 is shifted +1 day before joining.

We inner-join on the keys present in ALL streams so the ensemble mean is over a
common (basin,date) sample. Report:
  - each stream's own day-1 median NSE (sanity vs the paper's rungs)
  - the 7-stream plain-mean ensemble day-1 median NSE vs paper 0.8294
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

def load_seed_avg(paths, shift_days=0, point_col="ymed"):
    """Mean of the given seed dumps' point col, keyed on (station_id, date@h=1)."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, compression="gzip", usecols=lambda c: c in
                         ("station_id","t0","h","truth",point_col))
        df = df[df["h"] == 1].copy()
        df["station_id"] = df["station_id"].astype(str).str.zfill(8)
        d = pd.to_datetime(df["t0"])
        if shift_days:
            d = d + pd.Timedelta(days=shift_days)
        df["date"] = d.dt.strftime("%Y-%m-%d")
        frames.append(df[["station_id","date","truth",point_col]]
                      .rename(columns={point_col:"pred"}))
    # seed-average the pred per (station,date); truth is identical across seeds
    cat = pd.concat(frames, ignore_index=True)
    agg = cat.groupby(["station_id","date"], as_index=False).agg(
        pred=("pred","mean"), truth=("truth","first"))
    return agg

def median_nse(df, pred_col="pred"):
    nses = []
    for sid, g in df.groupby("station_id"):
        yt = g["truth"].to_numpy(float); yh = g[pred_col].to_numpy(float)
        ok = np.isfinite(yt) & np.isfinite(yh)
        yt, yh = yt[ok], yh[ok]
        if len(yt) < 20 or np.var(yt) < 1e-9:
            continue
        nses.append(1.0 - np.mean((yt-yh)**2)/np.var(yt))
    return float(np.median(nses)), len(nses)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm-dir", default="gpu1080/dumps")
    ap.add_argument("--dhbv-dir", default="dhbv_dumps")
    ap.add_argument("--dhbv-tag", default="", help="filename infix for dhbv dumps e.g. _s14_")
    ap.add_argument("--subset531", default="data/camels_gauge_ids.json")
    args = ap.parse_args()

    L = Path(args.lstm_dir); H = Path(args.dhbv_dir)
    # 3 LSTM single-forcing streams (3 seeds each)
    streams = {}
    for f in ("daymet","nldas","maurer"):
        seeds = [L/f"camels531ls_{f}_nhlstm_s{s}.csv.gz" for s in ("111","222","333")]
        # nldas: prefer the retrained s111 if present
        if f == "nldas" and (L/"camels531ls_nldas_nhlstm_s111_NEW.csv.gz").exists():
            seeds[0] = L/"camels531ls_nldas_nhlstm_s111_NEW.csv.gz"
        seeds = [p for p in seeds if p.exists()]
        streams[f"lstm_{f}"] = load_seed_avg(seeds)
    # LSTMmulti (3 seeds: 111,222,3334)
    mseeds = [L/f"camels531ls_multi_nhlstm_s{s}.csv.gz" for s in ("111","222","3334")]
    mseeds = [p for p in mseeds if p.exists()]
    streams["lstm_multi"] = load_seed_avg(mseeds)
    # 3 dHBV streams (shift t0 +1 day to align with LSTM t0 labeling)
    for f in ("daymet","nldas","maurer"):
        seeds = sorted(H.glob(f"camels531ls_{f}_dhbv_s[0-9]*.csv.gz"))
        seeds = [p for p in seeds if "_test_" not in p.name]
        if not seeds:
            print(f"WARN dhbv_{f}: no clean dumps found in {H}", file=sys.stderr)
        streams[f"dhbv_{f}"] = load_seed_avg(seeds, shift_days=1)
        print(f"dhbv_{f}: seeds={[p.name for p in seeds]}", file=sys.stderr)

    # per-stream sanity NSE
    print("=== per-stream day-1 median NSE ===")
    for name, df in streams.items():
        m, n = median_nse(df)
        print(f"  {name:14s} medNSE={m:.4f}  (basins={n}, rows={len(df)})")

    # 531 subset
    import json
    ids531 = set(str(x).zfill(8) for x in json.load(open(args.subset531)).get("531", []))
    print(f"\n531-subset ids loaded: {len(ids531)}")

    # inner-join all 7 streams on (station,date)
    merged = None
    for name, df in streams.items():
        d = df[["station_id","date","pred"]].rename(columns={"pred":name})
        if merged is None:
            t = df[["station_id","date","truth"]]
            merged = t.merge(d, on=["station_id","date"], how="inner")
        else:
            merged = merged.merge(d, on=["station_id","date"], how="inner")
    if ids531:
        merged = merged[merged["station_id"].isin(ids531)]
    stream_cols = list(streams.keys())
    # --- rw2: optional inverse-MSE stream weighting ---------------------------------
    # Equal weighting costs -0.0041 vs inverse-MSE weights (validated on a
    # three-way fit/select/score split; see dhbv-downweight-deployable-gain).
    # It is a poor rule here because member quality is HETEROGENEOUS: dHBV is
    # both the most decorrelated family and the weakest (own NSE 0.896-0.916 vs
    # LSTM 0.918-0.939), so at equal weight it drags more than it diversifies.
    # OFF by default so the committed 0.8298 reproduction is unchanged.
    import os as _os
    _mode = _os.environ.get("RW2_STREAM_WEIGHTS", "equal")
    if _mode == "invmse":
        # ⚠️ phase_c loads TEST dumps only, so weights MUST come from the
        # matching _TRAIN_ dumps. Fitting on the scored frame would be leakage;
        # the first version of this patch tried that and was correctly refused.
        import numpy as _np
        _tr_streams = {}
        for _name in stream_cols:
            _f = _name.replace("lstm_", "").replace("dhbv_", "")
            _isd = _name.startswith("dhbv")
            _dir = H if _isd else L
            _pat = (f"camels531ls_{_f}_dhbv_TRAIN_s*.csv.gz" if _isd
                    else f"camels531ls_{_f}_nhlstm_TRAIN_s*.csv.gz")
            _ps = sorted(p for p in _dir.glob(_pat))
            if _ps:
                _tr_streams[_name] = load_seed_avg(_ps, shift_days=1 if _isd else 0)
        _missing = [c for c in stream_cols if c not in _tr_streams]
        if _missing:
            raise SystemExit(f"invmse: no TRAIN dumps for {_missing}; cannot fit "
                             f"weights without leaking test data")
        _m = None
        for _n, _d in _tr_streams.items():
            _x = _d[["station_id", "date", "pred"]].rename(columns={"pred": _n})
            _m = (_d[["station_id", "date", "truth"]].merge(_x, on=["station_id", "date"])
                  if _m is None else _m.merge(_x, on=["station_id", "date"]))
        _mse = _np.array([_np.nanmean((_m[c].to_numpy(float)
                                       - _m["truth"].to_numpy(float)) ** 2)
                          for c in stream_cols])
        _theta, _lam = 4.0, 0.25   # chosen by the 3-way fit/select/score split
        _raw = _mse ** (-_theta); _raw = _raw / _raw.sum()
        _eq = _np.ones(len(stream_cols)) / len(stream_cols)
        _w = _lam * _eq + (1 - _lam) * _raw
        _w = _w / _w.sum()
        if _w.std() < 1e-6:
            raise SystemExit("invmse: weights collapsed to uniform — check scale")
        print(f"\n--- inverse-MSE stream weights (fit on {len(_m):,} TRAIN rows) ---")
        for _c, _x in zip(stream_cols, _w):
            print(f"    {_c:<16} {_x:.4f}")
        merged["ens"] = (merged[stream_cols].to_numpy(float) * _w).sum(axis=1)
    else:
        merged["ens"] = merged[stream_cols].mean(axis=1)
    # --- end rw2: optional inverse-MSE stream weighting -----------------------------
    print(f"\ncommon (basin,date) rows after 7-way inner join + 531: {len(merged)}"
          f"  basins={merged['station_id'].nunique()}")

    m, n = median_nse(merged, "ens")
    print(f"\n=== HEADLINE: 7-stream plain-mean ensemble day-1 median NSE = {m:.4f} "
          f"(basins={n}) vs paper 0.8294 ===")
    # also each stream on the COMMON sample
    print("\n--- each stream on the common sample ---")
    for c in stream_cols:
        mm, nn = median_nse(merged.rename(columns={c:"pred"}), "pred")
        print(f"  {c:14s} {mm:.4f}")

if __name__ == "__main__":
    main()
