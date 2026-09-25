#!/usr/bin/env python3
"""LEDGER 45 / T2c — DOES THE CCI SM INFORMATION LAND WHERE THE ERROR IS?

T2b: CCI SM passes at 3.94x noise and is NOT an availability/date artifact.
But deseasonalised it falls to 2.59x, so the signal is seasonal/slow -- while the
campaign's residual is EVENT-MAGNITUDE SCATTER. "Novel" is not "useful".

So partition the SAME partial correlation by day type and see where the channel
is actually informative:
  EVENT days   = top decile of basin precip (p7) OR top decile of |resid|
  ORDINARY days= the rest
A channel that only informs ordinary days cannot move an event-scatter residual.

Same anchor, same near-median cohort, same partial_corr(channel, |resid| | precip)
as the standing screen.
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

NOISE = S.NOISE
BAND = S.MEDIAN_BAND


def main():
    gids = sorted(json.load(open(H / "data/camels_gauge_ids.json"))["531"])
    forc = S.load_forcing(gids, ["prcp_daymet"])
    val, vf = S.load_product("cci_combined")

    tr, cols = build_9stream_train()
    fit, v0 = split(tr)
    w, _, _ = fit_global_invmse(fit, cols)
    base = score(v0, cols, w)
    anchor = float(base.median())
    print(f"ANCHOR {anchor:.6f} (prereg {PREREG})")
    assert abs(anchor - float(PREREG)) < 1e-5, "anchor mismatch — stop"
    v = v0.copy()
    v["resid"] = (v[cols].to_numpy(float) * w).sum(axis=1) - v.truth
    v["date"] = pd.to_datetime(v.date)
    med = base.median()
    near = set(base[(base - med).abs() <= BAND].index)

    acc = {k: [] for k in ("event_p", "ordinary_p", "event_r", "ordinary_r", "all")}
    COV = []
    POOL = []
    for g in gids:
        if g not in forc or val[g].std(skipna=True) == 0:
            continue
        rows = v[v.station_id == g][["date", "resid"]]
        if len(rows) < 100:
            continue
        fr = forc[g]
        ch = pd.DataFrame({"C": val[g]})
        ch["prcp"] = fr["prcp_daymet"].reindex(ch.index)
        ch["p7"] = ch["prcp"].rolling(7, min_periods=1).sum()
        m = rows.merge(ch.reset_index().rename(columns={"index": "date"}),
                       on="date", how="inner").dropna(subset=["C", "resid", "prcp"])
        if len(m) < 60:
            continue
        ar = np.abs(m.resid.values)
        z = lambda a: (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)
        POOL.append(pd.DataFrame({"gid": g, "C": z(m.C.values), "ar": z(ar),
                                  "prcp": m.prcp.values, "p7": m.p7.values,
                                  "arr": ar}))
        masks = {
            "event_p": m.p7.values >= np.nanquantile(m.p7.values, 0.90),
            "ordinary_p": m.p7.values < np.nanquantile(m.p7.values, 0.90),
            "event_r": ar >= np.nanquantile(ar, 0.90),
            "ordinary_r": ar < np.nanquantile(ar, 0.90),
            "all": np.ones(len(m), bool)}
        COV.append(len(m))
        for k, mk in masks.items():
            if mk.sum() < 25:
                continue
            pc = partial_corr(m.C.values[mk], ar[mk], m.prcp.values[mk])
            if np.isfinite(pc):
                acc[k].append((g, abs(pc)))

    # ---- POOLED test: per-basin standardised rows pooled, then split by day type.
    # Per-basin deciles are too thin (CCI early-era coverage), so pool.
    out = {"anchor": anchor, "noise": NOISE, "rows": {}}
    P = pd.concat(POOL, ignore_index=True)
    Pn = P[P.gid.isin(near)]
    print(f"\nPOOLED near-median rows: {len(Pn):,} over {Pn.gid.nunique()} basins")
    out["pooled"] = {}
    for label, frame in (("near_median", Pn), ("all_basins", P)):
        qp = np.nanquantile(frame.p7.values, 0.90)
        qr = np.nanquantile(frame.arr.values, 0.90)
        sets = {"event_precip": frame.p7.values >= qp,
                "ordinary_precip": frame.p7.values < qp,
                "event_resid": frame.arr.values >= qr,
                "ordinary_resid": frame.arr.values < qr,
                "all": np.ones(len(frame), bool)}
        print(f"  [{label}]")
        out["pooled"][label] = {}
        for k, mk in sets.items():
            if mk.sum() < 200:
                continue
            pc = partial_corr(frame.C.values[mk], frame.ar.values[mk],
                              frame.prcp.values[mk])
            out["pooled"][label][k] = {"pcorr": float(pc), "n": int(mk.sum())}
            print(f"    {k:<18} |pcorr| {abs(pc):.4f}   n={mk.sum():,}")

    print(f"\nmedian usable rows/basin: {int(np.median(COV)) if COV else 0}")
    print(f"\n{'subset':<12} {'near|pcorr|':>11} {'x noise':>8} {'n_near':>7} {'all':>9}")
    for k, vals in acc.items():
        a = pd.DataFrame(vals, columns=["gid", "p"])
        nv = a[a.gid.isin(near)].p
        mn = float(nv.median()) if len(nv) >= 10 else float("nan")
        out["rows"][k] = {"near_pcorr": mn, "x_noise": mn / NOISE,
                          "n_near": int(len(nv)), "all_pcorr": float(a.p.median()),
                          "n_basins": int(len(a))}
        print(f"{k:<12} {mn:>11.4f} {mn/NOISE:>8.2f} {len(nv):>7} {a.p.median():>9.4f}")

    pn = out["pooled"]["near_median"]
    e = abs(pn.get("event_resid", {}).get("pcorr", float("nan")))
    o = abs(pn.get("ordinary_resid", {}).get("pcorr", float("nan")))
    if not (np.isfinite(e) and np.isfinite(o)):
        out["verdict"] = "INVALID — non-finite statistics, no verdict"
    else:
        out["verdict"] = ("informative ON EVENT DAYS — aimed at the residual"
                          if e >= o else
                          "NOT event-informative — the signal is on ordinary days")
    print(f"\n=> POOLED near-median: event |pcorr| {e:.4f} vs ordinary {o:.4f}")
    print(f"=> {out['verdict']}")
    p = H / "benchmarks/ledger45_cci_eventday.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {p} md5 {hashlib.md5(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
