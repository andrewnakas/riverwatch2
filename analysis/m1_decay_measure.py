#!/usr/bin/env python3
"""Measure the DAY-1 forecast-forcing decay: perfect forcing vs real 1-day-ahead GFS.

Registered decision (c). The headline is day 1, so the operative question is narrow: what does
day-1 skill cost when the TARGET-DAY forcing is a 1-day-ahead forecast instead of an analysis?

Method. Two dumps of the SAME frozen checkpoints:
  perfect : the existing test1 dump (observed forcing)
  gfs     : a dump with --gfs-finetune, which substitutes the archived forecast into the DECODER
Both are inner-joined on (station_id, t0) so the comparison is paired on identical rows, and only
rows the forecast archive covers are used.

⚠️ Frame caveat, stated because this campaign has been burned by it. The archive begins 2021-05 and
is sampled every 14 days, so these rows are a SPARSE SUBSET of the test window. A stride-14 frame
once flattered an absolute median NSE here by +0.036. Only the paired DELTA is reportable from this
build; the absolute numbers below are NOT comparable to the headline and are printed for context.

⚠️ This reads TEST-window rows. It selects nothing -- both arms are the same frozen checkpoints --
and it is logged in benchmarks/m1_test_reads.log as a characterization, like the decade probe.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

AP = argparse.ArgumentParser()
AP.add_argument("--member", required=True)
AP.add_argument("--seeds", default="501,502,503,504,505")
AP.add_argument("--perfect-suffix", default="test1")
AP.add_argument("--gfs-suffix", default="test1gfs")
AP.add_argument("--out", default="benchmarks/m1_decay_day1.json")
a = AP.parse_args()
seeds = [s for s in a.seeds.split(",") if s]
DD = "data/mblstm/l51_dumps"


def load(suffix):
    acc = []
    used = []
    for s in seeds:
        f = f"{DD}/m1_{a.member}_s{s}_{suffix}.csv.gz"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, dtype={"station_id": str})
        if "h" in d.columns:
            d = d[d["h"] == 1]
        d["station_id"] = d.station_id.str.zfill(8)
        d["pred"] = 0.5 * (d["ylo"] + d["yhi"])
        acc.append(d[["station_id", "t0", "pred", "truth"]])
        used.append(s)
    if not acc:
        sys.exit(f"no dumps for {a.member} suffix {suffix}")
    m = pd.concat(acc).groupby(["station_id", "t0"], as_index=False).agg(
        pred=("pred", "mean"), truth=("truth", "mean"))
    return m, used


P, sp = load(a.perfect_suffix)
G, sg = load(a.gfs_suffix)
assert sp == sg, f"seed sets differ: perfect {sp} vs gfs {sg} -- seed-match both sides"
J = P.merge(G, on=["station_id", "t0"], suffixes=("_p", "_g"))
bad = ~np.isclose(J["truth_p"], J["truth_g"], rtol=1e-5, atol=1e-3)
assert bad.mean() < 1e-4, f"truth disagrees on {bad.mean():.2%} of rows"
print(f"member {a.member}  seeds {sp}")
print(f"paired rows {len(J):,}  basins {J.station_id.nunique()}")
print(f"date span   {J.t0.min()} .. {J.t0.max()}")


def per_basin_nse(df, col):
    out = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth_p"].to_numpy(float); p = g[col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        if ok.sum() < 20:
            continue
        y, p = y[ok], p[ok]
        den = np.sum((y - y.mean()) ** 2)
        if den <= 0:
            continue
        out[sid] = 1.0 - np.sum((y - p) ** 2) / den
    return out


np_, ng = per_basin_nse(J, "pred_p"), per_basin_nse(J, "pred_g")
common = sorted(set(np_) & set(ng))
assert common, "no basin has >=20 scoreable rows on the forecast-covered frame"
vp = np.array([np_[b] for b in common]); vg = np.array([ng[b] for b in common])
print(f"basins with >=20 rows: {len(common)}")
print(f"\n  perfect forcing  median NSE  {np.median(vp):.6f}   (SPARSE FRAME - not the headline)")
print(f"  GFS day-1 forcing median NSE  {np.median(vg):.6f}")
d = vg - vp
rng = np.random.default_rng(0)
bs = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(5000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\n  PAIRED delta (gfs - perfect)  {np.median(d):+.6f}")
print(f"  95% CI                        [{lo:+.6f}, {hi:+.6f}]")
print(f"  breadth (basins where gfs is better) {float(np.mean(d > 0)):.3f}")
print()
if hi < 0:
    print(f"⇒ Real 1-day-ahead forcing costs {abs(np.median(d)):.6f} of median NSE at lead 1,")
    print("  CI excluding zero. This is the honest operational discount on the perfect-forcing")
    print("  headline, measured rather than assumed.")
elif lo > 0:
    print("⚠️ Forecast forcing SCORED BETTER, which is not physically sensible at lead 1 -- suspect")
    print("   the forcing files or the window filter before believing it.")
else:
    print("⇒ The cost is not distinguishable from zero on this frame (CI includes zero).")
json.dump({"member": a.member, "seeds": sp, "rows": int(len(J)),
           "basins_scored": len(common),
           "span": [str(J.t0.min()), str(J.t0.max())],
           "median_nse_perfect_SPARSE_FRAME": float(np.median(vp)),
           "median_nse_gfs_day1": float(np.median(vg)),
           "paired_delta": float(np.median(d)), "ci95": [float(lo), float(hi)],
           "breadth": float(np.mean(d > 0)),
           "frame_caveat": "sparse: archive starts 2021-05, 14-day stride; only the DELTA is "
                           "reportable, absolute NSE is not comparable to the headline"},
          open(a.out, "w"), indent=2)
print(f"-> {a.out}")
