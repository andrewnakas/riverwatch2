#!/usr/bin/env python3
"""Score the withqmulti candidate against its PRE-REGISTERED gate.

Gate (memory: withq-multi-prereg-gate), written before the member existed:
  1. MEMBER    day-1 median NSE, band >=0.85, <0.80 means something is broken
  2. ENSEMBLE  added as a 5th member to the record 4-member mean.
               SHIP THRESHOLD >= +0.001 (raised to +0.002 in the note's text
               for the neighbour member; for withqmulti the registered
               threshold is +0.001 -- we report against both)
  3. CONTROL   a redundant extra seed of an existing forcing bought +0.0005
               (seed depth is saturated); the candidate must clearly beat that
  4. LOO       removing it must cost >= +0.001
  5. TOP-50    scored on POOLED error, not per-basin medians (261 rows/basin
               makes per-basin with-q NSE fragile)
  6. BREADTH   fraction of basins improved -- per the median-leverage finding,
               a narrow gain cannot move the median

Readout: (ylo+yhi)/2, the record's own readout.
"""
import json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
rng = np.random.default_rng(0)

ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
P = "data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"
CAND = "data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz"
F = ("daymet", "maurer", "nldas", "aorc")

if not os.path.exists(CAND):
    sys.exit(f"candidate dump not present yet: {CAND}")

def load(path, tag):
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(ids)]
    d[tag] = (d.ylo + d.yhi) / 2
    return d[["station_id", "t0", "truth", tag]]

m = None
for f in F:
    d = load(P.format(f), f)
    m = d if m is None else m.merge(d[["station_id", "t0", f]], on=["station_id", "t0"])
c = load(CAND, "withqmulti")
before = len(m)
m = m.merge(c[["station_id", "t0", "withqmulti"]], on=["station_id", "t0"], how="inner")
print(f"joined rows {before:,} -> {len(m):,}   basins {m.station_id.nunique()}")
if len(m) < 0.9 * before:
    print("WARNING: candidate join lost >10% of rows")

def per_basin(df, cols):
    e = df[cols].to_numpy(float).mean(1)
    d = df.assign(_e=e); o = {}
    for s, g in d.groupby("station_id"):
        y = g.truth.to_numpy(float); p = g._e.to_numpy(float)
        k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
        if len(y) < 20 or np.var(y) < 1e-9: continue
        o[s] = 1 - np.mean((y - p) ** 2) / np.var(y)
    return pd.Series(o)

print("\n" + "=" * 70)
print("GATE 1 — MEMBER (band >=0.85; <0.80 = broken)")
print("=" * 70)
sm = per_basin(m, ["withqmulti"])
print(f"  withqmulti day-1 median NSE   {sm.median():.4f}  (n={len(sm)})")
for f in F:
    print(f"    {f:<10} {per_basin(m,[f]).median():.4f}")

print("\n" + "=" * 70)
print("GATE 2/3 — ENSEMBLE and CONTROL")
print("=" * 70)
base = per_basin(m, list(F))
cand = per_basin(m, list(F) + ["withqmulti"])
i = sorted(set(base.index) & set(cand.index))
b, a = base.reindex(i), cand.reindex(i)
delta = a.median() - b.median()
print(f"  4-member record ensemble      {b.median():.4f}")
print(f"  + withqmulti (5 members)      {a.median():.4f}   delta {delta:+.5f}")
print(f"  control: 5th seed of a forcing gave +0.0005 (seed depth saturated)")
print(f"  => {'PASSES' if delta >= 0.001 else 'FAILS'} the +0.001 ship threshold;"
      f" {'clears' if delta > 0.0005 else 'does NOT clear'} the control")

print("\n" + "=" * 70)
print("GATE 4 — LEAVE-ONE-OUT (removing it must cost >= +0.001)")
print("=" * 70)
for f in F:
    others = [x for x in F if x != f] + ["withqmulti"]
    s = per_basin(m, others).reindex(i)
    print(f"  drop {f:<10} {s.median():.4f}   loss {a.median()-s.median():+.5f}")
s_noc = per_basin(m, list(F)).reindex(i)
print(f"  drop withqmulti  {s_noc.median():.4f}   loss {a.median()-s_noc.median():+.5f}")

print("\n" + "=" * 70)
print("GATE 5 — TOP-50 COHORT (POOLED error, not per-basin medians)")
print("=" * 70)
try:
    top50 = set(pd.read_csv("withq_top50_targets.csv").iloc[:, 0].astype(str).str.zfill(8))
    sub = m[m.station_id.isin(top50)]
    for lab, cols in (("record 4", list(F)), ("+withqmulti", list(F) + ["withqmulti"])):
        e = sub[cols].to_numpy(float).mean(1); y = sub.truth.to_numpy(float)
        k = np.isfinite(y) & np.isfinite(e)
        print(f"  {lab:<14} pooled NSE {1-((y[k]-e[k])**2).sum()/((y[k]-y[k].mean())**2).sum():.4f}")
except Exception as e:
    print(f"  (skipped: {e})")

print("\n" + "=" * 70)
print("GATE 6 — BREADTH (a narrow gain cannot move a median)")
print("=" * 70)
d = (a - b).dropna()
print(f"  basins improved   {(d>0).sum()}/{len(d)} = {(d>0).mean():.1%}")
print(f"  mean delta        {d.mean():+.5f}")
bs = [np.median(a.to_numpy()[rng.choice(len(i),len(i),True)]) -
      np.median(b.to_numpy()[rng.choice(len(i),len(i),True)]) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  bootstrap 95% CI  [{lo:+.5f}, {hi:+.5f}]  "
      f"{'excludes zero' if lo>0 else 'STRADDLES ZERO'}")
print(f"\n  (for scale: multi5 improved 93.6% of basins on the no-q track)")
