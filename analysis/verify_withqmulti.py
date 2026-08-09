#!/usr/bin/env python3
"""withqmulti PASSED (+0.00498). Verify before believing it.

V1. Is the candidate accidentally NESTED with the record members -- i.e. is the
    fused corpus just daymet/maurer/nldas, so we are re-averaging the same
    information? (It IS the same 3 forcings! The record members are daymet,
    maurer, nldas, aorc. So withqmulti shares 3 of 4 sources.)
    -> If it were pure redundancy it would NOT add +0.005. But check the
       residual correlation to see what it actually is.
V2. Truth identity: does the candidate's truth column match the record dumps
    bit-for-bit? (a misalignment would fake a gain)
V3. Is the gain an artifact of "5 members beat 4"? Control: add a DUPLICATE of
    an existing member as a 5th and see what that alone buys.
V4. The CI straddles zero -- how worried should we be? Report the sign test and
    a paired bootstrap (the right test), not the difference-of-medians bootstrap.
"""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
rng = np.random.default_rng(0)
ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
P = "data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"
CAND = "data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz"
F = ("daymet", "maurer", "nldas", "aorc")

def load(path, tag):
    d = pd.read_csv(path, usecols=["station_id","t0","h","truth","ylo","yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(ids)]
    d[tag] = (d.ylo + d.yhi) / 2
    return d[["station_id","t0","truth",tag]]

m = None
for f in F:
    d = load(P.format(f), f)
    m = d if m is None else m.merge(d[["station_id","t0",f]], on=["station_id","t0"])
c = load(CAND, "withqmulti")
print("V2. TRUTH IDENTITY")
j = m[["station_id","t0","truth"]].merge(
        c[["station_id","t0","truth"]].rename(columns={"truth":"t2"}),
        on=["station_id","t0"], how="inner")
print(f"  rows compared {len(j):,}   max|diff| = {(j.truth-j.t2).abs().max():.10f}")
assert (j.truth-j.t2).abs().max() < 1e-6, "TRUTH MISMATCH"
m = m.merge(c[["station_id","t0","withqmulti"]], on=["station_id","t0"])

print("\nV1. RESIDUAL CORRELATION vs the existing members")
res = {k: (m[k]-m.truth).to_numpy(float) for k in list(F)+["withqmulti"]}
print("        " + "".join(f"{k:>12}" for k in list(F)+["withqmulti"]))
for a in list(F)+["withqmulti"]:
    row = "".join(f"{np.corrcoef(res[a],res[b])[0,1]:>12.4f}" for b in list(F)+["withqmulti"])
    print(f"  {a:<6}{row}")
cm = np.mean([np.corrcoef(res['withqmulti'],res[b])[0,1] for b in F])
print(f"\n  mean corr(withqmulti, existing members) = {cm:.4f}")
print(f"  mean corr among the 4 existing members   = "
      f"{np.mean([np.corrcoef(res[a],res[b])[0,1] for a in F for b in F if a<b]):.4f}")

def pb(df, cols):
    e = df[cols].to_numpy(float).mean(1)
    d = df.assign(_e=e); o = {}
    for s,g in d.groupby("station_id"):
        y=g.truth.to_numpy(float); p=g._e.to_numpy(float)
        k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
        if len(y)<20 or np.var(y)<1e-9: continue
        o[s]=1-np.mean((y-p)**2)/np.var(y)
    return pd.Series(o)

print("\nV3. IS IT JUST '5 MEMBERS BEAT 4'? (duplicate-member control)")
base = pb(m, list(F))
for dup in F:
    m["_dup"] = m[dup]
    s = pb(m, list(F)+["_dup"])
    i = sorted(set(s.index)&set(base.index))
    print(f"  duplicate {dup:<8} {s.reindex(i).median():.4f}   "
          f"delta {s.reindex(i).median()-base.reindex(i).median():+.5f}")
cand = pb(m, list(F)+["withqmulti"])
i = sorted(set(cand.index)&set(base.index))
print(f"  withqmulti (REAL)   {cand.reindex(i).median():.4f}   "
      f"delta {cand.reindex(i).median()-base.reindex(i).median():+.5f}")

print("\nV4. PAIRED significance (the correct test)")
d = (cand.reindex(i)-base.reindex(i)).dropna()
print(f"  basins improved {(d>0).sum()}/{len(d)} = {(d>0).mean():.1%}")
from math import sqrt
bs = [np.median(d.to_numpy()[rng.choice(len(d),len(d),True)]) for _ in range(5000)]
lo,hi = np.percentile(bs,[2.5,97.5])
print(f"  PAIRED bootstrap of the per-basin delta: median {d.median():+.5f}"
      f"  95% CI [{lo:+.5f}, {hi:+.5f}]  {'EXCLUDES ZERO' if lo>0 else 'straddles'}")
