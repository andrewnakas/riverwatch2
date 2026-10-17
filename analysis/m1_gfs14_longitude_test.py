"""Does the tmax lag deficit grow WESTWARD?

If the GFS lead day (a UTC 24-hour window) is offset from the corpus's observed
daily tmax (defined on a local-time day), the offset in hours is set by
longitude: a basin at -120 is 8 h behind UTC, one at -70 is 5 h. The prediction
is then DIRECTIONAL and falsifiable: the k=-1 preference should be stronger the
further west the basin.

If there is no relation, the day-definition hypothesis is wrong and the cause is
something else. Either way this is derived, not asserted -- 20+ of my causal
claims in this project have been overturned.
"""
import json
import numpy as np
import pandas as pd

LEAD = "12"          # a lead where the deficit is large and resolvable
VAR = "temperature_2m_max"

d = json.load(open("benchmarks/m1_gfs_areal_screen14.json"))
rec = d["by_lead"][LEAD][VAR]
res = rec.get("resolution") or {}
print(f"lead {LEAD} {VAR}: best_k={rec['best_k']}, pooled r@0={rec['r']['0']:.4f}, "
      f"r@-1={rec['r']['-1']:.4f}")
print(f"  paired per-basin median delta {res.get('delta')}, CI {res.get('ci95')}, "
      f"n={res.get('n_basins')}")

# Per-basin r is not stored in the artifact, so recompute it for this one
# (lead, variable) directly -- cheap, one lead, one channel.
import glob
fs = sorted(glob.glob("data/mblstm/gfs_fcst_areal14/*.csv.gz"))
rows = []
for p in fs:
    init = p.split("/")[-1].split(".")[0]
    g = pd.read_csv(p, dtype={"station_id": str})
    g = g[g.lead_day == int(LEAD)]
    g["target"] = pd.to_datetime(init) + pd.to_timedelta(int(LEAD) - 1, unit="D")
    rows.append(g[["station_id", "target", VAR]])
G = pd.concat(rows, ignore_index=True)
G["station_id"] = G.station_id.str.zfill(8)

corp = "data/modern_corpus/camels_corpus_nldas_modern_areal"
# lat/lon live in stations_v15.json; camels_attrs.json carries only the 27
# Addor attributes and has no coordinates at all (the first version of this
# test read `gauge_lon` from it, got 0 basins, and said nan).
_st = json.load(open("data/stations_v15.json"))
_st = _st if isinstance(_st, list) else _st.get("stations", [])
attrs = {str(r.get("id") or r.get("site_no")).zfill(8): r for r in _st}
out = []
for sid, grp in G.groupby("station_id"):
    try:
        o = pd.read_csv(f"{corp}/{sid}.csv.gz", dtype={"station_id": str})
    except Exception:
        continue
    dc = next(c for c in o.columns if c.lower() in ("date", "datetime", "time"))
    if VAR not in o.columns:
        continue
    o[dc] = pd.to_datetime(o[dc])
    ser = o.set_index(dc)[VAR]
    ser = ser[~ser.index.duplicated(keep="last")]
    gv = grp[VAR].to_numpy(float)
    r = {}
    for k in (-1, 0):
        ov = ser.reindex(grp["target"] + pd.to_timedelta(k, unit="D")).to_numpy(float)
        m = np.isfinite(ov) & np.isfinite(gv)
        if m.sum() >= 30 and ov[m].std() > 0 and gv[m].std() > 0:
            r[k] = float(np.corrcoef(gv[m], ov[m])[0, 1])
    if len(r) != 2:
        continue
    a = attrs.get(sid) or {}
    lon = a.get("lon")
    lat = a.get("lat")
    if lon is None:
        continue
    out.append((sid, float(lon), float(lat) if lat else np.nan, r[0], r[-1], r[0] - r[-1]))

df = pd.DataFrame(out, columns=["sid", "lon", "lat", "r0", "rm1", "delta"])
print(f"\nbasins with both correlations and a longitude: {len(df)}")
print(f"delta = r@0 - r@-1 : median {df.delta.median():+.5f}  "
      f"({(df.delta < 0).mean():.1%} of basins prefer k=-1)")

from scipy import stats as st
rho, p = st.spearmanr(df.lon, df.delta)
print(f"\nSpearman(longitude, delta) = {rho:+.4f}  p = {p:.3g}")
print("  PREDICTION of the day-definition hypothesis: further west (more")
print("  negative lon) => stronger k=-1 preference => MORE NEGATIVE delta,")
print("  i.e. a POSITIVE rho between lon and delta.")
assert len(df) > 100, f"only {len(df)} basins matched a longitude"
q = pd.qcut(df.lon, 4, labels=["west", "w-mid", "e-mid", "east"])
print("\nby longitude quartile:")
for name, g in df.groupby(q, observed=True):
    print(f"  {str(name):<6} lon {g.lon.min():7.2f}..{g.lon.max():7.2f}  "
          f"n={len(g):3d}  median delta {g.delta.median():+.5f}  "
          f"prefer k=-1: {(g.delta < 0).mean():.1%}")
rho2, p2 = st.spearmanr(df.lat, df.delta, nan_policy="omit")
print(f"\ncontrol -- Spearman(latitude, delta) = {rho2:+.4f}  p = {p2:.3g}")
