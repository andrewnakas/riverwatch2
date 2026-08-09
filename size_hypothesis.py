"""Does reproduction quality depend on basin SIZE? Test on several published basins.

Two-point evidence suggested it: 126 km2 reproduced at corr 0.990, 986 km2 at
0.753. Our target is 1163 km2. But n=2 is not a hypothesis, it is a coincidence
waiting to be checked, and the decision on whether to reconstruct 13235000 rests
on it.

A plausible mechanism exists. AORC is ~1 km; HydroShare aggregates over the
catchment. For a 126 km2 basin our polygon and theirs enclose nearly the same
handful of cells, so any boundary discrepancy is small. For a 1000 km2 basin the
boundary encloses hundreds of cells and a modest difference in ridgeline
placement swaps many wet/dry cells -- exactly the orographic sensitivity
measured earlier.

If quality degrades with area, 13235000 at 1163 km2 would reconstruct poorly and
should stay excluded. If it does not, the two-point pattern was noise and the
reconstruction is worth doing.

Runs on 6 published basins spanning ~50-2000 km2, 720 h, GAGES-II polygons.
"""
import io, sys, tarfile
import numpy as np, pandas as pd, geopandas as gpd, json
sys.path.insert(0, ".")
from aorc_exact_extract import get_polygon, extract_year

TAR = "data/aorc_raw/water_year_1990.tar.gz"
att = json.load(open("data/camels_attrs.json"))

# published basins spanning a wide area range, all in the AORC product
with tarfile.open(TAR, "r:gz") as tf:
    avail = {m.name.split("/")[-1][:8]: m for m in tf.getmembers()
             if m.name.endswith(".csv")}
    cands = []
    for b in sorted(avail):
        a = att.get(b, {})
        ar = a.get("area_gages2")
        if ar and a.get("slope_mean", 0) > 20:
            cands.append((ar, b))
    cands.sort()
    picks = [cands[0][1], cands[len(cands)//5][1], cands[2*len(cands)//5][1],
             cands[3*len(cands)//5][1], cands[4*len(cands)//5][1], cands[-1][1]]
    pub = {}
    for b in picks:
        d = pd.read_csv(io.BytesIO(tf.extractfile(avail[b]).read()))
        d["time"] = pd.to_datetime(d["time"])
        pub[b] = d

print(f"{'basin':>9} {'area_km2':>9} {'ratio':>7} {'corr':>7}")
res = []
for b in picks:
    try:
        gdf = get_polygon(b)
        mine = extract_year(b, gdf, 1990, tmax=720)
    except Exception as e:
        print(f"{b:>9}  FAILED {str(e)[:50]}"); continue
    j = mine.merge(pub[b], on="time", suffixes=("_m", "_p"))
    a = j["APCP_surface_m"].to_numpy(float); c = j["APCP_surface_p"].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(c)
    r = np.corrcoef(a[ok], c[ok])[0, 1]
    ratio = np.nansum(a)/np.nansum(c) if np.nansum(c) else np.nan
    ar = att[b].get("area_gages2", np.nan)
    print(f"{b:>9} {ar:>9.0f} {ratio:>7.3f} {r:>7.4f}", flush=True)
    res.append((ar, ratio, r))

if len(res) >= 4:
    a = np.array(res, float)
    print(f"\ncorr(area, reproduction_corr) = {np.corrcoef(a[:,0], a[:,2])[0,1]:+.3f}")
    print(f"corr(area, ratio)            = {np.corrcoef(a[:,0], a[:,1])[0,1]:+.3f}")
    print()
    print("Strongly NEGATIVE => quality degrades with size; 13235000 (1163 km2)")
    print("would reconstruct poorly and should stay excluded.")
    print("Near zero => the two-point pattern was noise; reconstruct it.")
