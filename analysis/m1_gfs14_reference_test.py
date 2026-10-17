"""Which REFERENCE is the 14-lead areal GFS misaligned against?

The screen reports a resolvable 1-day preference for temperature at leads >=7.
It validates against `camels_corpus_gridmetfix3_m3` -- a gridMET corpus. But
the member is trained and served on NLDAS-2 AREAL forcing, and its decoder will
be fed areal GFS alongside an NLDAS-2 areal encoder. The alignment that matters
operationally is therefore GFS vs NLDAS-2, not GFS vs gridMET.

If GFS is aligned against NLDAS-2 and misaligned against gridMET, the offset
belongs to gridMET and the 14-lead build is fine for its actual use. If it is
misaligned against both, the build is suspect. Measured per basin and paired,
because that is the comparison the data supports.
"""
import glob, json
import numpy as np
import pandas as pd

REFS = {
    "gridmet (what the screen uses)": "data/modern_corpus/v3/camels_corpus_gridmetfix3_m3",
    "nldas2 areal (what is SERVED)": "data/modern_corpus/camels_corpus_nldas_modern_areal",
}
LEADS = [1, 7, 12, 14]
VARS = ["temperature_2m_max", "temperature_2m_mean", "precipitation_sum"]

fs = sorted(glob.glob("data/mblstm/gfs_fcst_areal14/*.csv.gz"))
print(f"init dates: {len(fs)}")
frames = []
for p in fs:
    init = p.split("/")[-1].split(".")[0]
    g = pd.read_csv(p, dtype={"station_id": str})
    g = g[g.lead_day.isin(LEADS)]
    g["init"] = pd.to_datetime(init)
    frames.append(g)
G = pd.concat(frames, ignore_index=True)
G["station_id"] = G.station_id.str.zfill(8)
G["target"] = G["init"] + pd.to_timedelta(G["lead_day"] - 1, unit="D")
print(f"GFS rows: {len(G):,}  basins: {G.station_id.nunique()}")

def per_basin_delta(ref_dir, lead, var):
    """median over basins of (r@0 - r@-1), plus breadth and a bootstrap CI."""
    gl = G[G.lead_day == lead]
    out = []
    for sid, grp in gl.groupby("station_id"):
        try:
            o = pd.read_csv(f"{ref_dir}/{sid}.csv.gz", usecols=["date", var])
        except Exception:
            continue
        o["date"] = pd.to_datetime(o["date"])
        ser = o.set_index("date")[var]
        ser = ser[~ser.index.duplicated(keep="last")]
        gv = grp[var].to_numpy(float)
        rs = {}
        for k in (-1, 0):
            ov = ser.reindex(grp["target"] + pd.to_timedelta(k, unit="D")).to_numpy(float)
            m = np.isfinite(ov) & np.isfinite(gv)
            if m.sum() >= 30 and ov[m].std() > 0 and gv[m].std() > 0:
                rs[k] = float(np.corrcoef(gv[m], ov[m])[0, 1])
        if len(rs) == 2:
            out.append(rs[0] - rs[-1])
    d = np.array(out, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return None
    rng = np.random.default_rng(0)
    draws = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)])
    return {"n": len(d), "median": float(np.median(d)),
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "frac_prefers_k0": float((d > 0).mean())}

print()
print("delta = r@0 - r@-1, per basin, median over basins.")
print("POSITIVE = correctly aligned at k=0.  NEGATIVE = prefers k=-1.")
print()
hdr = "%-20s %-5s %-34s %5s %10s %22s %8s"
print(hdr % ("variable", "lead", "reference", "n", "median", "CI95", "k=0 win"))
res = {}
for var in VARS:
    for lead in LEADS:
        for name, rd in REFS.items():
            r = per_basin_delta(rd, lead, var)
            if r is None:
                print(hdr % (var, lead, name, "-", "insufficient", "-", "-"))
                continue
            res[(var, lead, name)] = r
            print("%-20s %-5d %-34s %5d %+10.5f  [%+8.5f,%+8.5f] %7.1f%%"
                  % (var, lead, name, r["n"], r["median"],
                     r["ci"][0], r["ci"][1], 100 * r["frac_prefers_k0"]))
        print()
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()},
          open("/tmp/g14_ref_test.json", "w"), indent=1)
print("-> /tmp/g14_ref_test.json")
