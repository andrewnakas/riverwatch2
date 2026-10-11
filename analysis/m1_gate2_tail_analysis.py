"""GATE 2 follow-up: what predicts the per-basin damage from point-filling the
newest 4 encoder days? Two candidate mechanisms, both measurable with zero GPU:
  (A) basin properties  -> the basins are intrinsically sensitive
  (B) how much point and areal forcing actually DIFFER at that basin
      -> a better fill source would fix it
"""
import importlib.util, json, numpy as np, pandas as pd
from pathlib import Path
spec = importlib.util.spec_from_file_location("l51", "analysis/l51_withq_score.py")
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)

S = [501, 502, 503, 504, 505]
c = l51.load_member("nldasm9","val1",seeds=S,protocol="modern-v3-9yr",prefix="g1nl_").rename(columns={"nldasm9":"areal"})
t = l51.load_member("nldasm9","val1",seeds=S,protocol="modern-v3-9yr",prefix="g2sp4_").rename(columns={"nldasm9":"spliced"})
m = c.merge(t, on=["station_id","t0"], how="inner")
m = m.drop(columns=["truth_y"]).rename(columns={"truth_x":"truth"})
a = l51.per_basin(m, ["areal"]); b = l51.per_basin(m, ["spliced"])
k = sorted(set(a.index) & set(b.index))
d = (b.loc[k] - a.loc[k])
df = pd.DataFrame({"delta": d, "nse_areal": a.loc[k]})

# ---- (B) forcing disagreement per basin, over the val window -----------------
NL = Path("data/modern_corpus/v3/camels_corpus_nldas_g1sub")
OM = Path("data/modern_corpus/v3/camels_corpus_openmeteo_m3_cohort")
V = ["temperature_2m_mean","temperature_2m_max","temperature_2m_min",
     "precipitation_sum","shortwave_radiation_sum","vapor_pressure"]
rows = {}
for sid in df.index:
    fn, fo = NL / f"{sid}.csv.gz", OM / f"{sid}.csv.gz"
    if not (fn.exists() and fo.exists()):
        continue
    n = pd.read_csv(fn, parse_dates=["date"]).set_index("date")
    o = pd.read_csv(fo, parse_dates=["date"]).set_index("date")
    j = n.join(o, how="inner", lsuffix="_nl", rsuffix="_om")
    j = j.loc["2014-10-01":"2018-09-30"]
    if not len(j):
        continue
    r = {}
    for v in V:
        x, y = j[f"{v}_nl"].to_numpy(float), j[f"{v}_om"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30: continue
        sd = x[ok].std() or 1.0
        r[f"nrmse_{v}"] = float(np.sqrt(((x[ok]-y[ok])**2).mean()) / sd)
    # precip is the one that matters most for runoff; also its bias
    x, y = j["precipitation_sum_nl"].to_numpy(float), j["precipitation_sum_om"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    r["precip_bias_frac"] = float((y[ok].sum() - x[ok].sum()) / max(x[ok].sum(), 1e-9))
    r["precip_r"] = float(np.corrcoef(x[ok], y[ok])[0,1]) if ok.sum() > 30 else np.nan
    rows[sid] = r
fd = pd.DataFrame(rows).T
df = df.join(fd)

# ---- (A) basin statics -------------------------------------------------------
at = json.load(open("data/camels_attrs.json"))
keep = ["area_gages2","aridity","p_mean","frac_snow","elev_mean","slope_mean",
        "high_prec_freq","frac_forest","soil_depth_statsgo","p_seasonality"]
sd = pd.DataFrame({s: {k2: at.get(s, {}).get(k2, np.nan) for k2 in keep} for s in df.index}).T
df = df.join(sd)

tail = df[df.delta < -0.05]
rest = df[df.delta >= -0.05]
print(f"basins {len(df)}   tail (delta < -0.05) {len(tail)}   rest {len(rest)}")
print("\n=== Spearman(delta, predictor) over all basins — most negative = most damaging ===")
from scipy.stats import spearmanr
res = []
for col in df.columns:
    if col == "delta": continue
    v = df[col].to_numpy(float); dd = df.delta.to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(dd)
    if ok.sum() < 50: continue
    rho, p = spearmanr(v[ok], dd[ok])
    res.append((rho, p, col, int(ok.sum())))
for rho, p, col, n in sorted(res):
    print(f"  {col:28s} rho {rho:+.4f}  p {p:.2e}  n={n}")
print("\n=== tail vs rest medians ===")
for col in ["precip_r","nrmse_precipitation_sum","precip_bias_frac","nrmse_temperature_2m_mean",
            "area_gages2","aridity","p_mean","frac_snow","nse_areal"]:
    if col in df:
        print(f"  {col:28s} tail {tail[col].median():+.4f}   rest {rest[col].median():+.4f}")
print("\n=== the tail basins ===")
cols = [c for c in ["delta","nse_areal","precip_r","nrmse_precipitation_sum","area_gages2","aridity"] if c in df]
print(tail.sort_values("delta")[cols].to_string(float_format=lambda x: f"{x:+.4f}"))
df.to_csv("/tmp/g2_tail_table.csv")
print("\n-> /tmp/g2_tail_table.csv")
