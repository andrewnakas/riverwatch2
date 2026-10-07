#!/usr/bin/env python3
"""Does the misregistered gridMET member's advantage live where FUTURE rain is?

A score comparison alone cannot separate the two explanations for the original gridMET member
beating a correctly-registered one:
  (1) COST  -- the correction is right and something else about it hurts (e.g. target-day rain is
               worth more in the encoder's final step than in the decoder's first).
  (2) LEAK  -- the original is exploiting rain it is not entitled to.

This separates them without training anything, from the dumps on disk. For target day T the model
is entitled to forcing through T. In the ORIGINAL corpus, encoder row T-1 carries rain(T) and
decoder row T carries rain(T+1) -- so the original holds rain(T+1), which is BEYOND entitlement.
In the corrected corpus, rain(T) sits in the decoder's first step and rain(T+1) in its second.

If the original's advantage comes from rain(T+1), that advantage must CONCENTRATE on rows where
rain(T+1) is large while rain(T) is small -- the day before a storm, where knowing tomorrow's rain
changes the forecast most and legitimate information says least. If instead the advantage is flat
across that partition, the original is not reading the future and (1) is the explanation.

Usage: m1_leak_future_rain_attribution.py --a <corrected member> --b gridmetm9
"""
import argparse, glob, json
import numpy as np, pandas as pd

AP = argparse.ArgumentParser()
AP.add_argument("--a", required=True, help="corrected member (e.g. gridmetm9fix3)")
AP.add_argument("--b", default="gridmetm9", help="original member")
AP.add_argument("--frame", default="val1")
AP.add_argument("--out", default="benchmarks/m1_leakaudit_future_rain_attribution.json")
a = AP.parse_args()

TRUE = "data/modern_corpus/v3/camels_corpus_gridmetfix3_m3"   # calendar-day aligned precip

def seeds_of(member):
    import re
    return {re.search(r"_s(\d+)_", f).group(1) for f in
            glob.glob(f"data/mblstm/l51_dumps/m1_{member}_s50?_{a.frame}.csv.gz")}

# ⚠️ Seed-match BOTH sides automatically. The original member has 5 seeds and the corrected
# screens have 3; comparing 3 against 5 would confound registration with seed depth, which is
# worth ~0.002 here on its own.
SEEDS = sorted(seeds_of(a.a) & seeds_of(a.b))
assert SEEDS, f"no shared seeds: {a.a}={sorted(seeds_of(a.a))} {a.b}={sorted(seeds_of(a.b))}"
print(f"seed-matched on {SEEDS}")

def load(member):
    fs = sorted(f"data/mblstm/l51_dumps/m1_{member}_s{s}_{a.frame}.csv.gz" for s in SEEDS)
    assert fs, f"no dumps for {member}"
    acc = []
    for f in fs:
        d = pd.read_csv(f, dtype={"station_id": str})
        if "h" in d.columns:
            d = d[d["h"] == 1]
        d["pred"] = 0.5 * (d["ylo"] + d["yhi"])
        acc.append(d[["station_id", "t0", "pred", "truth"]])
    m = pd.concat(acc).groupby(["station_id", "t0"], as_index=False).agg(
        pred=("pred", "mean"), truth=("truth", "mean"))
    return m, len(fs)

A, na = load(a.a)
B, nb = load(a.b)
assert na == nb == len(SEEDS), f"seed mismatch: {na} vs {nb} vs {len(SEEDS)}"
J = A.merge(B, on=["station_id", "t0"], suffixes=("_a", "_b"))
# ⚠️ RELATIVE tolerance, not absolute. The dumps round-trip through float32, so a discharge of
# 1.09e5 cfs carries ~0.008 of representation error -- an absolute 1e-6 threshold is four orders
# of magnitude too tight and fired on 32% of rows while the two columns correlate at 1.00000000.
# Same trap as the tmean guard earlier in this audit: a tolerance must match the STORED precision.
_bad = ~np.isclose(J["truth_a"], J["truth_b"], rtol=1e-5, atol=1e-3)
assert _bad.mean() < 1e-4, (
    f"truth disagrees on {_bad.mean():.2%} of rows beyond float32 noise; "
    f"max |diff| = {(J['truth_a'] - J['truth_b']).abs().max():.6g}")
J["date"] = pd.to_datetime(J["t0"]) + pd.Timedelta(days=1)   # h=1 target date
print(f"joined {len(J)} rows, {J.station_id.nunique()} basins, {na} seeds each")

# attach true rain(T) and rain(T+1) from the calendar-day-aligned corpus
rain = []
for sid, grp in J.groupby("station_id"):
    f = f"{TRUE}/{sid}.csv.gz"
    try:
        c = pd.read_csv(f, dtype={"date": str})
    except FileNotFoundError:
        continue
    c["date"] = pd.to_datetime(c["date"])
    c = c.set_index("date")["precipitation_sum"]
    r0 = grp["date"].map(c)
    r1 = (grp["date"] + pd.Timedelta(days=1)).map(c)
    rain.append(pd.DataFrame({"station_id": sid, "t0": grp["t0"].to_numpy(),
                              "rain_T": r0.to_numpy(), "rain_T1": r1.to_numpy()}))
R = pd.concat(rain)
J = J.merge(R, on=["station_id", "t0"])
J = J[np.isfinite(J["rain_T"]) & np.isfinite(J["rain_T1"])]
print(f"{len(J)} rows with both rain(T) and rain(T+1)")

# squared error per row, in the same units for both members
J["se_a"] = (J["pred_a"] - J["truth_a"]) ** 2
J["se_b"] = (J["pred_b"] - J["truth_b"]) ** 2
# normalize per basin by that basin's own variance so rows are comparable across basins
den = J.groupby("station_id")["truth_a"].transform(lambda s: np.var(s) if np.var(s) > 0 else np.nan)
J["adv"] = (J["se_a"] - J["se_b"]) / den      # >0 means B (the ORIGINAL) is better on that row
J = J[np.isfinite(J["adv"])]

# the partition: tomorrow's rain large, today's small = "the day before a storm".
# Per-basin thresholds are computed ONCE here and reused as plain columns; recomputing them
# inside the bootstrap made the loop ~100x slower for no change in the estimate.
J["thr_T1_hi"] = J.groupby("station_id")["rain_T1"].transform(lambda s: s.quantile(0.90))
J["thr_T_lo"] = J.groupby("station_id")["rain_T"].transform(lambda s: s.quantile(0.50))
J["thr_T_hi"] = J.groupby("station_id")["rain_T"].transform(lambda s: s.quantile(0.90))
J["is_pre"] = (J["rain_T1"] >= J["thr_T1_hi"]) & (J["rain_T"] <= J["thr_T_lo"])
J["is_dry"] = (J["rain_T"] <= J["thr_T_lo"]) & (J["rain_T1"] < J["thr_T1_hi"])
pre_storm = J["is_pre"].to_numpy()
on_storm = (J["rain_T"] >= J["thr_T_hi"]).to_numpy()
dry = J["is_dry"].to_numpy()

print(f"\nORIGINAL-minus-CORRECTED advantage by row class (>0 = the ORIGINAL is better)\n")
print(f"  {'row class':<34}{'n':>9}{'mean advantage':>17}{'share of total':>16}")
tot = float(J["adv"].sum())
rows = {}
for name, mask in [("PRE-STORM  rain(T+1) high, rain(T) low", pre_storm),
                   ("ON-STORM   rain(T) high", on_storm),
                   ("DRY        both low", dry),
                   ("ALL ROWS", np.ones(len(J), bool))]:
    sub = J[mask]
    if not len(sub): continue
    mean = float(sub["adv"].mean())
    share = float(sub["adv"].sum() / tot) if tot != 0 else np.nan
    print(f"  {name:<34}{len(sub):>9}{mean:>17.6f}{share:>15.1%}")
    rows[name] = {"n": len(sub), "mean_adv": mean, "share": share}

pm = rows.get("PRE-STORM  rain(T+1) high, rain(T) low", {}).get("mean_adv", np.nan)
dm = rows.get("DRY        both low", {}).get("mean_adv", np.nan)
# bootstrap the pre-storm vs dry contrast over BASINS (rows within a basin are correlated)
# Cluster bootstrap over BASINS (rows within a basin are correlated). Per-basin sums and counts
# are precomputed so each resample is an index lookup, not a groupby.
rng = np.random.default_rng(0)
g = J.groupby("station_id")
agg = pd.DataFrame({
    "pre_sum": g.apply(lambda d: d.loc[d["is_pre"], "adv"].sum(), include_groups=False),
    "pre_n":   g["is_pre"].sum(),
    "dry_sum": g.apply(lambda d: d.loc[d["is_dry"], "adv"].sum(), include_groups=False),
    "dry_n":   g["is_dry"].sum(),
})
PS, PN = agg["pre_sum"].to_numpy(float), agg["pre_n"].to_numpy(float)
DS, DN = agg["dry_sum"].to_numpy(float), agg["dry_n"].to_numpy(float)
nb_ = len(agg)
bs = []
for _ in range(4000):
    i = rng.integers(0, nb_, nb_)
    pn, dn = PN[i].sum(), DN[i].sum()
    if pn > 0 and dn > 0:
        bs.append(PS[i].sum() / pn - DS[i].sum() / dn)
lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
print(f"\npre-storm minus dry contrast: {pm - dm:+.6f}   95% CI [{lo:+.6f}, {hi:+.6f}]")
print()
if np.isfinite(lo) and lo > 0:
    print("⛔ LEAK SIGNATURE: the original's advantage CONCENTRATES on rows where tomorrow's rain")
    print("   is large and today's is small. That is the signature of reading rain(T+1), which is")
    print("   beyond entitlement ⇒ retract.")
elif np.isfinite(hi) and hi < 0:
    print("✅ ANTI-LEAK: the original is relatively WORSE before a storm, so its advantage cannot")
    print("   come from knowing future rain.")
else:
    print("✅ NO LEAK SIGNATURE: the original's advantage is FLAT across the partition (CI includes")
    print("   zero), so it is not concentrated where future rain would help. The score difference is")
    print("   a cost of the correction, not exploitation of rain(T+1).")
json.dump({"a": a.a, "b": a.b, "frame": a.frame, "seeds": na, "classes": rows,
           "pre_storm_minus_dry": float(pm - dm), "ci95": [float(lo), float(hi)]},
          open(a.out, "w"), indent=2)
print(f"-> {a.out}")
