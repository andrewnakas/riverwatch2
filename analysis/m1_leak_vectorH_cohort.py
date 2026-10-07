#!/usr/bin/env python3
"""VECTOR H -- does the cohort rule select EASY basins?

The MODERN-1 cohort keeps a basin only if it has >=90% daily discharge observations over the
P-MODERN TEST window (2018-10-01..2025-09-30). 41 of the 531 are dropped, every one for coverage.
The rule was fixed in PREREG_v2.md before any score existed, so it is not p-hacking -- but it is
still a basin set conditioned on TEST-PERIOD data, and if gauges that keep reporting are
systematically easier to predict, the modern headline median is inflated relative to "the 531".

Availability carries no model information, so this is not leakage in the strict sense. The question
is whether it biases the headline, and that is measurable with artifacts already on disk and zero
GPU: the LEGACY dumps (camels531_l51_*) cover all 531 basins on the 1989-1999 test window. Basin
difficulty is largely a basin property -- terrain, aridity, flashiness, gauge quality -- so the
legacy per-basin NSE of the 41 dropped basins versus the 490 kept ones measures the selection
effect directly, on a decade where every one of them still reported.

A difference-of-medians is reported here on purpose and labelled as such: the two groups are
DIFFERENT basins, so no pairing exists and the paired estimator this campaign normally insists on
does not apply.
"""
import glob, json
import numpy as np, pandas as pd

cohort = set(str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"])
allids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
dropped = sorted(allids - cohort)
print(f"531 list: {len(allids)}   cohort: {len(cohort)}   dropped: {len(dropped)}")

def per_basin_nse(member, frame="test1"):
    fs = sorted(glob.glob(f"data/mblstm/l51_dumps/camels531_l51_{member}_s50?_{frame}.csv.gz"))
    if not fs:
        return None, 0
    acc = None
    for f in fs:
        d = pd.read_csv(f, dtype={"station_id": str})
        d = d[d["h"] == 1] if "h" in d.columns else d
        d["pred"] = 0.5 * (d["ylo"] + d["yhi"])
        g = d.groupby(["station_id", "t0"], as_index=False)[["pred", "truth"]].mean()
        acc = g if acc is None else pd.concat([acc, g])
    # seed-average the predictions on the shared grid, the record's readout
    m = acc.groupby(["station_id", "t0"], as_index=False).agg(
        pred=("pred", "mean"), truth=("truth", "mean"))
    out = {}
    for sid, grp in m.groupby("station_id"):
        y, p = grp["truth"].to_numpy(float), grp["pred"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        if ok.sum() < 20: continue
        y, p = y[ok], p[ok]
        den = np.sum((y - y.mean()) ** 2)
        if den <= 0: continue
        out[str(sid).zfill(8)] = 1.0 - np.sum((y - p) ** 2) / den
    return out, len(fs)

rows = []
for member in ["fused3h1", "daymeth1", "nldash1"]:
    nse, nf = per_basin_nse(member)
    if not nse:
        print(f"  {member}: no legacy dumps"); continue
    kept = [v for k, v in nse.items() if k in cohort]
    drop = [v for k, v in nse.items() if k in set(dropped)]
    if len(drop) < 5:
        print(f"  {member}: only {len(drop)} dropped basins scoreable"); continue
    mk, md = float(np.median(kept)), float(np.median(drop))
    rows.append((member, nf, len(kept), len(drop), mk, md, mk - md))

print(f"\nLEGACY per-basin day-1 NSE (test window 1989-10-01..1999-09-30, all 531 reporting)\n")
print(f"{'member':<12}{'seeds':>6}{'n kept':>8}{'n dropped':>11}"
      f"{'median KEPT':>13}{'median DROPPED':>16}{'difference':>12}")
for m, nf, nk, nd, mk, mdp, diff in rows:
    print(f"{m:<12}{nf:>6}{nk:>8}{nd:>11}{mk:>13.6f}{mdp:>16.6f}{diff:>+12.6f}")

if rows:
    diffs = [r[6] for r in rows]
    med = float(np.median(diffs))
    print(f"\nmedian across members: {med:+.6f}  (kept minus dropped; >0 = the rule kept EASIER basins)")
    # bootstrap the group difference for the strongest member
    m, nf, nk, nd, mk, mdp, _ = rows[0]
    nse, _ = per_basin_nse(m)
    kept = np.array([v for k, v in nse.items() if k in cohort])
    drop = np.array([v for k, v in nse.items() if k in set(dropped)])
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(kept, len(kept))) - np.median(rng.choice(drop, len(drop)))
          for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{m}: difference-of-medians {mk-mdp:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}]")
    print("  (difference-of-medians, NOT paired: the two groups are different basins, so no")
    print("   pairing exists and this campaign's usual paired estimator does not apply.)")
    print()
    if lo <= 0 <= hi:
        print("✅ VECTOR H: no measurable difficulty difference. The test-window coverage rule does")
        print("   not detectably select easier basins; the headline is not inflated by the cohort.")
    elif lo > 0:
        print(f"⚠️ VECTOR H FIRES: the kept basins are measurably EASIER by {mk-mdp:+.4f} on the")
        print("   legacy frame. The modern headline is NOT comparable to a 531-basin figure and must")
        print("   be reported as a 490-basin cohort number with this bias stated.")
    else:
        print(f"⚠️ the kept basins are measurably HARDER by {mdp-mk:+.4f} -- the headline is")
        print("   conservative relative to the 531, not inflated.")
    json.dump({"dropped": dropped, "per_member": [
        {"member": r[0], "n_kept": r[2], "n_dropped": r[3], "median_kept": r[4],
         "median_dropped": r[5], "difference": r[6]} for r in rows],
        "ci95_first_member": [float(lo), float(hi)],
        "estimator": "difference-of-medians (unpaired: different basins)"},
        open("benchmarks/m1_leakaudit_vectorH_cohort.json", "w"), indent=2)
