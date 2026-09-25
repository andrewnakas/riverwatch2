#!/usr/bin/env python3
"""LEDGER 56 — re-run the protocol's REMOVAL check after the composition changed.

The rule (PREREG, applied to nldash1/maurerh1 in L51 and to the AR pair in L55):
a member is removed only if the paired val1 gain from removing it has a CI that
EXCLUDES zero. The candidate set is the one already established by those ledgers
plus any member whose LOO point estimate is negative here. This is the same
fixed check re-run on a changed composition, not a search over subsets.
"""
import importlib.util, json, os
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("l51", "analysis/l51_withq_score.py")
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)
NINE = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1",
        "nhar", "nhar0", "nhar0h256", "fused3h1d3"]

m = None
for x in NINE:
    d = l51.load_member(x, "val1", seeds={501, 502, 503} if x == "fused3h1d3" else None)
    m = d if m is None else m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")


def pb(cols, how="mean"):
    P = m[cols].to_numpy(float)
    pred = P.mean(1) if (how == "mean" or len(cols) < 4) else (P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2)
    out = {}
    y = m.truth.to_numpy(float)
    for s, i in m.groupby("station_id").indices.items():
        yy, pp = y[i], pred[i]
        k = np.isfinite(yy) & np.isfinite(pp)
        yy, pp = yy[k], pp[k]
        if len(yy) < l51.MIN_ROWS or np.var(yy) < 1e-9:
            continue
        out[s] = 1 - np.mean((yy - pp) ** 2) / np.var(yy)
    return pd.Series(out)


full = pb(NINE)
print("\nALL9 mean %.6f   rows %d  basins %d" % (full.median(), len(m), m.station_id.nunique()))
print("%-26s %9s %19s %26s %s" % ("removal", "median", "gain from removing", "CI95", "breadth"))
cands = {"drop nhar": ["nhar"], "drop nhar0": ["nhar0"], "drop BOTH nhar+nhar0": ["nhar", "nhar0"],
         "drop nldash1": ["nldash1"], "drop nhar+nhar0+nldash1": ["nhar", "nhar0", "nldash1"],
         "drop aorch1": ["aorch1"]}
res = {"ALL9_mean": float(full.median()), "rows": int(len(m)), "removals": {}}
for name, drop in cands.items():
    sub = [x for x in NINE if x not in drop]
    e = pb(sub)
    d, lo, hi, br, _ = l51.paired_ci(e, full)
    res["removals"][name] = {"median": float(e.median()), "gain": d, "ci95": [lo, hi],
                             "breadth": br, "justified": bool(lo > 0)}
    flag = "  <-- REMOVE (CI excludes zero)" if lo > 0 else ""
    print("%-26s %.6f   %+.6f   [%+.6f,%+.6f]  %.3f%s" % (name, e.median(), d, lo, hi, br, flag))
for how in ("mean", "trimmed"):
    keep7 = [x for x in NINE if x not in ("nhar", "nhar0")]
    res[f"ALL9_{how}"] = float(pb(NINE, how).median())
    res[f"SEVEN_noARpair_{how}"] = float(pb(keep7, how).median())
    print("  ALL9 %-8s %.6f   7-member(no AR pair) %-8s %.6f"
          % (how, res[f"ALL9_{how}"], how, res[f"SEVEN_noARpair_{how}"]))
os.makedirs("benchmarks/l56", exist_ok=True)
json.dump(res, open("benchmarks/l56/REMOVAL9_val1.json", "w"), indent=1)
print("\nwrote benchmarks/l56/REMOVAL9_val1.json")
