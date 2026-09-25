#!/usr/bin/env python3
"""LEDGER 55 Track 2 verdict — SEED-MATCHED (3 seeds each).

l54_member_pair.py gives the control all 5 of its seeds against the candidate's 3.
A deeper seed average sits CLOSER to the consensus error, so it reads MORE
correlated — which biases the comparison in the candidate's favour. Redo it at
matched depth. Error-correlation definition lifted from l53_arfamily_map.py:41-60:
per-basin Pearson on raw error vectors, median over basins, mean over pairs.
"""
import importlib.util, json, os, sys
import numpy as np
spec = importlib.util.spec_from_file_location("l51", os.path.expanduser("~/riverwatch2/analysis/l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)

MB = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1"]
SEEDS = {501, 502, 503}

def err_corr(m, a, b):
    out = []
    for s, idx in m.groupby("station_id").indices.items():
        x = (m[a].to_numpy(float) - m.truth.to_numpy(float))[idx]
        y = (m[b].to_numpy(float) - m.truth.to_numpy(float))[idx]
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() < 20 or np.std(x[k]) < 1e-9 or np.std(y[k]) < 1e-9:
            continue
        out.append(np.corrcoef(x[k], y[k])[0, 1])
    return float(np.median(out))

for cand in ("nhar0h256fd30", "nhar0h256"):
    cols = MB + ["nhar", cand]
    m = l51.join(cols, "val1", seeds=SEEDS)
    vs_nhar = err_corr(m, cand, "nhar")
    vs_mb = float(np.mean([err_corr(m, cand, x) for x in MB]))
    mb_int = float(np.mean([err_corr(m, MB[i], MB[j])
                            for i in range(5) for j in range(i + 1, 5)]))
    solo = l51.per_basin(m, [cand]).median()
    six = l51.per_basin(m, MB + ["nhar"])
    sev = l51.per_basin(m, cols)
    d, lo, hi, br, nb = l51.paired_ci(sev, six)
    print(f"\n### {cand}  (3 seeds, val1, rows={len(m):,})")
    print(f"  solo                        {solo:.6f}")
    print(f"  err-corr vs nhar            {vs_nhar:.4f}")
    print(f"  err-corr vs MB block mean   {vs_mb:.4f}   (MB internal {mb_int:.4f})")
    print(f"  7-member median             {sev.median():.6f}   (6-member {six.median():.6f})")
    print(f"  paired LOO                  {d:+.6f}  CI[{lo:+.6f},{hi:+.6f}]  breadth {br:.3f}")
