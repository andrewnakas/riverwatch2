#!/usr/bin/env python3
"""LEDGER 42 B2 — σ-scenario / fraction-of-achievable on the TEST FRAME,
and the first NEAR-MEDIAN saturation analysis on the frame that is reported.

REGISTERED READS (ledger42_branch_table.md, NOT-A-QUERY):
  class 4 — observed discharge (`truth`) alone from the test window, no model
            predictions in the same computation. The gauge ceiling is a
            property of the gauge.
  class 2 — the ALREADY-SPENT per-basin NSEs in analysis/noq_test_result.json.
⛔ No prediction column is read here. Ceilings come from truth; skill comes
   from the spent query's JSON. Mixing a FRESH prediction read into this script
   would make it a spent query.

WHY IT MATTERS: [[near-median-cohort-the-real-target]] measured, TRAIN-side,
that under CENTRAL σ the ~78 basins that set the median are 78/78 saturated —
the median cannot move at all — while under OPTIMISTIC σ there is +0.0076 of
headroom. That analysis has NEVER been run on the test frame (0.8363), and its
own note says: "The *shape* transfers; the level does not. Do not quote 0.9581
as a test-side bound without recomputing on the test frame." This recomputes it.

Ceiling form (CORRECTED — [[ceiling-formula-CORRECTED]]):
    NSE_max = 1 - (M/V)(e^{σ²} - 1)          [OLD/simple]
    NSE_max = 1 - E[q²(e^{σ²}-1)] / Vobs     [NEW/heteroscedastic]
Both are reported; the NEW form is the one to quote.

Run on the 1080:  gpu1080/.venv/bin/python analysis/sigma_scenarios_test.py
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))

from gate_eval import build_merged          # noqa: E402
from target_feasibility import sigma_series  # noqa: E402

TARGET = 0.84
NEAR_MEDIAN_WINDOW = 0.01     # +/- NSE around the median: the cohort that sets it
SCENARIOS = {                 # (sigma_low, sigma_high, sigma_high_flashy)
    "central":    (0.30, 0.18, 0.35),
    "optimistic": (0.25, 0.13, 0.13),
}


def main():
    # ---- class 2: the spent query's per-basin skill ------------------------
    # The artifact lives at repo root on the 1080 and in analysis/ on the Mac;
    # md5 verified identical (c4d7639df8647308e48dedc9f45f824e). Assert on
    # CONTENT, not just existence — this is the one-shot query's output and a
    # different file here would silently change every number below.
    cands = [Path("noq_test_result.json"), Path("analysis/noq_test_result.json")]
    src = next((p for p in cands if p.exists()), None)
    if src is None:
        raise SystemExit(f"spent-query result not found in {[str(p) for p in cands]}")
    res = json.load(open(src))
    if res.get("mode") != "frozen_9stream_invmse" or res.get("basins") != 531:
        raise SystemExit(f"{src}: not the frozen 9-stream query "
                         f"(mode={res.get('mode')}, basins={res.get('basins')})")
    if abs(res["median_nse"] - 0.8362893021622821) > 1e-9:
        raise SystemExit(f"{src}: median {res['median_nse']} is not the "
                         f"recorded 0.8362893021622821 — wrong artifact")
    print(f"spent-query artifact: {src}")
    cur = pd.Series({str(k).zfill(8): float(v)
                     for k, v in res["per_basin_nse"].items()})
    print("=" * 78)
    print("LEDGER 42 B2 — TEST-FRAME CEILING & NEAR-MEDIAN SATURATION")
    print("=" * 78)
    print(f"spent-query median = {cur.median():.6f}  (recorded "
          f"{res['median_nse']:.6f})   basins={len(cur)}")

    # ---- class 4: TRUTH ONLY from the test frame --------------------------
    with contextlib.redirect_stderr(io.StringIO()):
        m, _cols = build_merged(train=False)
    m = m[["station_id", "date", "truth"]]      # <- predictions dropped here
    m["station_id"] = m["station_id"].astype(str).str.zfill(8)
    print(f"truth-only test frame: {len(m):,} rows, "
          f"{m.station_id.nunique()} basins")

    flash = m.groupby("station_id")["truth"].apply(
        lambda s: (np.percentile(s[s > 0], 99) / np.median(s[s > 0])
                   if (s > 0).sum() > 20 else np.nan))
    thr = float(flash.median())

    # the cohort that actually sets the reported median
    med_now = float(cur.median())
    near = cur[(cur - med_now).abs() <= NEAR_MEDIAN_WINDOW]
    print(f"near-median cohort (+/-{NEAR_MEDIAN_WINDOW}): {len(near)} basins "
          f"around {med_now:.4f}")

    out = {"target": TARGET, "current_median": med_now,
           "near_median_window": NEAR_MEDIAN_WINDOW,
           "near_median_n": int(len(near)), "scenarios": {}}

    print(f"\n{'scenario':<12}{'ceil_old':>10}{'ceil_new':>10}"
          f"{'all->ceil':>11}{'saturated':>12}{'nm_sat':>10}{'nm_head':>10}")
    print("-" * 78)
    for lab, (lo, hi, fl) in SCENARIOS.items():
        old, new = {}, {}
        for sid, g in m.groupby("station_id"):
            q = g["truth"].to_numpy(float)
            q = q[np.isfinite(q)]
            if q.size < 50:
                continue
            V = float(((q - q.mean()) ** 2).mean())
            if V <= 0:
                continue
            M = float((q ** 2).mean())
            sh = fl if flash.get(sid, 0) > thr else hi
            sg = sigma_series(q, lo, sh)
            old[sid] = 1 - (M / V) * (np.exp(np.nanmean(sg ** 2)) - 1)
            num = float(np.mean(q ** 2 * (np.exp(sg ** 2) - 1)))
            vobs = float(np.mean(q ** 2 * np.exp(sg ** 2)) - np.mean(q) ** 2)
            new[sid] = 1 - num / vobs
        old, new = pd.Series(old), pd.Series(new)
        idx = sorted(set(new.index) & set(cur.index))
        old, new, c = old.reindex(idx), new.reindex(idx), cur.reindex(idx)

        # every basin raised to its own ceiling => hard upper bound on the median
        allc_new = float(np.median(np.maximum(c, np.minimum(new, 1.0))))
        allc_old = float(np.median(np.maximum(c, np.minimum(old, 1.0))))
        sat = int((new <= c).sum())

        nm_idx = [s for s in near.index if s in new.index]
        nm_sat = int((new.reindex(nm_idx) <= c.reindex(nm_idx)).sum())
        nm_head = float(np.median(new.reindex(nm_idx) - c.reindex(nm_idx)))

        print(f"{lab:<12}{old.median():>10.4f}{new.median():>10.4f}"
              f"{allc_new:>11.4f}{sat:>7}/{len(new)}"
              f"{nm_sat:>6}/{len(nm_idx)}{nm_head:>10.4f}")

        reach = (TARGET <= allc_new)
        need = ((TARGET - med_now) / (allc_new - med_now) * 100
                if allc_new > med_now else float("nan"))
        out["scenarios"][lab] = {
            "sigma": [lo, hi, fl],
            "ceiling_median_old_form": float(old.median()),
            "ceiling_median_new_form": float(new.median()),
            "max_achievable_median_all_at_ceiling": allc_new,
            "max_achievable_median_old_form": allc_old,
            "basins_saturated": sat, "basins_scored": int(len(new)),
            "near_median_saturated": nm_sat,
            "near_median_n": len(nm_idx),
            "near_median_median_headroom": nm_head,
            "target_reachable_in_principle": bool(reach),
            "pct_of_recoverable_headroom_needed_for_target": float(need),
        }

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    for lab, s in out["scenarios"].items():
        verdict = ("REACHABLE in principle" if s["target_reachable_in_principle"]
                   else "ABOVE THE MEASUREMENT CEILING")
        print(f"  {lab:<11}: max achievable median = "
              f"{s['max_achievable_median_all_at_ceiling']:.4f}  => {TARGET} is "
              f"{verdict}")
        print(f"               near-median cohort saturated "
              f"{s['near_median_saturated']}/{s['near_median_n']}, "
              f"median headroom {s['near_median_median_headroom']:+.4f}")
        if s["target_reachable_in_principle"]:
            print(f"               reaching it needs "
                  f"{s['pct_of_recoverable_headroom_needed_for_target']:.0f}% "
                  f"of ALL recoverable headroom, everywhere, at once")

    Path("benchmarks/ledger42_b2_sigma_test.json").write_text(json.dumps(out, indent=2))
    print("\nwrote benchmarks/ledger42_b2_sigma_test.json")


if __name__ == "__main__":
    sys.exit(main())
