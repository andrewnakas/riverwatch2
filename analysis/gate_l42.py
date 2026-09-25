#!/usr/bin/env python3
"""LEDGER 42 — the Track-A gate. Written BEFORE any treatment number existed.

Two bars, both pre-registered in benchmarks/ledger42_branch_table.md and
neither movable by anything this script sees:

  PRIMARY   complete seed separation, min(treatment) > max(control),
            >= --min-seeds per arm (3 v 3 = p 0.05 by rank-sum; fewer
            degenerates to "is A > B" and returns INCONCLUSIVE, never PASS).
  SHIP      recombine the treatment stream into the frozen 9, refit the ONE
            GLOBAL inverse-MSE vector on train rows, score the val slice:
              recombined median      >= 0.954458   (anchor 0.950458 + 0.004)
              near-median delta      >= +0.0008
            BOTH required. Fail either and the candidate does not ship and
            does not appear in Stage Q.

⚠️ WHY THIS IS NOT `gate_l41.py --out something_else`: that script PRINTS the
measured null for the **daymet** recipe ("median dNSE -0.0127..+0.0127 ...")
as a fixed string. Pointed at an lstm_multi arm it would display the wrong
noise band beside the right numbers, and a reader quoting the output would
read a ~15x too-wide null. This campaign has already shipped one canned
explanation attached to a correct verdict; the null is loaded from JSON here.

NULL SOURCES (measured before the treatment existed, never tuned to a result):
  stream level    benchmarks/ledger42_s03_null_multi.json
  ensemble level  benchmarks/ledger42_s03b_null_ensemble.json

⛔ Train-side frame only. Nothing here touches 1995-2010.

Usage:
  python analysis/gate_l42.py \
      --treatment gpu1080/dumps/camels531ls_l42evsubtreat_nhlstm_TRAIN_s{111,222,333}.csv.gz \
      --control   gpu1080/dumps/camels531ls_l42evsubctrl_nhlstm_TRAIN_s{111,222,333}.csv.gz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))

from gate_l41 import load, per_basin_nse, seed_average, CV_VAL_START  # noqa: E402
from gate_eval import build_merged, L                                 # noqa: E402
from phase_c import load_seed_avg                                     # noqa: E402

THETA, LAM = 4.0, 0.25
CV_FIT_END, CV_VAL_END = "1990-09-30", "1995-09-30"
ANCHOR = 0.950458
SHIP_MEDIAN = 0.954458          # anchor + 0.004
SHIP_NEAR_MEDIAN = 0.0008
NEAR_WINDOW = 0.01
TRIPWIRE = 0.02
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")


def _null(path, keys):
    try:
        d = json.loads(Path(path).read_text())
        return {k: d.get(k) for k in keys}, str(path)
    except Exception as e:
        return None, f"{path} UNAVAILABLE ({e})"


def ens_per_basin_nse(df, col):
    out = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = g[col].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(p)
        y, p = y[m], p[m]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out[sid] = 1.0 - np.mean((y - p) ** 2) / np.var(y)
    return pd.Series(out)


def recombine(multi_paths):
    """Frozen 9 streams with lstm_multi supplied by the caller; refit the one
    global inverse-MSE vector on TRAIN rows; score the val slice."""
    tr, cols = build_merged(train=True, drop=("lstm_multi",))
    def add(m, name, paths):
        d = load_seed_avg([Path(p) for p in paths])[["station_id", "date", "pred"]]
        before = len(m)
        m = m.merge(d.rename(columns={"pred": name}),
                    on=["station_id", "date"], how="inner")
        if len(m) < 0.95 * before:
            raise SystemExit(f"{name}: join lost >5% of rows — schema mismatch")
        return m
    tr = add(tr, "lstm_multi", multi_paths)
    tr = add(tr, "lstm_multi5",
             [L / f"camels531ls_multi5_nhlstm_TRAIN_{s}.csv.gz" for s in MULTI5])
    tr = add(tr, "lstm_multi6",
             [L / f"camels531ls_multi6_nhlstm_TRAIN_{s}.csv.gz" for s in MULTI6])
    cols = list(cols) + ["lstm_multi", "lstm_multi5", "lstm_multi6"]

    d = pd.to_datetime(tr["date"])
    fit = tr[d <= CV_FIT_END]
    val = tr[(d >= CV_VAL_START) & (d <= CV_VAL_END)].copy()
    truth = fit["truth"].to_numpy(float)
    mse = np.array([np.nanmean((fit[c].to_numpy(float) - truth) ** 2) for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM * (np.ones(len(cols)) / len(cols)) + (1 - LAM) * raw
    w = w / w.sum()
    val["ens"] = (val[cols].to_numpy(float) * w).sum(axis=1)
    return ens_per_basin_nse(val, "ens"), dict(zip(cols, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", nargs="+", required=True)
    ap.add_argument("--control", nargs="+", required=True)
    ap.add_argument("--from-date", default=CV_VAL_START)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--skip-ship", action="store_true",
                    help="separation only (used to dry-run the branches)")
    ap.add_argument("--out", default="benchmarks/ledger42_gate_trackA.json")
    a = ap.parse_args()

    s_null, s_src = _null("benchmarks/ledger42_s03_null_multi.json",
                          ["median_delta_range", "breadth_range",
                           "near_median_delta_range", "per_seed_median_spread"])
    e_null, e_src = _null("benchmarks/ledger42_s03b_null_ensemble.json",
                          ["median_delta_null", "near_median_delta_null"])

    # ---------------- PRIMARY: per-seed medians, never hidden in a mean ------
    per_seed = {"treatment": [], "control": []}
    for arm, paths in (("treatment", a.treatment), ("control", a.control)):
        for p in paths:
            d = load(p)
            d = d[d["date"] >= a.from_date]
            n = per_basin_nse(d)
            per_seed[arm].append((Path(p).name, float(n.median()), len(n)))

    print("=" * 78)
    print(f"LEDGER 42 TRACK-A GATE — train-side val slice from {a.from_date}")
    print("=" * 78)
    for arm in ("control", "treatment"):
        print(f"{arm}:")
        for name, med, nb in per_seed[arm]:
            print(f"  {med:.6f}  ({nb} basins)  {name}")

    t = [m for _, m, _ in per_seed["treatment"]]
    c = [m for _, m, _ in per_seed["control"]]
    separated = min(t) > max(c)
    reverse = max(t) < min(c)
    underpowered = min(len(t), len(c)) < a.min_seeds
    print(f"\ntreatment [{min(t):.6f}, {max(t):.6f}]   "
          f"control [{min(c):.6f}, {max(c):.6f}]")
    if s_null:
        print(f"  same-recipe seed spread (null, {Path(s_src).name}): "
              f"{s_null['per_seed_median_spread']:.6f}")

    print(f"\n{'PRIMARY BAR: complete seed separation':46s} "
          f"{'PASS' if separated else 'FAIL'}")
    if underpowered:
        print(f"  ⚠️ UNDERPOWERED: {len(t)} vs {len(c)} seeds; {a.min_seeds} per "
              f"arm required. Below that, separation is just an ordering.")
    if not separated:
        if reverse:
            print("  ⚠️ REVERSE SEPARATION: every treatment seed BELOW every "
                  f"control seed ({max(t):.6f} < {min(c):.6f}). NOT 'within "
                  "the seed spread' — consistently WORSE.")
        else:
            print("  the arms OVERLAP — within the seed spread.")

    # ---------------- descriptive, on the CORRECT null -----------------------
    ta = seed_average(a.treatment, a.from_date)
    ca = seed_average(a.control, a.from_date)
    nt, nc = per_basin_nse(ta), per_basin_nse(ca)
    common = sorted(set(nt.index) & set(nc.index))
    if len(common) < 400:
        sys.exit(f"only {len(common)} basins in common — refusing to score.")
    nt, nc = nt[common], nc[common]
    dd = nt - nc
    med_d = float(dd.median())
    breadth = float((dd > 0).mean())
    near_idx = nc[(nc - float(nc.median())).abs() <= NEAR_WINDOW].index
    near_d = float(dd[near_idx].mean()) if len(near_idx) else float("nan")
    print(f"\nseed-averaged arms, {len(common)} basins (DESCRIPTIVE — not a bar):")
    print(f"  paired median Δ {med_d:+.6f}   breadth {100*breadth:.1f}%   "
          f"near-median {near_d:+.6f} (n={len(near_idx)})")
    if s_null:
        print(f"  STREAM-level null for THIS recipe ({Path(s_src).name}): "
              f"median Δ {s_null['median_delta_range']}, "
              f"near-median {s_null['near_median_delta_range']}")
    if med_d > TRIPWIRE:
        print(f"\n⚠️⚠️ TRIPWIRE: Δ > +{TRIPWIRE} was pre-registered as a LEAKAGE "
              "signal, not a success. Re-audit before celebrating.")

    # ---------------- SHIP: contribution, the binding bar --------------------
    ship = None
    if not a.skip_ship:
        print("\n" + "-" * 78)
        print("SHIP BAR — recombination into the frozen 9 (CONTRIBUTION)")
        print("-" * 78)
        s_t, w_t = recombine([Path(p) for p in a.treatment])
        s_c, w_c = recombine([Path(p) for p in a.control])
        m_t, m_c = float(s_t.median()), float(s_c.median())
        ci = sorted(set(s_t.index) & set(s_c.index))
        de = s_t[ci] - s_c[ci]
        near_e = s_c[ci][(s_c[ci] - float(s_c[ci].median())).abs() <= NEAR_WINDOW].index
        near_de = float(de[near_e].mean()) if len(near_e) else float("nan")
        print(f"  control-recombined  median {m_c:.6f}   (anchor {ANCHOR:.6f})")
        print(f"  treatment-recombined median {m_t:.6f}")
        print(f"  lstm_multi weight: treat {w_t['lstm_multi']:.4f}  "
              f"ctrl {w_c['lstm_multi']:.4f}")
        print(f"  near-median Δ (ensemble) {near_de:+.6f}  (n={len(near_e)})")
        if e_null:
            print(f"  ENSEMBLE-level null ({Path(e_src).name}): "
                  f"median {e_null['median_delta_null']}, "
                  f"near-median {e_null['near_median_delta_null']}")
        b1 = m_t >= SHIP_MEDIAN
        b2 = near_de >= SHIP_NEAR_MEDIAN
        print(f"\n  {'recombined median >= ' + str(SHIP_MEDIAN):46s} "
              f"{'PASS' if b1 else 'FAIL'}  ({m_t:.6f})")
        print(f"  {'near-median Δ >= +' + str(SHIP_NEAR_MEDIAN):46s} "
              f"{'PASS' if b2 else 'FAIL'}  ({near_de:+.6f})")
        ship = {"recombined_median_treatment": m_t,
                "recombined_median_control": m_c,
                "near_median_delta_ensemble": near_de,
                "bar_median_pass": bool(b1), "bar_near_median_pass": bool(b2),
                "ship": bool(b1 and b2),
                "weights_treatment": {k: float(v) for k, v in w_t.items()}}

    # ---------------- verdict ------------------------------------------------
    if underpowered:
        verdict = "INCONCLUSIVE (underpowered)"
    elif not separated:
        verdict = ("FAIL (reverse separation)" if reverse else "FAIL (arms overlap)")
    elif ship is None:
        verdict = "SEPARATION PASS (ship bar not evaluated)"
    elif ship["ship"]:
        verdict = "PASS — separation AND contribution"
    else:
        verdict = "FAIL AT SHIP BAR (separated but does not contribute)"
    print(f"\n{'=' * 78}\nLEDGER 42 TRACK-A VERDICT: {verdict}\n{'=' * 78}")
    if verdict.startswith("FAIL AT SHIP"):
        print("⭐ The stream improved and the ensemble did not follow — the "
              "outcome AMENDMENT 3 predicted from the weight arithmetic "
              "(0.131 weight ⇒ +0.004 needs ~+0.03 on the stream). Report it "
              "as a STREAM gain, never as a record gain.")

    Path(a.out).write_text(json.dumps({
        "from_date": a.from_date, "min_seeds": a.min_seeds,
        "per_seed": {k: [{"dump": n, "median_nse": m, "n_basins": b}
                         for n, m, b in v] for k, v in per_seed.items()},
        "seed_separation": bool(separated), "reverse_separation": bool(reverse),
        "underpowered": bool(underpowered),
        "treatment_range": [min(t), max(t)], "control_range": [min(c), max(c)],
        "descriptive": {"paired_median_delta": med_d, "breadth": breadth,
                        "near_median_delta": near_d, "n_basins": len(common)},
        "null_sources": {"stream": s_src, "ensemble": e_src},
        "stream_null": s_null, "ensemble_null": e_null,
        "ship": ship, "verdict": verdict,
    }, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
