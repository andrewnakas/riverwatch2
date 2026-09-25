#!/usr/bin/env python3
"""LEDGER 42 A1 — THE MAGNITUDE-AWARE FORCING REFEREE. Zero GPU.

WHY IT EXISTS
-------------
`scripts/ledger41_forcing_referee.py` judges a precip product by
peak-over-lag corr(precip, dQ). That is deliberately registration-invariant —
and therefore **scale-invariant**: it measures TIMING and covariation, nothing
about magnitude. Its own scope note flags this, and it bites here specifically,
because the measured residual on this campaign is MAGNITUDE, not phase:
the model is ON TIME and TOO SMALL on the ~1% of days carrying 96.5% of the
squared error.

This is the magnitude counterpart. The referee is STREAMFLOW, never another
forcing (forcing-to-forcing agreement had a 63% false-positive rate on the
Maurer realignment arm).

THE STATISTICS (both registered before any number existed)
----------------------------------------------------------
Per basin, over events:
  S1  CV of the event runoff ratio  (quickflow volume / precip depth)  LOWER better
  S2  R² of  volume ~ depth  through the origin                       HIGHER better

Both are invariant to a GLOBAL precip scale factor (which a model can simply
learn) and sensitive to PER-EVENT magnitude error (which it cannot). That is
the invariance this screen needs.
  reported for continuity only:  corr(depth, volume)  — scale-invariant, NOT a bar

CALIBRATION DECIDES WHICH IS A1-1 (pre-committed rule, ledger42_branch_table
AMENDMENT 1): the selected statistic must rank conus404 LAST and daymet FIRST
against the measured member-skill ordering
(daymet .7542 > maurer .7453 > nldas .7336 > aorc .7047 >> conus404 .5094).
If neither qualifies, A1 is an INVALID SCREEN and licenses nothing.

EVENT DEFINITION (fixed here, before any result)
------------------------------------------------
  event peak  : q in the basin's top 1% of positive-flow days AND the maximum
                within +/-3 days (so one storm counts once, not five times)
  depth       : sum of candidate precip over [t-3, t]
  baseflow    : minimum q over [t-10, t-4]  (pre-event, outside the depth window)
  volume      : sum over [t-3, t+3] of max(q - baseflow, 0)
  kept        : depth > 1 mm and volume > 0; basin needs >= 10 such events

Area cancels: S1 is a CV and S2 an R², so cfs-days vs mm never needs converting.

WINDOW: train side only, 1981-01-01 .. 1995-09-30. Nothing here touches the
test window.

Usage:
  python scripts/ledger42_magnitude_referee.py --calibrate
  python scripts/ledger42_magnitude_referee.py --candidates DIR1,DIR2 ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))

CORP = Path("gpu1080/corpora671")
SUBSET = Path("data/camels_gauge_ids.json")
TRAIN_START, TRAIN_END = "1981-01-01", "1995-09-30"
MIN_EVENTS = 10
MIN_BASINS = 500          # registered: below this the verdict is INCONCLUSIVE
REF = "camels_corpus_daymet_v2"

# measured member skill — the calibration's known truth
KNOWN_SKILL = {
    "camels_corpus_daymet_v2": 0.7542,
    "camels_corpus_maurer_v2": 0.7453,
    "camels_corpus_nldas_v2": 0.7336,
    "camels_corpus_aorc_v2": 0.7047,
    "camels_corpus_conus404_v2": 0.5094,
}


def basin_stats(path, q_series=None):
    """S1, S2 (and corr) for one basin/product. q comes from the product's own
    q_cfs unless a shared q_series is supplied — the referee must judge every
    product against the SAME discharge, so callers pass q_series."""
    df = pd.read_csv(path, usecols=["date", "q_cfs", "precipitation_sum"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].sort_values("date")
    if q_series is not None:
        df = df.drop(columns=["q_cfs"]).merge(q_series, on="date", how="inner")
    if len(df) < 500:
        return None
    q = df["q_cfs"].to_numpy(float)
    p = df["precipitation_sum"].to_numpy(float)
    if not np.isfinite(q).any() or not np.isfinite(p).any():
        return None

    pos = q[np.isfinite(q) & (q > 0)]
    if pos.size < 200:
        return None
    thr = np.percentile(pos, 99)

    depths, vols = [], []
    n = len(q)
    for t in range(10, n - 3):
        qt = q[t]
        if not np.isfinite(qt) or qt < thr:
            continue
        win = q[t - 3:t + 4]
        if not np.isfinite(win).all() or qt < np.nanmax(win):
            continue                      # not the local peak: same storm
        pre = q[t - 10:t - 3]
        if not np.isfinite(pre).all():
            continue
        base = float(np.min(pre))
        d = float(np.nansum(p[t - 3:t + 1]))
        v = float(np.sum(np.maximum(win - base, 0.0)))
        if d > 1.0 and v > 0:
            depths.append(d)
            vols.append(v)

    if len(depths) < MIN_EVENTS:
        return None
    d = np.array(depths)
    v = np.array(vols)
    rr = v / d
    s1 = float(np.std(rr, ddof=1) / np.mean(rr)) if np.mean(rr) > 0 else np.nan
    # R^2 through the origin: v ~ b*d
    b = float((d * v).sum() / (d * d).sum())
    ss_res = float(((v - b * d) ** 2).sum())
    ss_tot = float((v ** 2).sum())
    s2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = float(np.corrcoef(d, v)[0, 1]) if len(d) > 2 else np.nan
    return {"n_events": len(d), "S1_cv_rr": s1, "S2_r2_origin": s2, "corr": corr}


def load_q(basin):
    """The reference discharge series, shared by every product."""
    df = pd.read_csv(CORP / REF / f"{basin}.csv.gz", usecols=["date", "q_cfs"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def run_product(corpus, basins):
    out = {}
    for b in basins:
        p = CORP / corpus / f"{b}.csv.gz"
        if not p.exists():
            continue
        try:
            r = basin_stats(p, q_series=load_q(b))
        except Exception:
            continue
        if r:
            out[b] = r
    return out


def paired(a, b, key, lower_better):
    """Paired per-basin comparison of product a vs reference b."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    da = np.array([a[i][key] for i in ids], float)
    db = np.array([b[i][key] for i in ids], float)
    m = np.isfinite(da) & np.isfinite(db)
    da, db = da[m], db[m]
    if len(da) == 0:
        return None
    diff = (db - da) if lower_better else (da - db)   # >0 => candidate better
    return {"n": int(len(da)), "median_candidate": float(np.median(da)),
            "median_reference": float(np.median(db)),
            "paired_median_gain": float(np.median(diff)),
            "win_share": float((diff > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="run the pre-committed calibration on known products")
    ap.add_argument("--candidates", default="",
                    help="comma-separated corpus dir names to screen")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/ledger42_magnitude_referee.json")
    a = ap.parse_args()

    # data/camels_gauge_ids.json is {"_note":..., "671":[...], "531":[...]}
    raw = json.loads(SUBSET.read_text())
    if isinstance(raw, dict):
        if "531" not in raw:
            raise SystemExit(f"expected a '531' key in {SUBSET}, got {list(raw)}")
        raw = raw["531"]
    basins = [str(b).zfill(8) for b in raw]
    if len(basins) != 531:
        raise SystemExit(f"expected 531 basins, got {len(basins)}")
    if a.limit:
        basins = basins[:a.limit]
    print(f"basins requested: {len(basins)}  window {TRAIN_START}..{TRAIN_END}")
    if len(basins) < MIN_BASINS and not a.limit:
        print(f"⚠️ fewer than {MIN_BASINS} basins — verdict will be INCONCLUSIVE")

    products = list(KNOWN_SKILL) if a.calibrate else []
    products += [c for c in a.candidates.split(",") if c]
    if REF not in products:
        products.insert(0, REF)

    res, per = {}, {}
    for c in products:
        if not (CORP / c).exists():
            print(f"  {c:<42} MISSING — skipped")
            continue
        r = run_product(c, basins)
        per[c] = r
        s1 = np.nanmedian([v["S1_cv_rr"] for v in r.values()]) if r else np.nan
        s2 = np.nanmedian([v["S2_r2_origin"] for v in r.values()]) if r else np.nan
        cr = np.nanmedian([v["corr"] for v in r.values()]) if r else np.nan
        ev = np.nanmedian([v["n_events"] for v in r.values()]) if r else np.nan
        res[c] = {"basins": len(r), "median_S1_cv_rr": float(s1),
                  "median_S2_r2_origin": float(s2), "median_corr": float(cr),
                  "median_events": float(ev)}
        print(f"  {c:<42} n={len(r):3d}  S1={s1:.4f}  S2={s2:.4f}  "
              f"corr={cr:.4f}  events={ev:.0f}")

    out = {"window": [TRAIN_START, TRAIN_END], "min_events": MIN_EVENTS,
           "min_basins": MIN_BASINS, "products": res}

    # ---- calibration verdict (pre-committed selection rule) ----------------
    if a.calibrate:
        known = [c for c in KNOWN_SKILL if c in res]

        # ⚠️⚠️ LIVENESS GUARD — added after the smoke test PASSED ON NOTHING.
        # With n=0 everywhere every median is NaN; sorted() on NaN keys is a
        # NO-OP, so the products kept the order they were listed in — which is
        # KNOWN_SKILL's order — and the calibration reported "qualifies=True,
        # rank-corr=+1.000" against an empty computation. A gate that returns
        # its strongest PASS on an empty input is the exact failure shape this
        # campaign keeps hitting. The verdict is now refused unless the inputs
        # could actually have produced it.
        dead = [c for c in known
                if res[c]["basins"] == 0
                or not np.isfinite(res[c]["median_S1_cv_rr"])
                or not np.isfinite(res[c]["median_S2_r2_origin"])]
        thin = [c for c in known if res[c]["basins"] < MIN_BASINS]
        # ⚠️ A --limit run is a SMOKE TEST BY CONSTRUCTION and may never render
        # a verdict: at n=2 the smoke run happily printed "INVALID SCREEN",
        # which is a verdict-shaped noise reading. This campaign has already
        # been burned by a 10-gauge screen that sign-flipped against n=510.
        if len(known) < len(KNOWN_SKILL) or dead or thin or a.limit:
            msg = (f"CALIBRATION REFUSED — "
                   f"{'--limit smoke run (no verdict permitted); ' if a.limit else ''}"
                   f"products missing={sorted(set(KNOWN_SKILL) - set(known))} "
                   f"dead={dead} thin(<{MIN_BASINS})={thin}")
            print("\n  ⛔ " + msg)
            print("     No statistic is selected and NO candidate may be "
                  "screened. This is a REFUSAL, not a FAIL.")
            out["calibration"] = {"verdict": "REFUSED", "reason": msg,
                                  "selected_statistic": None,
                                  "basins_per_product": {c: res[c]["basins"]
                                                         for c in known}}
            Path(a.out).write_text(json.dumps(out, indent=2))
            print(f"\nwrote {a.out}")
            return 2

        # S1 lower is better -> ascending rank is "best first"
        r_s1 = sorted(known, key=lambda c: res[c]["median_S1_cv_rr"])
        r_s2 = sorted(known, key=lambda c: -res[c]["median_S2_r2_origin"])
        r_sk = sorted(known, key=lambda c: -KNOWN_SKILL[c])

        def qualifies(order):
            return (order[-1] == "camels_corpus_conus404_v2"
                    and order[0] == "camels_corpus_daymet_v2")

        q1, q2 = qualifies(r_s1), qualifies(r_s2)
        sel = "S1_cv_rr" if q1 else ("S2_r2_origin" if q2 else None)

        def spearman(order):
            rk = {c: i for i, c in enumerate(order)}
            x = np.array([rk[c] for c in known], float)
            y = np.array([r_sk.index(c) for c in known], float)
            return float(np.corrcoef(x, y)[0, 1]) if len(known) > 2 else np.nan

        print("\n--- CALIBRATION (pre-committed rule) ---")
        print(f"  known skill order : {[c.split('_')[2] for c in r_sk]}")
        print(f"  S1 order          : {[c.split('_')[2] for c in r_s1]}  "
              f"qualifies={q1}  rank-corr={spearman(r_s1):+.3f}")
        print(f"  S2 order          : {[c.split('_')[2] for c in r_s2]}  "
              f"qualifies={q2}  rank-corr={spearman(r_s2):+.3f}")
        if sel:
            print(f"  ✅ A1-1 STATISTIC = {sel}")
        else:
            print("  ⛔ NEITHER statistic recovers the known ordering.\n"
                  "     A1 IS AN INVALID SCREEN — it licenses NO GPU, and that "
                  "is the recorded result.")
        out["calibration"] = {
            "known_skill_order": r_sk, "S1_order": r_s1, "S2_order": r_s2,
            "S1_qualifies": q1, "S2_qualifies": q2,
            "S1_rank_corr_with_skill": spearman(r_s1),
            "S2_rank_corr_with_skill": spearman(r_s2),
            "selected_statistic": sel,
            "verdict": "VALID" if sel else "INVALID_SCREEN",
        }

    # ---- candidate screening (paired vs the reference) ---------------------
    cands = [c for c in a.candidates.split(",") if c and c in per]
    if cands:
        stat = (out.get("calibration", {}).get("selected_statistic")
                or "S1_cv_rr")
        lower = (stat == "S1_cv_rr")
        print(f"\n--- PAIRED vs {REF} on {stat} ---")
        out["paired"] = {}
        for c in cands:
            pr = paired(per[c], per[REF], stat, lower)
            out["paired"][c] = pr
            if pr:
                verdict = ("PASS A1-1" if pr["paired_median_gain"] > 0
                           and pr["win_share"] >= 0.55 else "FAIL A1-1")
                if pr["n"] < MIN_BASINS:
                    verdict = "INCONCLUSIVE (n<500)"
                print(f"  {c:<42} n={pr['n']:3d}  gain="
                      f"{pr['paired_median_gain']:+.4f}  "
                      f"wins={pr['win_share']:.1%}  {verdict}")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
