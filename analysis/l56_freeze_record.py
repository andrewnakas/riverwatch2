#!/usr/bin/env python3
"""LEDGER 56 — freeze the record for a composition that may ADD members, REPLACE
members, or both, at the pre-registered ship bar.

The ship-bar comparison is always **the new composition vs the INCUMBENT one**,
computed on ONE join so both are scored on identical rows. For a pure addition
that is "9 vs 8"; for a replacement it is "8-with-new vs 8-with-old". The frozen
record's own rule (equal-weight MEAN) is always reported alongside whichever
combiner val1 selected, so the headline can never be an artifact of the combiner.

  * ship-bar seed count enforced per member (5 unless --screen-ok),
  * val1 decides, test1 is read once and selects nothing,
  * every backing dump's sidecar re-checked for the guarded training window.

usage: l56_freeze_record.py --members <new composition> --incumbent <old composition> \
         --shipbar fused3h1d3:501,502,503,504,505 [--seeds3 x,y] [--combiner mean|trimmed] --out X.json
"""
import argparse, glob, hashlib, importlib.util, json, os, subprocess, sys
from datetime import datetime, timezone
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)
DD = "data/mblstm/l51_dumps"; NEARING = 0.879


def combine(df, cols, how):
    P = df[cols].to_numpy(float)
    if how == "mean" or len(cols) < 4:
        return P.mean(1)
    return (P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2)


def per_basin_pred(df, pred):
    out = {}; y = df.truth.to_numpy(float)
    for s, idx in df.groupby("station_id").indices.items():
        yy, pp = y[idx], pred[idx]
        k = np.isfinite(yy) & np.isfinite(pp); yy, pp = yy[k], pp[k]
        if len(yy) < l51.MIN_ROWS or np.var(yy) < 1e-9:
            continue
        out[s] = 1 - np.mean((yy - pp) ** 2) / np.var(yy)
    return pd.Series(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="the NEW composition")
    ap.add_argument("--incumbent", required=True, help="the composition being beaten")
    ap.add_argument("--shipbar", default="", help="member:seed,seed,... enforced at >=5 seeds")
    ap.add_argument("--seeds3", default="", help="members deliberately scored at 3 seeds (screens)")
    ap.add_argument("--combiner", choices=["mean", "trimmed"], default="mean")
    ap.add_argument("--screen-ok", action="store_true", help="allow a sub-ship-bar freeze (NOT a record)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    members = [x for x in a.members.split(",") if x]
    incumbent = [x for x in a.incumbent.split(",") if x]
    seeds_of = {}
    for tok in [t for t in a.shipbar.split(" ") if t]:
        m, ss = tok.split(":"); seeds_of[m] = {int(x) for x in ss.split(",")}
    for m in [t for t in a.seeds3.split(",") if t]:
        seeds_of.setdefault(m, {501, 502, 503})
    new_members = [m for m in members if m not in incumbent]

    for m, ss in seeds_of.items():
        if m in [t.split(":")[0] for t in a.shipbar.split(" ") if t]:
            if len(ss) < 5 and not a.screen_ok:
                sys.exit(f"SHIP BAR NOT MET: {m} at {len(ss)} seeds, need 5.")
            have = sorted(int(f.split("_s")[-1].split("_")[0])
                          for f in glob.glob(f"{DD}/camels531_l51_{m}_s*_test1.csv.gz"))
            missing = [s for s in sorted(ss) if s not in have]
            if missing:
                sys.exit(f"SHIP BAR NOT MET: {m} missing test1 dumps for seeds {missing} (have {have}).")

    res = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "composition": members, "incumbent_composition": incumbent,
           "added_members": new_members, "removed_members": [m for m in incumbent if m not in members],
           "seeds_used": {m: sorted(s) for m, s in seeds_of.items()}, "combiner": a.combiner,
           "screen_only": bool(a.screen_ok),
           "protocol": "nearing2022: train 1999-10-01..2008-09-30, val 1980-10-01..1989-09-30, "
                       "test 1989-10-01..1999-09-30, 531 basins, day-1, ALL daily observations, median NSE",
           "combination": f"equal weight ({a.combiner}), zero fitted parameters", "readout": "(ylo+yhi)/2",
           "reference_nearing_2022_single_model": NEARING}
    try:
        res["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        res["git_commit"] = None

    prov = {}
    for frame in ("val1", "test1"):                   # val1 FIRST; test1 selects nothing
        m = None
        for name in sorted(set(members) | set(incumbent)):
            d = l51.load_member(name, frame, seeds=seeds_of.get(name))
            m = d if m is None else m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
        ens = per_basin_pred(m, combine(m, members, a.combiner))
        ens_mean = per_basin_pred(m, combine(m, members, "mean"))
        inc = per_basin_pred(m, combine(m, incumbent, "mean"))      # the frozen record's own rule
        d, lo, hi, br, n = l51.paired_ci(ens, inc)
        d2, lo2, hi2, br2, _ = l51.paired_ci(ens_mean, inc)
        blk = {"rows": int(len(m)), "basins": int(m.station_id.nunique()),
               "ensemble_median_nse": float(ens.median()),
               "ensemble_median_nse_mean_combiner": float(ens_mean.median()),
               "incumbent_median_nse": float(inc.median()),
               "solo": {x: float(per_basin_pred(m, m[x].to_numpy(float)).median()) for x in members},
               "vs_incumbent": {"delta": d, "ci": [lo, hi], "breadth": br, "significant": bool(lo > 0)},
               "vs_incumbent_mean_combiner": {"delta": d2, "ci": [lo2, hi2], "breadth": br2,
                                              "significant": bool(lo2 > 0)},
               "loo_all": {}}
        for x in members:
            rest = [y for y in members if y != x]
            dd, l2, h2, b2, _ = l51.paired_ci(ens, per_basin_pred(m, combine(m, rest, a.combiner)))
            blk["loo_all"][x] = {"delta": dd, "ci": [l2, h2], "breadth": b2}
        res[frame] = blk
        print(f"[{frame}] {a.combiner} {ens.median():.6f} (mean {ens_mean.median():.6f})  "
              f"incumbent {inc.median():.6f}  paired {d:+.6f} CI[{lo:+.6f},{hi:+.6f}] breadth {br:.3f}")
        rows = []
        for x in members:
            for f in sorted(glob.glob(f"{DD}/camels531_l51_{x}_s*_{frame}.csv.gz")):
                s = int(f.split("_s")[-1].split("_")[0])
                if x in seeds_of and s not in seeds_of[x]:
                    continue
                side = json.load(open(f.replace(".csv.gz", ".json")))
                assert side["train_start"] == "1999-10-01" and side["train_end"] == "2008-09-30", side
                rows.append({"member": x, "seed": s, "rows": side["rows"], "ckpt_md5": side.get("ckpt_md5"),
                             "engine": side.get("engine", "mblstm"), "ledger": side.get("ledger", 51)})
        prov[frame] = rows

    v = res["val1"]["vs_incumbent"]
    res["ship_bar"] = {"val1_paired_positive": v["delta"] > 0, "val1_ci_excludes_zero": v["significant"],
                       "val1_breadth_ge_0.5": v["breadth"] >= 0.5,
                       "PASS": bool(v["delta"] > 0 and v["significant"] and v["breadth"] >= 0.5)}
    res["record_test1"] = res["test1"]["ensemble_median_nse"]
    res["record_test1_mean_combiner"] = res["test1"]["ensemble_median_nse_mean_combiner"]
    res["margin_vs_nearing"] = res["record_test1"] - NEARING
    res["reaches_0.90"] = bool(res["record_test1"] >= 0.90)
    res["provenance"] = prov
    json.dump(res, open(a.out, "w"), indent=1)
    h = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    print(f"\nSHIP BAR: {'PASS' if res['ship_bar']['PASS'] else 'FAIL'}   RECORD (test1) = "
          f"{res['record_test1']:.6f}   vs Nearing {NEARING} = {res['margin_vs_nearing']:+.6f}   "
          f"REACHES 0.90: {res['reaches_0.90']}")
    print(f"wrote {a.out}  md5 {h}  ({len(prov['test1'])} test1 dumps backing it)")
    return 0 if res["ship_bar"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
