#!/usr/bin/env python3
"""Ledger 41, gates G2a and G2b: is the prior worth finetuning from?

Both gates read scored artifacts, never a loss curve.

  G2a  own-domain median NSE on the 191 pool basins held out of pretraining --
       a SPATIAL holdout, so it measures generalisation to unseen catchments,
       which is the property a prior needs. Bar 0.60. Ledger 40's prior reached
       0.4804 on its own domain and was doomed before it ever touched CAMELS.

  G2b  zero-shot median NSE on CAMELS, train-side frame, from the 1-epoch
       lr=0.0 run. Bar 0.45. This is a confound detector rather than a success
       bar: ledger 40's prior scored -7.63 here, i.e. far worse than predicting
       the mean, which is what made its finetune a rescue mission instead of a
       measurement.

Reads NH's own pickled results so the NSE is computed by the same code path the
production numbers use.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def median_nse_from_results(res_path, target="q_mm"):
    with open(res_path, "rb") as f:
        res = pickle.load(f)
    if not res:
        sys.exit(f"{res_path} unpickled to an EMPTY dict -- with "
                 "save_validation_results unset NH still writes the file. "
                 "Set it True and re-evaluate.")
    nses, skipped = {}, 0
    for basin, freqs in res.items():
        try:
            xr_ = freqs["1D"]["xr"]
            obs = np.asarray(xr_[f"{target}_obs"]).squeeze()
            sim = np.asarray(xr_[f"{target}_sim"]).squeeze()
        except (KeyError, TypeError):
            skipped += 1
            continue
        ok = np.isfinite(obs) & np.isfinite(sim)
        obs, sim = obs[ok], sim[ok]
        if len(obs) < 20 or np.var(obs) < 1e-9:
            skipped += 1
            continue
        nses[str(basin)] = 1.0 - np.mean((obs - sim) ** 2) / np.var(obs)
    return pd.Series(nses), skipped


def load_id_list(path):
    """Basin ids from a json list, a pool json, or a plain text list."""
    txt = Path(path).read_text()
    if not txt.lstrip().startswith(("[", "{")):
        return {l.strip().zfill(8) for l in txt.split() if l.strip()}
    obj = json.loads(txt)
    ids = set()
    if isinstance(obj, list):
        ids = {str(x).zfill(8) for x in obj}
    elif isinstance(obj, dict):
        # a pool file ({"basins": [{gauge_id: ...}, ...]}) or a registry keyed by
        # cohort name ({"531": [...], "671": [...]}) -- both are used here
        for b in obj.get("basins", []):
            if isinstance(b, dict) and "gauge_id" in b:
                ids.add(str(b["gauge_id"]).zfill(8))
        for k, v in obj.items():
            if k.startswith("_") or k == "basins":
                continue
            if isinstance(v, list):
                ids |= {str(x).zfill(8) for x in v if not isinstance(x, (dict, list))}
    if not ids:
        raise SystemExit(f"{path}: parsed to ZERO basin ids -- an empty subset "
                         "silently scores nothing, so this is an error not a filter")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="NH *_results.p")
    ap.add_argument("--gate", choices=["G2a", "G2b"], required=True)
    ap.add_argument("--bar", type=float, default=None)
    ap.add_argument("--subset", default="",
                    help="restrict scoring to a basin-id list. The G2a failure "
                         "branch calls for scoring the GAGES-II REFERENCE "
                         "subset on its own -- if reference-only clears the bar "
                         "while the full pool does not, the pool is too broad "
                         "and gets rebuilt reference-heavy rather than the axis "
                         "being closed. Accepts a json list, a json object with "
                         "a 'basins' list of records carrying gauge_id, or a "
                         "newline-delimited text file.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    bar = args.bar if args.bar is not None else (0.60 if args.gate == "G2a" else 0.45)

    nse, skipped = median_nse_from_results(args.results)
    if args.subset:
        keep = load_id_list(args.subset)
        before = len(nse)
        nse = nse[[b for b in nse.index if str(b).zfill(8) in keep]]
        print(f"subset {Path(args.subset).name}: {len(keep)} ids -> "
              f"{len(nse)} of {before} scored basins retained")
        if len(nse) < 20:
            sys.exit("subset leaves too few basins to score")
    if len(nse) == 0:
        sys.exit("no scorable basins in the results file")
    med = float(nse.median())
    print(f"{args.gate}: {len(nse)} basins scored ({skipped} skipped)")
    print(f"  median NSE {med:.6f}   mean {nse.mean():.6f}")
    print(f"  quartiles  {nse.quantile(.25):.4f} / {nse.quantile(.5):.4f} / "
          f"{nse.quantile(.75):.4f}")
    print(f"  frac > 0.5 {float((nse > 0.5).mean()):.3f}   "
          f"frac > 0   {float((nse > 0).mean()):.3f}")
    ok = med >= bar
    print(f"\n  bar {bar:.2f} -> {'PASS' if ok else 'FAIL'}")
    if args.gate == "G2a" and not ok:
        print("  FAIL -> per the decision tree: score the GAGES-II reference "
              "subset separately. If reference-only clears 0.60, rebuild the "
              "pool reference-heavy and retrain ONCE. If it is below 0.55, "
              "audit lag/scaler/units; a clean audit closes the axis for this "
              "recipe.")
    if args.gate == "G2b" and not ok:
        print("  FAIL -> if G2a passed, this is the normalisation rider: run "
              "the pre-registered P1b scaler-injection arm before spending any "
              "finetune GPU.")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "gate": args.gate, "results": args.results, "bar": bar,
            "n_basins": len(nse), "n_skipped": skipped, "median_nse": med,
            "mean_nse": float(nse.mean()),
            "frac_gt_0.5": float((nse > 0.5).mean()),
            "frac_gt_0": float((nse > 0).mean()),
            "verdict": "PASS" if ok else "FAIL"}, indent=1))
        print(f"-> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
