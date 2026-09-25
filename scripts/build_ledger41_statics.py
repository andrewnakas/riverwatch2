#!/usr/bin/env python3
"""Ledger 41: produce the 27 production statics for the pretraining pool.

Two routes, assigned per attribute by which one better reproduces the published
CAMELS values in a head-to-head on the same basins (Probe C,
benchmarks/ledger41_probeC_statics.json):

  COMPUTED from the basin's own daily forcings -- the 8 climate indices that
  reproduce CAMELS at r = 1.000 with the conventions Probe C recovered
  (frac_snow threshold +1.0 C, Priestley-Taylor PET from shortwave radiation).

  IMPUTED by a random forest fit on the 671 CAMELS basins from GAGES-II
  (Falcone 2011) features -- the 18 landscape attributes (median CV R2 0.871)
  plus pet_mean, which is the one climate index the formulas do not recover
  (CAMELS calibrated Priestley-Taylor per catchment; computed r 0.836 vs
  imputed 0.863, so imputation wins for that one attribute only).

The regressors are fit on CAMELS and applied to non-CAMELS basins. That is a
train-side fit which never touches the scored window, but it does mean the
pretrain statics are partly a function of CAMELS statics -- stated in the
prereg's false-positive list rather than left implicit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from impute_camels_statics import load_features, addor_climate, CLIMATE

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "ledger41_basins.json"
CORPUS = ROOT / "data" / "local_corpora" / "gages2_hysets_v1"
CAMELS_ATTRS = ROOT / "data" / "camels_attrs.json"
OUT = ROOT / "data" / "ledger41_attrs.json"

# pet_mean is the single climate index where imputation beat computation.
COMPUTED = [c for c in CLIMATE if c != "pet_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    pool = {b["gauge_id"]: b for b in json.load(open(POOL))["basins"]}
    camels = {k.zfill(8): v for k, v in json.load(open(CAMELS_ATTRS)).items()}
    corpus = Path(args.corpus)
    built = sorted(p.name.split(".")[0].zfill(8) for p in corpus.glob("*.csv.gz")
                   if not p.name.startswith("._"))
    built = [b for b in built if b in pool]
    print(f"pool basins with a built corpus: {len(built)}")

    # ---- computed climate indices, from each basin's own forcing series
    print("computing climate indices from the forcings...")
    computed = {}
    for i, gid in enumerate(built, 1):
        df = pd.read_csv(corpus / f"{gid}.csv.gz", parse_dates=["date"]).set_index("date")
        df.attrs["lat"] = pool[gid]["lat"]
        try:
            computed[gid] = addor_climate(df)
        except Exception as e:
            print(f"  {gid}: climate failed ({type(e).__name__}: {e})")
        if i % 400 == 0:
            print(f"  [{i}/{len(built)}]", flush=True)
    print(f"  computed for {len(computed)} basins")

    # ---- imputed attributes: fit on CAMELS, predict on the pool
    impute_targets = [a for a in sorted(camels[next(iter(camels))]) if a not in COMPUTED]
    fit_ids = sorted(camels)
    Xfit = load_features(fit_ids)
    Xnew = load_features(built)
    Xnew = Xnew.reindex(columns=Xfit.columns, fill_value=np.nan)

    print(f"imputing {len(impute_targets)} attributes from GAGES-II features...")
    imputed = {g: {} for g in built}
    for t in impute_targets:
        y = pd.Series([camels[g].get(t) for g in fit_ids], index=fit_ids, dtype="float64")
        ok = y.notna()
        if ok.sum() < 50:
            print(f"  {t}: only {int(ok.sum())} reference values, filling with CAMELS mean")
            for g in built:
                imputed[g][t] = float(y[ok].mean()) if ok.any() else np.nan
            continue
        m = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                random_state=0, n_jobs=-1))
        m.fit(Xfit[ok.values], y[ok])
        pred = m.predict(Xnew)
        for g, v in zip(built, pred):
            imputed[g][t] = float(v)

    # area_gages2 is not a prediction problem: HYSETS and GAGES-II both measured it
    for g in built:
        imputed[g]["area_gages2"] = float(pool[g]["area_km2"])

    out, dropped = {}, []
    for g in built:
        if g not in computed:
            dropped.append(g)
            continue
        rec = {**imputed[g], **{k: float(v) for k, v in computed[g].items()
                                if k in COMPUTED}}
        if not all(np.isfinite(v) for v in rec.values()):
            dropped.append(g)
            continue
        out[g] = rec

    ref_keys = set(camels[next(iter(camels))])
    assert set(next(iter(out.values()))) == ref_keys, (
        "static schema differs from CAMELS -- the finetune interface would not match")
    print(f"\nwrote statics for {len(out)} basins ({len(dropped)} dropped)")

    # sanity: pool distributions should sit inside the CAMELS range, not beside it
    print(f"\n{'attribute':26s} {'pool median':>12s} {'CAMELS median':>14s} {'route':>9s}")
    for a in sorted(ref_keys):
        pv = np.median([r[a] for r in out.values()])
        cv = np.nanmedian([v.get(a) for v in camels.values()
                           if v.get(a) is not None])
        route = "computed" if a in COMPUTED else "imputed"
        print(f"{a:26s} {pv:12.3f} {cv:14.3f} {route:>9s}")

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
