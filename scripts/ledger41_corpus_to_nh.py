#!/usr/bin/env python3
"""Ledger 41: convert the pretraining corpus to a neuralhydrology data_dir.

Deliberately imports its conventions from `scripts/corpus_to_nh.py` rather than
restating them. The experiment only means anything if the prior and the
finetune target occupy the same input space, and the cheapest way to guarantee
that is to run the same code: the same FORCING_MAP, the same q_cfs -> q_mm area
normalisation, the same target name, the same static column order. A parallel
implementation that agrees today is a domain shift waiting to happen.

Differences from the CAMELS path, both structural:
  * basins come from data/ledger41_basins.json, not the CAMELS 531;
  * statics come from data/ledger41_attrs.json (Probe C), not camels_attrs.json;
  * a basin-level train/held-out split is written, because the Stage-2
    own-domain gate is evaluated on basins the pretrain never saw.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_to_nh import (FORCING_MAP, DYNAMIC_INPUTS, STATIC_ATTRS, TARGET,
                          basin_to_nc)

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/local_corpora/gages2_hysets_v1")
    ap.add_argument("--attrs", default="data/ledger41_attrs.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--subset", default="", help="optional basin-list json to "
                    "restrict to (e.g. the pre-registered q90 fallback pool)")
    args = ap.parse_args()

    corpus, out = Path(args.corpus_dir), Path(args.out)
    attrs = {k.zfill(8): v for k, v in json.load(open(args.attrs)).items()}
    pool = {b["gauge_id"]: b for b in json.load(open(ROOT / "data" / "ledger41_basins.json"))["basins"]}
    keep = None
    if args.subset:
        keep = {str(x).zfill(8) for x in json.load(open(args.subset))}

    files = sorted(p for p in corpus.glob("*.csv.gz") if not p.name.startswith("._"))
    written, attr_rows = [], []
    for i, p in enumerate(files, 1):
        bid = p.name.split(".")[0].zfill(8)
        if bid not in attrs or bid not in pool:
            continue
        if keep is not None and bid not in keep:
            continue
        if basin_to_nc(p, out / "time_series" / f"{bid}.nc", pool[bid]["area_km2"]):
            written.append(bid)
            a = attrs[bid]
            attr_rows.append({"gauge_id": bid,
                              **{k: float(a[k]) if a.get(k) is not None
                                 and np.isfinite(a[k]) else np.nan
                                 for k in STATIC_ATTRS}})
        if i % 300 == 0:
            print(f"[{i}/{len(files)}] wrote {len(written)}", flush=True)

    adf = pd.DataFrame(attr_rows).set_index("gauge_id")
    n_nan = int(adf.isna().to_numpy().sum())
    adf = adf.fillna(adf.mean(numeric_only=True))
    (out / "attributes").mkdir(parents=True, exist_ok=True)
    adf.to_csv(out / "attributes" / "attributes.csv")

    # Basin split for the Stage-2 own-domain gate. Held-out basins are excluded
    # from pretraining entirely, so G2a measures generalisation to unseen
    # catchments rather than to unseen dates.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(written))
    n_hold = int(round(args.holdout_frac * len(written)))
    hold = sorted(written[i] for i in idx[:n_hold])
    train = sorted(written[i] for i in idx[n_hold:])
    (out / "basins_all.txt").write_text("\n".join(sorted(written)) + "\n")
    (out / "basins_train.txt").write_text("\n".join(train) + "\n")
    (out / "basins_holdout.txt").write_text("\n".join(hold) + "\n")

    print(f"\nDONE: {len(written)} basins -> {out}")
    print(f"  train {len(train)} | held-out {len(hold)} (seed {args.seed})")
    print(f"  attributes.csv: {len(STATIC_ATTRS)} attrs, {n_nan} NaN cells imputed")
    print(f"  dynamic_inputs: {DYNAMIC_INPUTS}  target: {TARGET}")
    assert list(adf.columns) == STATIC_ATTRS, (
        "static column order differs from the CAMELS path")


if __name__ == "__main__":
    main()
