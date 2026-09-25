#!/usr/bin/env python3
"""LEDGER 44 / S1b — THE HELD-OUT BREADTH CURVE.

Pre-registered in benchmarks/ledger44_branch_table.md AMENDMENT 1, before any
number here existed. Read that first; the ordering, the screen and what may be
claimed are all fixed there.

S1 bounded DEPTH (more seeds of the same 9 recipes) at +0.0016..+0.0023 held-out.
That asymptote belongs to this ensemble family. BREADTH -- more distinct recipes
-- moves the asymptote itself, and it has never been measured held-out. 20 extra
streams exist with matched seeds on both surfaces, all already trained, all
rejected on the in-sample surface. Cost: zero GPU.

⛔ NO INTERIOR POINT OF THE CURVE MAY BE SHIPPED. Only the take-all
configuration involves no selection. Choosing where to stop, having seen
held-out scores, is selection on test.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())

import gate_eval as G                                              # noqa: E402
from phase_c import load_seed_avg                                  # noqa: E402
from ledger44_ensembling_law import (fit_global_invmse, per_basin_nse,  # noqa: E402
                                     ANCHOR_TRAIN, ANCHOR_TEST)

CV_FIT_END, CV_VAL_START = G.CV_FIT_END, G.CV_VAL_START
SOLO_BAR = 0.90            # train-side competence band (AMENDMENT 1 step 2)
MULTI5_SEEDS = ("s111", "s222", "s333", "s444", "s555")
MULTI6_SEEDS = ("s111", "s222", "s333")
KEPT_ORDER = ["multih512", "multi9", "multi8", "multi5b", "multi730",
              "multi671", "multi_eps05", "multi10", "multidrop", "multih128",
              "aorc"]   # fixed by AMENDMENT 1's train-side solo screen
SHIPPED = ["lstm_daymet", "lstm_nldas", "lstm_maurer", "lstm_multi",
           "lstm_multi5", "lstm_multi6", "dhbv_daymet", "dhbv_nldas", "dhbv_maurer"]
# Seed-mismatched between surfaces -> excluded (AMENDMENT 1 step 1).
EXCLUDE_TAGS = {"multi5swa", "multi", "multi5", "multi6", "daymet", "nldas", "maurer"}


def pool_tags():
    """Extra streams with MATCHED seed sets on both surfaces."""
    tr, te = {}, {}
    for p in G.L.glob("camels531ls_*_nhlstm_*.csv.gz"):
        n = p.name
        if any(x in n for x in G._EXCLUDE_MARKERS) or "QUARANTINE" in n:
            continue
        m = re.match(r"camels531ls_(.+?)_nhlstm_(TRAIN_)?(s\w+)\.csv\.gz", n)
        if not m:
            continue
        tag, istr, seed = m.group(1), bool(m.group(2)), m.group(3)
        (tr if istr else te).setdefault(tag, set()).add(seed)
    out = []
    for t in sorted(set(tr) & set(te)):
        if t in EXCLUDE_TAGS:
            continue
        if tr[t] == te[t]:
            out.append((t, sorted(tr[t])))
    return out


def paths_for(name, tag_seeds, train):
    t = "TRAIN_" if train else ""
    if name in ("lstm_daymet", "lstm_nldas", "lstm_maurer"):
        return G._lstm_seeds(name.split("_")[1], train), 0
    if name == "lstm_multi":
        return G._multi_seeds(train), 0
    if name == "lstm_multi5":
        return [G.L / f"camels531ls_multi5_nhlstm_{t}{s}.csv.gz" for s in MULTI5_SEEDS], 0
    if name == "lstm_multi6":
        return [G.L / f"camels531ls_multi6_nhlstm_{t}{s}.csv.gz" for s in MULTI6_SEEDS], 0
    if name.startswith("dhbv_"):
        return G._dhbv_seeds(name.split("_")[1], train), 1
    tag = name.replace("pool_", "")
    return [G.L / f"camels531ls_{tag}_nhlstm_{t}{s}.csv.gz" for s in tag_seeds[tag]], 0


def load(names, tag_seeds, train):
    ids = set(str(x).zfill(8) for x in json.load(open(G.SUBSET))["531"])
    series, truth = {}, None
    for nm in names:
        ps, shift = paths_for(nm, tag_seeds, train)
        miss = [p for p in ps if not Path(p).exists()]
        if miss:
            raise SystemExit(f"{nm}: missing {miss}")
        d = load_seed_avg([Path(p) for p in ps], shift_days=shift)
        d = d[d.station_id.isin(ids)].set_index(["station_id", "date"])
        series[nm] = d["pred"]
        if truth is None:
            truth = d["truth"]
    idx = None
    for s in series.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.intersection(truth.index)
    mat = {nm: s.reindex(idx).to_numpy(float) for nm, s in series.items()}
    return (mat, np.array([b for b, _ in idx]),
            np.array([d for _, d in idx]), truth.reindex(idx).to_numpy(float))


def score(TR, TE, cols, surface):
    trm, trs, trd, trt = TR
    tem, tes, ted, tet = TE
    if surface == "train":
        fm, vm = trd <= CV_FIT_END, trd >= CV_VAL_START
        w = fit_global_invmse(np.stack([trm[c][fm] for c in cols]), trt[fm])
        ens = (np.stack([trm[c][vm] for c in cols]) * w[:, None]).sum(axis=0)
        return float(np.median(per_basin_nse(trs[vm], trt[vm], ens))), w
    w = fit_global_invmse(np.stack([trm[c] for c in cols]), trt)
    ens = (np.stack([tem[c] for c in cols]) * w[:, None]).sum(axis=0)
    return float(np.median(per_basin_nse(tes, tet, ens))), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/ledger44_breadth_curve.json")
    args = ap.parse_args()

    pool = pool_tags()
    tag_seeds = dict(pool)
    pool_names = [f"pool_{t}" for t, _ in pool]
    print(f"pool ({len(pool)} extra streams): {[t for t, _ in pool]}\n", file=sys.stderr)

    names = SHIPPED + pool_names
    cache = Path("benchmarks/.ledger44_breadth_cache.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        TR = (z["trm"].item(), z["trs"], z["trd"], z["trt"])
        TE = (z["tem"].item(), z["tes"], z["ted"], z["tet"])
        print(f"loaded cached surfaces from {cache}", file=sys.stderr)
    else:
        print("loading TRAIN...", file=sys.stderr)
        TR = load(names, tag_seeds, True)
        print("loading TEST...", file=sys.stderr)
        TE = load(names, tag_seeds, False)
        np.savez_compressed(cache, trm=TR[0], trs=TR[1], trd=TR[2], trt=TR[3],
                            tem=TE[0], tes=TE[1], ted=TE[2], tet=TE[3])
    print(f"  common frames: TRAIN {len(TR[2]):,} rows / {len(set(TR[1]))} basins, "
          f"TEST {len(TE[2]):,} rows / {len(set(TE[1]))} basins "
          f"({min(TE[2].tolist())}..{max(TE[2].tolist())})", file=sys.stderr)

    # ---- anchors on THIS frame (the pool's inner join may shrink the row set;
    #      the shipped-9 score on the reduced frame is the honest comparator)
    base_tr, _ = score(TR, TE, SHIPPED, "train")
    base_te, _ = score(TR, TE, SHIPPED, "test")
    print("=" * 78)
    print("ANCHORS ON THE POOL'S COMMON FRAME")
    print("=" * 78)
    print(f"  shipped 9, TRAIN  {base_tr:.6f}   (full-frame anchor {ANCHOR_TRAIN:.6f}, "
          f"diff {base_tr-ANCHOR_TRAIN:+.6f})")
    print(f"  shipped 9, TEST   {base_te:.6f}   (record {ANCHOR_TEST:.6f}, "
          f"diff {base_te-ANCHOR_TEST:+.6f})")

    # ---- train-side solo competence screen + ordering (AMENDMENT 1 steps 2-3)
    print("\n" + "=" * 78)
    print(f"TRAIN-SIDE SOLO SCREEN (bar {SOLO_BAR}) AND ORDERING — no test information")
    print("=" * 78)
    solos = []
    trm, trs, trd, trt = TR
    vm = trd >= CV_VAL_START
    for nm in pool_names:
        s = float(np.median(per_basin_nse(trs[vm], trt[vm], trm[nm][vm])))
        solos.append((nm, s))
    solos.sort(key=lambda x: -x[1])
    kept = [(n, s) for n, s in solos if s >= SOLO_BAR]
    for n, s in solos:
        print(f"  {n.replace('pool_',''):<14} solo(train val) {s:.6f}   "
              f"{'KEEP' if s >= SOLO_BAR else 'DROP (below band)'}")
    print(f"\n  kept {len(kept)} of {len(solos)}")

    # ---------------------------------------------------- the curve
    print("\n" + "=" * 78)
    print("THE BREADTH CURVE — adding in train-side solo order, rule refit each step")
    print("⛔ NO INTERIOR POINT IS SHIPPABLE. Only the take-all row is.")
    print("=" * 78)
    print(f"{'k':>3} {'added':<16} | {'TRAIN':>10} {'delta':>10} | {'TEST':>10} {'delta':>10}")
    print("-" * 70)
    cols = list(SHIPPED)
    curve = [{"k": len(cols), "added": "(shipped 9)", "train": base_tr,
              "test": base_te, "train_delta": 0.0, "test_delta": 0.0}]
    print(f"{len(cols):>3} {'(shipped 9)':<16} | {base_tr:>10.6f} {0.0:>+10.6f} | "
          f"{base_te:>10.6f} {0.0:>+10.6f}")
    for nm, _ in kept:
        cols.append(nm)
        tr_s, _ = score(TR, TE, cols, "train")
        te_s, _ = score(TR, TE, cols, "test")
        curve.append({"k": len(cols), "added": nm.replace("pool_", ""),
                      "train": tr_s, "test": te_s,
                      "train_delta": tr_s - base_tr, "test_delta": te_s - base_te})
        print(f"{len(cols):>3} {nm.replace('pool_',''):<16} | {tr_s:>10.6f} "
              f"{tr_s-base_tr:>+10.6f} | {te_s:>10.6f} {te_s-base_te:>+10.6f}")

    take_all = curve[-1]
    print("\n" + "=" * 78)
    print("PRIMARY READOUT — THE TAKE-ALL CONFIGURATION (no selection)")
    print("=" * 78)
    print(f"  streams                 {take_all['k']}")
    print(f"  held-out median NSE     {take_all['test']:.6f}")
    print(f"  vs shipped 9 (same frame) {take_all['test_delta']:+.6f}")
    print(f"  vs the record {ANCHOR_TEST:.6f}  {take_all['test']-ANCHOR_TEST:+.6f}")
    print(f"  in-sample would have said {take_all['train_delta']:+.6f}")
    if take_all["test_delta"] >= 0.003:
        v = "D-PASS (pending the all-leads frame)"
    elif take_all["test_delta"] > 0:
        v = "D-PARTIAL — reported as measured, NOT claimed as skill"
    else:
        v = "D-FAIL — breadth is exhausted held-out too"
    print(f"\n  => {v}")

    json.dump({"pool": [t for t, _ in pool], "solo_screen": dict(solos),
               "kept": [n for n, _ in kept], "anchors": {"train": base_tr, "test": base_te},
               "curve": curve, "take_all": take_all, "verdict": v},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
