#!/usr/bin/env python3
"""LEDGER 50 — DIRECT per-stream seed-depth pricing on the held-out frame, and a
linearity check on the G0b lift map.

Why: AMENDMENT 2 predicted the depth lane's value by mapping a SOLO gain through G0b's
+0.03 pull. Both steps are extrapolations. This measures the thing itself: rebuild each
stream from k of its seeds, refit the production weights on TRAIN, score the record's
h==1 frame, fit ensemble(k) = a - b/k, and predict the gain from the seeds being bought.

⛔ BOUNDS/PLANNING ONLY, like G0/G0b: it reads held-out rows to price GPU spend before the
dumps exist. No candidate is selected on test performance and the ship bar is untouched.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
import gate_eval as G                                              # noqa: E402
from phase_lever import load_allh                                  # noqa: E402
from rescore9 import glob_seeds                                    # noqa: E402
from ledger44_gate_transfer import (build_streams, merge, per_basin_nse,  # noqa: E402
                                    invmse_weights, TEST_END)
from ledger45_weightest import boot_ci                             # noqa: E402

RECORD = 0.8362893021622821
CACHE = Path("/tmp/l45_cache")
META = ("station_id", "date", "h", "truth")
HD = Path("dhbv_dumps")
PATS = {"lstm_multi":  ("camels531ls_multi_nhlstm_{t}s*.csv.gz", G.L, 0),
        "lstm_multi5": ("camels531ls_multi5_nhlstm_{t}s*.csv.gz", G.L, 0),
        "lstm_multi6": ("camels531ls_multi6_nhlstm_{t}s*.csv.gz", G.L, 0),
        "lstm_daymet": ("camels531ls_daymet_nhlstm_{t}s*.csv.gz", G.L, 0),
        "lstm_nldas":  ("camels531ls_nldas_nhlstm_{t}s*.csv.gz",  G.L, 0),
        "lstm_maurer": ("camels531ls_maurer_nhlstm_{t}s*.csv.gz", G.L, 0),
        "dhbv_daymet": ("camels531ls_daymet_dhbv_ALLH_{t}s*.csv.gz", HD, 1)}
WEAK = {"lstm_multi": ("s333", "s444")}


def seed_paths(name, train):
    pat, base, shift = PATS[name]
    ps = glob_seeds(pat.format(t="TRAIN_" if train else ""), base)
    if name in WEAK:
        ps = [p for p in ps if not any(f"_{w}." in p.name for w in WEAK[name])]
    return ps, shift


def stream_from(paths, shift, frame_keys):
    d = load_allh(paths, shift_days=shift)[["station_id", "date", "h", "pred"]]
    return d.merge(frame_keys, on=["station_id", "date", "h"], how="right")["pred"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["lstm_multi6", "lstm_multi5", "lstm_multi"])
    ap.add_argument("--buy", type=int, default=3, help="seeds to be added, for the prediction")
    ap.add_argument("--out", default="benchmarks/ledger50_depth.json")
    a = ap.parse_args()
    TR = pd.read_pickle(CACHE / "base_tr.pkl"); TE = pd.read_pickle(CACHE / "base_te.pkl")
    cols = [c for c in TR.columns if c not in META]
    fit0 = TR[TR.h == 1].reset_index(drop=True)
    sc0 = TE[(TE.h == 1) & (TE.date <= TEST_END)].reset_index(drop=True)
    base = per_basin_nse(sc0.assign(_e=(sc0[cols].to_numpy(float) * invmse_weights(fit0, cols)).sum(1)), "_e")
    print(f"ANCHOR {base.median():.16f}  diff {base.median()-RECORD:+.2e}")
    ktr, kte = fit0[list(META[:3])], sc0[list(META[:3])]
    out = {"anchor": float(base.median()), "record": RECORD, "streams": {}}
    for name in a.streams:
        ptr, shift = seed_paths(name, True); pte, _ = seed_paths(name, False)
        ntr, nte = len(ptr), len(pte)
        k_max = min(ntr, nte)
        print(f"\n=== {name}: {ntr} TRAIN / {nte} TEST seeds")
        if ntr != nte:
            print("  ⚠️ seed-count mismatch; using the common count by name order")
        curve = []
        for k in range(1, k_max + 1):
            subs = list(itertools.combinations(range(k_max), k))
            if len(subs) > 6:
                subs = subs[:6]
            meds = []
            for sub in subs:
                f = fit0.copy(); s = sc0.copy()
                f[name] = stream_from([ptr[i] for i in sub], shift, ktr)
                s[name] = stream_from([pte[i] for i in sub], shift, kte)
                w = invmse_weights(f, cols)
                s["_e"] = (s[cols].to_numpy(float) * w).sum(1)
                meds.append(float(per_basin_nse(s, "_e").median()))
            curve.append({"k": k, "median": float(np.mean(meds)), "n_subsets": len(subs),
                          "spread": float(np.max(meds) - np.min(meds)) if len(meds) > 1 else 0.0})
            print(f"  k={k}  ensemble median {np.mean(meds):.6f}  "
                  f"(subsets {len(subs)}, spread {curve[-1]['spread']:.6f})")
        ks = np.array([c["k"] for c in curve], float)
        ys = np.array([c["median"] for c in curve])
        if len(ks) >= 2:
            A = np.stack([np.ones_like(ks), -1.0 / ks], 1)
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            aa, bb = float(coef[0]), float(coef[1])
            now, then = k_max, k_max + a.buy
            pred = (aa - bb / then) - (aa - bb / now)
            inf = aa - (aa - bb / now)
            print(f"  fit a={aa:.6f} b={bb:.6f}  ⇒ +{a.buy} seeds ({now}->{then}) = "
                  f"{pred:+.6f}   to infinity {inf:+.6f}")
            out["streams"][name] = {"curve": curve, "a": aa, "b": bb, "k_now": now,
                                    "gain_buying": pred, "gain_to_infinity": inf}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
