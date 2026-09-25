#!/usr/bin/env python3
"""LEDGER 50 — the final-number scorer.

⚠️ v1 of this file reimplemented the stream globs and got 0.8365215 (+0.000232 off the
record) because it missed `gate_eval._lstm_seeds`'s rule that the retrained nldas member
(`_s111_NEW` / `_TRAIN_s1111`) DISPLACES the original s111 on both sides. It therefore
averaged 4 nldas seeds where the record averages 3. The self-test caught it. v2 builds the
frozen 9 through the SAME code path as every other ledger-44/45 read
(`ledger44_gate_transfer.build_streams` → `merge`, which call gate_eval's seed selectors),
so a stream can never be defined differently here than in the gate.

Live globs, so new seeds are picked up automatically — run --selftest FIRST on an unchanged
dumps dir: it must print the record to 1e-9, otherwise something upstream has changed.

⛔ Never overwrites analysis/noq_test_result.json.

  --rebuild NAME:shift:PATTERN:s1,s2,...   restrict/replace a stream's seeds ({t},{s})
  --add     NAME:shift:PATTERN:s1,s2,...   add a stream (e.g. multiL) as an extra member
  --swap    OLD=NEW                        drop OLD after adding NEW
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
from phase_lever import load_allh                                      # noqa: E402
from ledger44_gate_transfer import (build_streams, merge, per_basin_nse,  # noqa: E402
                                    invmse_weights, TEST_END, THETA, LAM)
from ledger45_weightest import boot_ci                                 # noqa: E402

RECORD = 0.8362893021622821
META = ("station_id", "date", "h", "truth")


def apply_specs(F, train, specs, cols):
    t = "TRAIN_" if train else ""
    for spec in specs:
        nm, sh, pattern, seedlist = spec.split(":", 3)
        ps = [pattern.format(t=t, s=sd) for sd in seedlist.split(",")]
        for q in ps:
            if not Path(q).exists():
                raise SystemExit(f"{nm} [{'TRAIN' if train else 'TEST'}]: missing {q}")
        d = load_allh(ps, shift_days=int(sh))[list(META[:3]) + ["pred"]]
        m = F[list(META[:3])].merge(d, on=list(META[:3]), how="left")
        miss = int(m["pred"].isna().sum())
        if miss:
            raise SystemExit(f"{nm} [{'TRAIN' if train else 'TEST'}]: {miss:,} rows unmatched")
        F[nm] = m["pred"].to_numpy(float)
        if nm not in cols:
            cols.append(nm)
    return F, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rebuild", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--swap", default="")
    ap.add_argument("--out", default="analysis/noq_test_result_l50.json")
    a = ap.parse_args()

    TR = merge(build_streams(True)); TE = merge(build_streams(False))
    cols = [c for c in TR.columns if c not in META]
    fit = TR[TR.h == 1].reset_index(drop=True)
    sc = TE[(TE.h == 1) & (TE.date <= TEST_END)].reset_index(drop=True)
    w = invmse_weights(fit, cols)
    s0 = per_basin_nse(sc.assign(_e=(sc[cols].to_numpy(float) * w).sum(1)), "_e")
    print(f"frozen-9 live rebuild: {s0.median():.16f}  record {RECORD:.16f}  "
          f"diff {s0.median()-RECORD:+.2e}  {'OK' if abs(s0.median()-RECORD) < 1e-9 else 'MISMATCH'}")
    if a.selftest:
        return 0 if abs(s0.median() - RECORD) < 1e-9 else 1

    for specs in (a.rebuild, a.add):
        fit, cols = apply_specs(fit, True, specs, cols)
        sc, cols = apply_specs(sc, False, specs, cols)
    if a.swap:
        old, new = a.swap.split("=")
        if old not in cols or new not in cols:
            raise SystemExit(f"--swap {a.swap}: need both in {cols}")
        cols = [c for c in cols if c != old]
        print(f"swap: dropped {old}, kept {new}")

    w = invmse_weights(fit, cols)
    sc = sc.copy(); sc["_e"] = (sc[cols].to_numpy(float) * w).sum(1)
    s = per_basin_nse(sc, "_e")
    idx = s0.index.intersection(s.index)
    d = (s[idx] - s0[idx]).to_numpy()
    lo, hi = boot_ci(d)
    print("=" * 74)
    print(f"  basins {len(s)}  rows {len(sc):,}  window {sc.date.min().date()} -> {sc.date.max().date()}")
    for c, x in zip(cols, w):
        print(f"    {c:<14} w={x:.4f}   solo {per_basin_nse(sc, c).median():.4f}")
    print(f"  MEDIAN NSE   : {s.median():.16f}")
    print(f"  record       : {RECORD:.16f}   delta {s.median()-RECORD:+.6f}")
    print(f"  paired vs record config: {np.median(d):+.6f}  CI [{lo:+.6f},{hi:+.6f}]  "
          f"breadth {(d>0).mean():.3f}")
    ships = (np.median(d) > 0 and (lo > 0 or hi < 0) and abs(np.median(d)) > 0.00103
             and (d > 0).mean() >= 0.5)
    print(f"  => {'SHIPS' if ships else 'does NOT ship'} under the ledger-50 bar")
    print("=" * 74)
    Path(a.out).write_text(json.dumps(
        {"median_nse": float(s.median()), "record": RECORD,
         "delta_vs_record": float(s.median() - RECORD), "paired": float(np.median(d)),
         "boot95": [lo, hi], "breadth": float((d > 0).mean()), "ships": bool(ships),
         "basins": int(len(s)), "rows": int(len(sc)), "streams": cols,
         "weights": {c: float(x) for c, x in zip(cols, w)},
         "rebuild": a.rebuild, "add": a.add, "swap": a.swap,
         "theta": THETA, "lam": LAM,
         "per_basin_nse": {k: float(v) for k, v in s.items()}}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
