#!/usr/bin/env python3
"""LEDGER 50 — held-out gate wrapper (pre-registered in PREREG_v2.md, LEDGER 50).

Reuses ledger 44/45 code paths verbatim: build_streams/merge (frozen 9, all leads),
invmse_weights (production rule, THETA=4 LAM=0.25, TRAIN rows only), per_basin_nse,
boot_ci (basin bootstrap of the paired median). The BASE frames are the cached
/tmp/l45_cache/base_{tr,te}.pkl from ledger 45 (rebuilt if absent).

Modes
  --anchor                     reproduce the record from the shipped 9 at h==1 (must be exact)
  --solo P [P ...] [--shift-days N]
                               solo h==1 held-out median of the seed-average of the given
                               dumps, and of each dump alone (B1 reads, seed sd)
  --dilution                   G0: synthetic δHBV pulled toward truth; bound on the ensemble
  --cand NAME --test P... --train P... [--shift-days N] --replaces STREAM
                               swap and add, both frames, ship-bar statistics
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.home() / "riverwatch2"))
from phase_lever import load_allh                                            # noqa: E402
from ledger44_gate_transfer import (build_streams, merge, per_basin_nse,     # noqa: E402
                                    invmse_weights, TEST_END)
from ledger45_weightest import boot_ci                                       # noqa: E402

RECORD = 0.8362893021622821
FLOOR = 0.00103
# FROZEN baseline frames: built BEFORE any ledger-50 seed landed, so the anchor keeps
# reproducing the record even as new seeds appear in gpu1080/dumps/. Never rebuild these.
FROZEN = Path.home() / "riverwatch2" / "frozen_base"
CACHE = Path("/tmp/l45_cache")
META = ("station_id", "date", "h", "truth")


def load_base():
    for d, tag in ((FROZEN, "FROZEN"), (CACHE, "cached")):
        ftr, fte = d / "base_tr.pkl", d / "base_te.pkl"
        if ftr.exists() and fte.exists():
            print(f"loading {tag} BASE frames from {d}...", file=sys.stderr)
            return pd.read_pickle(ftr), pd.read_pickle(fte)
    raise SystemExit(
        "FROZEN base frames are missing. Rebuilding them from live globs would fold in any\n"
        "seeds added since, and the anchor would no longer be the record. Restore\n"
        "~/riverwatch2/frozen_base/base_{tr,te}.pkl (MD5SUMS is beside them) before any read.")


def cols_of(df):
    return [c for c in df.columns if c not in META]


def frame(df, which, test=False):
    d = df[df.h == 1] if which == "h1" else df
    if test:
        d = d[d.date <= TEST_END]
    return d


def ens(fit, sc, cols):
    w = invmse_weights(fit, cols)
    sc = sc.copy()
    sc["_ens"] = (sc[cols].to_numpy(float) * w).sum(axis=1)
    return per_basin_nse(sc, "_ens"), w


def compare(base, cand, label):
    idx = base.index.intersection(cand.index)
    d = (cand[idx] - base[idx]).to_numpy()
    lo, hi = boot_ci(d)
    r = {"base_median": float(base[idx].median()), "median": float(cand[idx].median()),
         "paired": float(np.median(d)),
         "diff_of_medians": float(cand[idx].median() - base[idx].median()),
         "breadth": float((d > 0).mean()), "boot95": [lo, hi], "n": int(len(idx)),
         "ci_excludes_zero": bool(lo > 0 or hi < 0),
         "above_floor": bool(abs(np.median(d)) > FLOOR)}
    print(f"  {label:<40} med {r['median']:.6f}  paired {r['paired']:+.6f}  "
          f"dom {r['diff_of_medians']:+.6f}  CI [{lo:+.6f},{hi:+.6f}]  breadth {r['breadth']:.3f}  "
          f"{'CI-excl-0' if r['ci_excludes_zero'] else 'ci-spans-0'}")
    return r


def anchor(TR, TE, cols):
    b, w = ens(frame(TR, "h1"), frame(TE, "h1", True), cols)
    am = float(b.median())
    ok = abs(am - RECORD) < 1e-9
    print(f"ANCHOR shipped9 h==1 = {am:.16f}  record {RECORD:.16f}  diff {am-RECORD:+.2e}  "
          f"{'OK' if ok else 'MISMATCH'}")
    return am, ok, b


def both_frames(TR, TE, base_cols, cand_cols, tag, TRb=None, TEb=None):
    """Baseline is scored on (TRb, TEb) when given — the UNMODIFIED frames — so a
    synthetic/pulled candidate frame never contaminates its own baseline."""
    TRb = TR if TRb is None else TRb
    TEb = TE if TEb is None else TEb
    out = {}
    for fr in ("h1", "all"):
        fit, sc = frame(TR, fr), frame(TE, fr, True)
        b, _ = ens(frame(TRb, fr), frame(TEb, fr, True), base_cols)
        c, w = ens(fit, sc, cand_cols)
        out[fr] = compare(b, c, f"{tag} [{fr}]")
        out[fr]["weights"] = {k: float(v) for k, v in zip(cand_cols, w)}
    out["ships"] = bool(out["h1"]["paired"] > 0 and out["h1"]["ci_excludes_zero"]
                        and out["h1"]["above_floor"]
                        and np.sign(out["all"]["paired"]) == np.sign(out["h1"]["paired"])
                        and out["h1"]["breadth"] >= 0.5)
    print(f"  => {tag}: {'SHIPS' if out['ships'] else 'does not ship'}")
    return out


def pull(df, col, alpha):
    y = df["truth"].to_numpy(float); p = df[col].to_numpy(float)
    return np.where(np.isfinite(y), p + alpha * (y - p), p)


def solo_median(df, col):
    return float(per_basin_nse(df, col).median())


def calibrate_alpha(TE1, col, lift, iters=25):
    base = solo_median(TE1, col)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        t = TE1.copy(); t[col] = pull(t, col, mid)
        if solo_median(t, col) - base < lift:
            lo = mid
        else:
            hi = mid
    return hi, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--solo", nargs="+")
    ap.add_argument("--dilution", action="store_true")
    ap.add_argument("--dilute-members", nargs="+",
                    help="G0b: members to pull (default: the delta-HBV arms)")
    ap.add_argument("--lifts", nargs="+", type=float, default=[0.03, 0.06])
    ap.add_argument("--cand")
    ap.add_argument("--test", nargs="+")
    ap.add_argument("--train", nargs="+")
    ap.add_argument("--replaces", default="dhbv_daymet")
    ap.add_argument("--shift-days", type=int, default=0)
    ap.add_argument("--rebuild", action="append", default=[],
                    help="NAME:shift:PATTERN:s1,s2,... — recompute a frozen stream's column from "
                         "the given seeds in BOTH frames. PATTERN carries {t} (''/'TRAIN_') and "
                         "{s} (the seed), matching the campaign's dump convention, so each frame "
                         "gets its own dumps. Use to match seed depth when a "
                         "given seed dumps, in BOTH frames. Use to match seed depth when a "
                         "candidate has fewer seeds than the stream it replaces (an unmatched "
                         "swap charges the candidate for the incumbent's seed averaging).")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {"record": RECORD, "floor": FLOOR}

    if a.solo:
        TE1 = None
        ss = {}
        avg = load_allh(a.solo, shift_days=a.shift_days)
        avg = avg[(avg.h == 1) & (avg.date <= TEST_END) & (avg.date >= "1995-10-01")]
        ss["seed_avg"] = solo_median(avg, "pred")
        print(f"solo h==1 held-out, seed-average of {len(a.solo)}: {ss['seed_avg']:.6f}  rows {len(avg):,}  basins {avg.station_id.nunique()}")
        for p in a.solo:
            d = load_allh([p], shift_days=a.shift_days)
            d = d[(d.h == 1) & (d.date <= TEST_END) & (d.date >= "1995-10-01")]
            ss[Path(p).name] = solo_median(d, "pred")
            print(f"  {Path(p).name:<55} {ss[Path(p).name]:.6f}")
        single = [v for k, v in ss.items() if k != "seed_avg"]
        if len(single) > 1:
            print(f"  single-seed mean {np.mean(single):.6f}  sd {np.std(single, ddof=1):.6f}  "
                  f"=> B1 bar (mean + 3 sd) {np.mean(single) + 3*np.std(single, ddof=1):.6f}")
        res["solo"] = ss
    else:
        TR, TE = load_base()
        cols = cols_of(TR)
        am, ok, _ = anchor(TR, TE, cols)
        res["anchor"] = {"median": am, "reproduces": ok}
        if not ok:
            print("⚠️ ANCHOR MISMATCH — stop.")
        if a.rebuild:
            res["rebuilt"] = {}
            for spec in a.rebuild:
                nm, sh, pattern, seedlist = spec.split(":", 3)
                seeds = seedlist.split(",")
                assert nm in cols, f"{nm} is not a frozen stream: {cols}"
                for F, tag, t in ((TR, "TRAIN", "TRAIN_"), (TE, "TEST", "")):
                    ps = [pattern.format(t=t, s=sd) for sd in seeds]
                    for q in ps:
                        if not Path(q).exists():
                            raise SystemExit(f"rebuild {nm} [{tag}]: missing {q}")
                    d = load_allh(ps, shift_days=int(sh))[list(META[:3]) + ["pred"]]
                    m = F[list(META[:3])].merge(d, on=list(META[:3]), how="left")
                    miss = int(m["pred"].isna().sum())
                    if miss:
                        raise SystemExit(f"rebuild {nm} [{tag}]: {miss:,} rows unmatched")
                    F[nm] = m["pred"].to_numpy(float)
                print(f"rebuilt {nm} from {len(seeds)} seeds: {seeds}")
                res["rebuilt"][nm] = seeds
            am2, _, _ = anchor(TR, TE, cols)
            print(f"  baseline WITH rebuilt streams = {am2:.16f} "
                  f"({am2-am:+.6f} vs the record; this is the matched-depth baseline, "
                  f"NOT a record claim)")
            res["anchor_rebuilt"] = {"median": am2, "delta_vs_record": am2 - am}
        if a.dilution:
            TE1 = frame(TE, "h1", True)
            res["dilution"] = {}
            groups = ([[m] for m in a.dilute_members] if a.dilute_members
                      else [["dhbv_daymet"], ["dhbv_daymet", "dhbv_nldas", "dhbv_maurer"]])
            for members in groups:
                for lift in a.lifts:
                    tag = f"{'+'.join(members)} @+{lift:.4f}"   # 4dp: 0.005 and 0.01 collide at 2dp
                    TR2, TE2 = TR.copy(), TE.copy()
                    alphas = {}
                    for m in members:
                        alpha, b0 = calibrate_alpha(TE1, m, lift)
                        alphas[m] = alpha
                        TR2[m] = pull(TR2, m, alpha); TE2[m] = pull(TE2, m, alpha)
                        print(f"  {m}: solo h1 {b0:.4f} -> +{lift:.2f} at alpha={alpha:.4f}")
                    r = both_frames(TR2, TE2, cols, cols, tag, TRb=TR, TEb=TE)
                    r["alphas"] = alphas
                    res["dilution"][tag] = r
        if a.cand:
            for side, paths in (("TRAIN", a.train), ("TEST", a.test)):
                for p in paths:
                    assert Path(p).exists(), p
            ctr = load_allh(a.train, shift_days=a.shift_days)[list(META[:3]) + ["pred"]].rename(columns={"pred": a.cand})
            cte = load_allh(a.test, shift_days=a.shift_days)[list(META[:3]) + ["pred"]].rename(columns={"pred": a.cand})
            n0, m0 = len(TR), len(TE)
            TR2 = TR.merge(ctr, on=list(META[:3]), how="inner")
            TE2 = TE.merge(cte, on=list(META[:3]), how="inner")
            print(f"join: TRAIN {n0:,}->{len(TR2):,}  TEST {m0:,}->{len(TE2):,}")
            if len(TR2) < 0.95 * n0 or len(TE2) < 0.95 * m0:
                raise SystemExit("candidate join lost >5% of rows")
            # anchor must still reproduce on the joined frame (BASE frame integrity)
            am2, ok2, _ = anchor(TR2, TE2, cols)
            res["anchor_on_joined"] = {"median": am2, "reproduces": ok2}
            res["swap"] = both_frames(TR2, TE2, cols, [c for c in cols if c != a.replaces] + [a.cand],
                                      f"SWAP {a.cand} for {a.replaces}")
            res["add"] = both_frames(TR2, TE2, cols, cols + [a.cand], f"ADD {a.cand} as 10th")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
