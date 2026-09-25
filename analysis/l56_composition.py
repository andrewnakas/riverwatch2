#!/usr/bin/env python3
"""LEDGER 56 — composition + LOO for an arbitrary member list, ONE join.

Every composition is scored on identical rows (a 9-member inner join has fewer
rows than an 8-member one, so scoring them separately compares different row
sets). Reports the mean and trimmed-mean combiners, per-member LOO with paired
CI and breadth, and the pairwise error-correlation block structure.

Removal is decided by the pre-registered rule: a member is dropped only if its
LOO paired CI excludes zero on the NEGATIVE side. This is NOT an argmax over
subsets ([[an-argmax-gap-is-not-headroom]]).

usage: l56_composition.py --frame val1 --members a,b,c --seeds3 fused3h1d3 [--out X.json]
"""
import argparse, importlib.util, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)


def combine(df, cols, how):
    P = df[cols].to_numpy(float)
    if how == "mean" or len(cols) < 4:
        return P.mean(1)
    return (P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2)


def per_basin_pred(df, pred):
    out = {}; y = df.truth.to_numpy(float)
    for s, idx in df.groupby("station_id").indices.items():
        yy, pp = y[idx], pred[idx]; k = np.isfinite(yy) & np.isfinite(pp); yy, pp = yy[k], pp[k]
        if len(yy) < l51.MIN_ROWS or np.var(yy) < 1e-9:
            continue
        out[s] = 1 - np.mean((yy - pp) ** 2) / np.var(yy)
    return pd.Series(out)


def err_corr(m, a, b, idx):
    ea = m[a].to_numpy(float) - m.truth.to_numpy(float)
    eb = m[b].to_numpy(float) - m.truth.to_numpy(float)
    out = []
    for s, ii in idx.items():
        x, y = ea[ii], eb[ii]
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() < 20 or np.std(x[k]) < 1e-9 or np.std(y[k]) < 1e-9:
            continue
        out.append(np.corrcoef(x[k], y[k])[0, 1])
    return float(np.median(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--members", required=True)
    ap.add_argument("--seeds3", default="", help="members restricted to seeds 501-503")
    ap.add_argument("--baseline", default="", help="comma list scored as the reference composition")
    ap.add_argument("--corr", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    members = [x for x in a.members.split(",") if x]
    s3 = {x for x in a.seeds3.split(",") if x}
    seeds_of = {x: {501, 502, 503} for x in s3}
    base = [x for x in a.baseline.split(",") if x] or None

    m = None
    for name in members:
        d = l51.load_member(name, a.frame, seeds=seeds_of.get(name))
        m = d if m is None else m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    idx = m.groupby("station_id").indices
    print(f"\njoined {len(m):,} rows  {len(idx)} basins  members={len(members)}\n")
    res = {"frame": a.frame, "members": members, "seeds3": sorted(s3),
           "rows": int(len(m)), "basins": int(len(idx))}

    ens = {h: per_basin_pred(m, combine(m, members, h)) for h in ("mean", "trimmed")}
    res["ensemble"] = {h: float(v.median()) for h, v in ens.items()}
    print(f"ENSEMBLE({len(members)})  mean {ens['mean'].median():.6f}   trimmed {ens['trimmed'].median():.6f}")
    d, lo, hi, br, _ = l51.paired_ci(ens["trimmed"], ens["mean"])
    res["trimmed_vs_mean"] = {"delta": d, "ci95": [lo, hi], "breadth": br}
    print(f"  trimmed vs mean paired {d:+.6f} CI [{lo:+.6f},{hi:+.6f}] breadth {br:.3f}")
    if base:
        eb = per_basin_pred(m, combine(m, base, "mean"))
        res["baseline"] = {"members": base, "median": float(eb.median())}
        for h in ("mean", "trimmed"):
            d, lo, hi, br, _ = l51.paired_ci(ens[h], eb)
            res["baseline"][f"vs_{h}"] = {"delta": d, "ci95": [lo, hi], "breadth": br,
                                          "significant": bool(lo > 0)}
            print(f"  vs baseline({len(base)}) {eb.median():.6f}  [{h}] paired {d:+.6f} "
                  f"CI [{lo:+.6f},{hi:+.6f}] breadth {br:.3f}")

    res["solo"] = {}; res["loo"] = {}
    print("\nmember          solo        LOO paired        CI95                    breadth")
    for x in members:
        solo = float(per_basin_pred(m, m[x].to_numpy(float)).median())
        rest = [y for y in members if y != x]
        e2 = per_basin_pred(m, combine(m, rest, "mean"))
        d, lo, hi, br, _ = l51.paired_ci(ens["mean"], e2)
        res["solo"][x] = solo
        res["loo"][x] = {"delta": d, "ci95": [lo, hi], "breadth": br,
                         "drop_by_rule": bool(hi < 0)}
        flag = "  <-- DROP (CI below zero)" if hi < 0 else ""
        print(f"{x:15s} {solo:.6f}   {d:+.6f}   [{lo:+.6f},{hi:+.6f}]   {br:.3f}{flag}")

    if a.corr:
        res["err_corr"] = {}
        print("\nmedian per-basin error correlation:")
        print("               " + " ".join(f"{y[:9]:>9s}" for y in members))
        for x in members:
            row = {y: (1.0 if x == y else err_corr(m, x, y, idx)) for y in members}
            res["err_corr"][x] = row
            print(f"{x:14s} " + " ".join(f"{row[y]:9.4f}" for y in members))
        for x in members:
            others = [y for y in members if y != x]
            res["err_corr"][x]["_mean_vs_others"] = float(np.mean([res["err_corr"][x][y] for y in others]))
        print("\nmean corr vs the rest: " + "  ".join(
            f"{x}={res['err_corr'][x]['_mean_vs_others']:.4f}" for x in members))

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
