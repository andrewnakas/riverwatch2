#!/usr/bin/env python
"""MODERN-1: composition, LOO, error structure, combiner, and greedy admission.

Every statistic is computed on ONE join, so a composition and its LOO deltas are
never compared across different row sets ([[a-null-must-be-computed-on-the-same-rows]]).

Rules this enforces, all of them measured conclusions of this campaign rather than
preferences:
  * EQUAL WEIGHT, zero fitted combination parameters. Inverse-MSE, per-basin,
    cluster and drop-one weighting all lost to equal weight on held-out data.
  * PAIRED per-basin bootstrap for every delta. Difference-of-medians has
    disagreed with paired nine times and lost every time, because the median is a
    rank statistic ([[paired-median-vs-difference-of-medians]]).
  * DUPLICATE-MEMBER CONTROL: adding a copy of a member must score <= 0. If it
    does not, the "gain" is member count, not information.
  * GREEDY FORWARD SELECTION with a CI gate -- admit one member at a time in
    descending order of its own paired delta, re-scoring after each admission.
    NOT an argmax over subsets, which is a forking path.
  * The COMBINER is a 1-bit choice on the FINAL composition (mean vs trimmed);
    both are always reported.
"""
from __future__ import annotations
import argparse, importlib.util, itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)


def combine(P: np.ndarray, how: str) -> np.ndarray:
    if how == "mean":
        return P.mean(1)
    if how == "trimmed":                      # drop each row's max and min
        if P.shape[1] < 3:
            return P.mean(1)
        return (P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2)
    if how == "median":
        return np.median(P, 1)
    raise SystemExit(f"unknown combiner {how}")


def basin_nse(df, pred) -> dict:
    out = {}
    y_all = df.truth.to_numpy(float)
    for s, idx in df.groupby("station_id").indices.items():
        y, ph = y_all[idx], pred[idx]
        k = np.isfinite(y) & np.isfinite(ph)
        y, ph = y[k], ph[k]
        if len(y) < l51.MIN_ROWS or np.var(y) < 1e-9:
            continue
        out[s] = 1 - np.mean((y - ph) ** 2) / np.var(y)
    return out


def series(d: dict):
    import pandas as pd
    return pd.Series(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="val1")
    ap.add_argument("--members", required=True, help="comma-separated")
    ap.add_argument("--seeds", default="501,502,503")
    ap.add_argument("--protocol", default="modern-v3")
    ap.add_argument("--combiner", default="mean", choices=["mean", "trimmed", "median"])
    ap.add_argument("--greedy", action="store_true", help="greedy forward selection with a CI gate")
    ap.add_argument("--corr", action="store_true", help="pairwise per-basin error correlation")
    ap.add_argument("--dup-control", action="store_true")
    ap.add_argument("--against", default="",
                    help="comma-separated members forming a SECOND composition; the "
                         "two are scored on ONE join and reported as a paired delta. "
                         "This is what the --dec-q arm's falsifier requires: it is "
                         "judged on the COMPOSITION delta, not the solo lifts, because "
                         "a uniform new input can lift every member and still collapse "
                         "them toward each other.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    members = [m for m in a.members.split(",") if m]
    against = [x for x in a.against.split(",") if x]
    seeds = set(int(s) for s in a.seeds.split(","))
    # ONE join over BOTH compositions, so the two are compared on identical rows.
    # ⚠️ DEDUPLICATE: the two lists legitimately overlap when asking "is a superset
    # better than its subset?" -- the most natural width question there is. Passing
    # a duplicated name to join() produced duplicate columns and then
    # KeyError: "['gridmetm9', ...] not in index".
    _load = list(dict.fromkeys(members + against))
    m = l51.join(_load, a.frame, seeds=seeds, protocol=a.protocol)
    print(f"\njoined {len(m):,} rows  {m.station_id.nunique()} basins  "
          f"seeds={sorted(seeds)}  protocol={a.protocol}")
    P = m[members].to_numpy(float)
    res: dict = {"frame": a.frame, "protocol": a.protocol, "members": members,
                 "seeds": sorted(seeds), "rows": int(len(m)),
                 "basins": int(m.station_id.nunique())}

    # ---- solos --------------------------------------------------------------
    solos = {}
    print(f"\n{'member':16s} {'solo':>10s}")
    for i, mm in enumerate(members):
        solos[mm] = basin_nse(m, P[:, i])
        print(f"  {mm:14s} {np.median(list(solos[mm].values())):10.6f}")
    res["solo"] = {k: float(np.median(list(v.values()))) for k, v in solos.items()}

    # ---- composition, both combiners ---------------------------------------
    print(f"\n{'combiner':16s} {'median NSE':>12s}")
    comps = {}
    for how in ("mean", "trimmed", "median"):
        comps[how] = basin_nse(m, combine(P, how))
        print(f"  {how:14s} {np.median(list(comps[how].values())):12.6f}")
    res["composition"] = {k: float(np.median(list(v.values()))) for k, v in comps.items()}
    base = comps[a.combiner]
    print(f"\nselected combiner: {a.combiner} -> {np.median(list(base.values())):.6f}")

    # ---- leave-one-out ------------------------------------------------------
    print(f"\n{'drop':16s} {'paired delta':>13s} {'CI95':>28s} {'breadth':>8s}")
    loo = {}
    for i, mm in enumerate(members):
        if len(members) < 3:
            break
        keep = [j for j in range(len(members)) if j != i]
        red = basin_nse(m, combine(P[:, keep], a.combiner))
        common = sorted(set(base) & set(red))
        d, lo, hi, br, n = l51.paired_ci(series(base).loc[common], series(red).loc[common])
        loo[mm] = dict(delta=float(d), ci=[float(lo), float(hi)], breadth=float(br))
        flag = "" if (lo > 0 or hi < 0) else "   <- CI includes zero"
        print(f"  {mm:14s} {d:+13.6f} [{lo:+.6f},{hi:+.6f}] {br:8.3f}{flag}")
    res["loo"] = loo

    # ---- duplicate-member control ------------------------------------------
    if a.dup_control:
        print("\nduplicate-member control (must be <= 0):")
        dup = {}
        for i, mm in enumerate(members):
            Pd = np.concatenate([P, P[:, [i]]], axis=1)
            d2 = basin_nse(m, combine(Pd, a.combiner))
            common = sorted(set(base) & set(d2))
            d, lo, hi, br, n = l51.paired_ci(series(d2).loc[common], series(base).loc[common])
            bad = " ⚠️ POSITIVE - the gain is member count, not information" if lo > 0 else ""
            print(f"  +{mm:13s} {d:+13.6f} [{lo:+.6f},{hi:+.6f}]{bad}")
            dup[mm] = float(d)
        res["dup_control"] = dup

    # ---- error structure ----------------------------------------------------
    if a.corr:
        print("\nmean pairwise per-basin error correlation:")
        y = m.truth.to_numpy(float)
        E = P - y[:, None]
        cors = {}
        for i, j in itertools.combinations(range(len(members)), 2):
            per = []
            for _, idx in m.groupby("station_id").indices.items():
                ei, ej = E[idx, i], E[idx, j]
                k = np.isfinite(ei) & np.isfinite(ej)
                if k.sum() < l51.MIN_ROWS or ei[k].std() == 0 or ej[k].std() == 0:
                    continue
                per.append(np.corrcoef(ei[k], ej[k])[0, 1])
            if per:
                cors[f"{members[i]}|{members[j]}"] = float(np.median(per))
        for k, v in sorted(cors.items(), key=lambda kv: -kv[1]):
            mark = "  ⚠️ >0.93" if v > 0.93 else ""
            print(f"  {k:36s} {v:.4f}{mark}")
        res["pairwise_err_corr"] = cors
        if cors:
            res["mean_err_corr"] = float(np.mean(list(cors.values())))
            print(f"  {'MEAN':36s} {res['mean_err_corr']:.4f}")

    # ---- greedy forward selection ------------------------------------------
    if a.greedy:
        print("\ngreedy forward selection (CI gate; not an argmax over subsets):")
        order = sorted(members, key=lambda x: -res["solo"][x])
        chosen = [order[0]]
        print(f"  seed member: {order[0]} (best solo {res['solo'][order[0]]:.6f})")
        for cand in order[1:]:
            idx_c = [members.index(x) for x in chosen]
            cur = basin_nse(m, combine(P[:, idx_c], a.combiner)) if len(chosen) > 1 \
                else solos[chosen[0]]
            trial = chosen + [cand]
            idx_t = [members.index(x) for x in trial]
            new = basin_nse(m, combine(P[:, idx_t], a.combiner))
            common = sorted(set(cur) & set(new))
            d, lo, hi, br, n = l51.paired_ci(series(new).loc[common], series(cur).loc[common])
            if lo > 0:
                chosen = trial
                print(f"  + {cand:14s} {d:+.6f} [{lo:+.6f},{hi:+.6f}] br {br:.3f}  ADMIT")
            else:
                print(f"  - {cand:14s} {d:+.6f} [{lo:+.6f},{hi:+.6f}] br {br:.3f}  reject (CI incl. 0)")
        idx_f = [members.index(x) for x in chosen]
        finals = {h: float(np.median(list(basin_nse(m, combine(P[:, idx_f], h)).values())))
                  for h in ("mean", "trimmed")}
        print(f"\n  GREEDY COMPOSITION ({len(chosen)}): {','.join(chosen)}")
        print(f"    mean {finals['mean']:.6f}   trimmed {finals['trimmed']:.6f}")
        res["greedy"] = {"chosen": chosen, "final": finals}

    # ---- composition-vs-composition, on the same join ----------------------
    if against:
        Q = m[against].to_numpy(float)
        other = basin_nse(m, combine(Q, a.combiner))
        common = sorted(set(base) & set(other))
        d, lo, hi, br, n = l51.paired_ci(series(base).loc[common], series(other).loc[common])
        print("\n" + "=" * 76)
        print(f"{'A: ' + ','.join(members):56s} {np.median(list(base.values())):.6f}")
        print(f"{'B: ' + ','.join(against):56s} {np.median(list(other.values())):.6f}")
        print(f"{'PAIRED delta (A - B)':56s} {d:+.6f}")
        print(f"{'   95% CI':56s} [{lo:+.6f}, {hi:+.6f}]")
        print(f"{'   breadth (basins favouring A)':56s} {br:.3f}")
        print(f"{'   basins':56s} {len(common)}")
        print("=" * 76)
        if lo <= 0 <= hi:
            print("VERDICT: NULL - the CI includes zero. Composition A is not better.")
        elif d > 0:
            print("VERDICT: A WINS, CI excludes zero.")
        else:
            print("VERDICT: A LOSES, CI excludes zero.")
        if "mean_err_corr" in res and res["mean_err_corr"] > 0.93:
            print(f"⚠️ mean pairwise error correlation {res['mean_err_corr']:.4f} > 0.93 — "
                  "per the registered rule the solo gains are presumed DILUTED.")
        res["comparison"] = {"A": members, "B": against,
                             "median_A": float(np.median(list(base.values()))),
                             "median_B": float(np.median(list(other.values()))),
                             "paired_delta": float(d), "ci95": [float(lo), float(hi)],
                             "breadth": float(br), "basins": len(common)}

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
