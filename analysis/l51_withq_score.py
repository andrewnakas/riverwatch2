#!/usr/bin/env python3
"""LEDGER 51 — score the CLEAN with-q ensemble on the Nearing 2022 protocol.

Reads the per-seed dumps written by `gpu1080/queue_l51_withq.sh` and REFUSES any
dump whose JSON sidecar does not record the guarded training window. The August
2026 with-q members were trained on their own test decade because a caller-side
flag was dropped when a script was copied; this scorer is the downstream half of
that fix — provenance is checked here, not assumed.

Seed averaging matches `app/mblstm.py`: average the physical quantile slots
FIRST, then sort, then clip at 0 (denorm is affine under --q-transform linear,
so averaging in physical space is identical to averaging in z-space).

Readout is the record's `(ylo+yhi)/2`, not `ymed` (worth +0.0026 on the old
frames — a readout difference that was once mistaken for a composition one).

usage:
  l51_withq_score.py --frame test14 --members daymet,maurer,nldas,aorc,fused3
  l51_withq_score.py --frame test1 --members ... --loo --bootstrap
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
DD = "data/mblstm/l51_dumps"
IDS = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
MIN_ROWS = 20


def load_member(member, frame, readout="mid", require_guard=True, seeds=None):
    """One member = the seed-average of its per-seed dumps. Returns a frame of
    (station_id, t0, truth, <member>)."""
    files = sorted(glob.glob(f"{DD}/camels531_l51_{member}_s*_{frame}.csv.gz"))
    if seeds is not None:
        files = [f for f in files if int(f.split("_s")[-1].split("_")[0]) in seeds]
    if not files:
        sys.exit(f"no dumps for member {member!r} frame {frame!r}")
    lo = md = hi = None
    truth = key = None
    used = []
    for f in files:
        side = f.replace(".csv.gz", ".json")
        if not os.path.exists(side):
            sys.exit(f"{f}: no sidecar — provenance unknown, refusing to score")
        s = json.load(open(side))
        if require_guard and (s.get("train_start") != "1999-10-01"
                              or s.get("train_end") != "2008-09-30"):
            sys.exit(f"{f}: sidecar says train {s.get('train_start')}..{s.get('train_end')}"
                     " — NOT the guarded window, refusing to score")
        d = pd.read_csv(f, dtype={"station_id": str})
        d["station_id"] = d.station_id.str.zfill(8)
        d = d[(d.h == 1) & d.station_id.isin(IDS)].sort_values(["station_id", "t0"])
        k = d.station_id + "|" + d.t0
        if key is None:
            key, truth = k.to_numpy(), d.truth.to_numpy(float)
            lo, md, hi = (np.zeros(len(d)) for _ in range(3))
        elif not np.array_equal(key, k.to_numpy()):
            sys.exit(f"{f}: window grid differs from the first seed of {member}")
        lo = lo + d.ylo.to_numpy(float)
        md = md + d.ymed.to_numpy(float)
        hi = hi + d.yhi.to_numpy(float)
        used.append(s["seed"])
    n = len(used)
    q = np.sort(np.stack([lo, md, hi], axis=1) / n, axis=1)   # average, THEN sort
    q = np.clip(q, 0.0, None)                                  # then clip
    pred = (q[:, 0] + q[:, 2]) / 2 if readout == "mid" else q[:, 1]
    sid = np.array([k.split("|")[0] for k in key])
    t0 = np.array([k.split("|")[1] for k in key])
    print(f"  {member:10s} {frame:7s} seeds={sorted(used)} rows={len(pred):,} "
          f"basins={len(set(sid))}")
    return pd.DataFrame({"station_id": sid, "t0": t0, "truth": truth, member: pred})


def join(members, frame, **kw):
    m = None
    for name in members:
        d = load_member(name, frame, **kw)
        if m is None:
            m = d
        else:
            t = m[["station_id", "t0", "truth"]].merge(
                d[["station_id", "t0", "truth"]], on=["station_id", "t0"],
                suffixes=("_a", "_b"))
            # Scale the check to the BASIN'S OWN FLOW, which is what NSE
            # normalises by. Truth is reconstructed per member as ys*sd+mu in
            # float32; recovering a 1.1 cfs day from a basin mean of ~100 loses
            # ~1e-4 absolute to cancellation, so a plain relative bar fails on a
            # low-flow row for a rounding difference. Verified against the
            # corpora: both hold q_cfs = 1.1 exactly for the worst row.
            t["_d"] = (t.truth_a - t.truth_b).abs()
            g = t.groupby("station_id")
            bad = float((g._d.max() / g.truth_a.std().clip(lower=1e-9)).max())
            assert bad < 1e-4, (f"truth disagrees between members by {bad:.3g} of the "
                                "basin's own flow sd — not a rounding difference")
            m = m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    return m


def per_basin(df, cols, weights=None):
    p = df[cols].to_numpy(float)
    w = np.ones(len(cols)) if weights is None else np.asarray([weights[c] for c in cols])
    e = p @ (w / w.sum())
    out = {}
    for s, idx in df.groupby("station_id").indices.items():
        y, ph = df.truth.to_numpy(float)[idx], e[idx]
        k = np.isfinite(y) & np.isfinite(ph)
        y, ph = y[k], ph[k]
        if len(y) < MIN_ROWS or np.var(y) < 1e-9:
            continue
        out[s] = 1 - np.mean((y - ph) ** 2) / np.var(y)
    return pd.Series(out)


def paired_ci(a, b, n=5000, seed=0):
    """Paired per-basin bootstrap of median(a) - median(b). The gate script that
    blessed 0.9253 bootstrapped a difference of INDEPENDENT medians and got a CI
    straddling zero; the paired test is the correct one for member deltas."""
    common = a.index.intersection(b.index)
    d = (a[common] - b[common]).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.median(d)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), \
        float((d > 0).mean()), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="test14")
    ap.add_argument("--members", default="daymet,maurer,nldas,aorc,fused3")
    ap.add_argument("--seeds", default="", help="comma list to restrict seed depth")
    ap.add_argument("--readout", choices=["mid", "ymed"], default="mid")
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--dup-control", action="store_true",
                    help="add a DUPLICATE of each member as an extra stream: must be <= 0")
    ap.add_argument("--weights", default="", help="json file of {member: weight}")
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-unguarded", action="store_true")
    a = ap.parse_args()

    members = [x for x in a.members.split(",") if x]
    seeds = set(int(x) for x in a.seeds.split(",") if x) or None
    print(f"LEDGER 51 — frame={a.frame} readout={a.readout}")
    m = join(members, a.frame, readout=a.readout,
             require_guard=not a.allow_unguarded, seeds=seeds)
    print(f"joined rows {len(m):,}  basins {m.station_id.nunique()}\n")

    W = json.load(open(a.weights)) if a.weights else None
    ens = per_basin(m, members, W)
    res = {"frame": a.frame, "readout": a.readout, "members": members,
           "seeds": sorted(seeds) if seeds else "all", "rows": int(len(m)),
           "basins": int(m.station_id.nunique()),
           "weights": W, "ensemble_median_nse": float(ens.median()),
           "n_scored": int(len(ens))}
    print(f"{'ENSEMBLE':<22} {ens.median():.6f}   (n={len(ens)})")
    for mem in members:
        s = per_basin(m, [mem])
        res.setdefault("member_median_nse", {})[mem] = float(s.median())
        print(f"  member {mem:<14} {s.median():.6f}")
    print(f"\nNearing 2022 AR = 0.879   delta {ens.median() - 0.879:+.6f}")

    if a.loo and len(members) > 1:
        print("\nLEAVE-ONE-OUT (loss = ensemble - ensemble_without)")
        res["loo"] = {}
        for mem in members:
            rest = [x for x in members if x != mem]
            sub = per_basin(m, rest, W)
            d, lo, hi, br, n = paired_ci(ens, sub)
            res["loo"][mem] = {"loss_median": float(ens.median() - sub.median()),
                               "paired_delta": d, "ci": [lo, hi], "breadth": br}
            print(f"  drop {mem:<12} {sub.median():.6f}  loss {ens.median()-sub.median():+.6f}"
                  f"   paired {d:+.6f} CI[{lo:+.6f},{hi:+.6f}] breadth {br:.3f}")

    if a.dup_control:
        print("\nDUPLICATE-MEMBER CONTROL (must be <= 0)")
        res["dup_control"] = {}
        for mem in members:
            mm = m.copy(); mm["_dup"] = mm[mem]
            sub = per_basin(mm, members + ["_dup"], None)
            res["dup_control"][mem] = float(sub.median() - ens.median())
            print(f"  +dup({mem:<12}) {sub.median():.6f}  delta {sub.median()-ens.median():+.6f}")

    if a.bootstrap:
        b = per_basin(m, members[:1])
        d, lo, hi, br, n = paired_ci(ens, b)
        res["vs_first_member"] = {"paired": d, "ci": [lo, hi], "breadth": br, "n": n}
        print(f"\nensemble vs {members[0]}: paired {d:+.6f} CI[{lo:+.6f},{hi:+.6f}] "
              f"breadth {br:.3f} n={n}")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
