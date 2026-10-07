#!/usr/bin/env python3
"""MODERN-1 — backward elimination under the standing removal rule, mechanised.

The rule (PREREG_v2.md): drop a member only when dropping it GAINS with a CI excluding
zero, i.e. its LOO is negative and CI-separated from zero. Re-score after EVERY removal,
one member at a time.

⭐ Why one-at-a-time matters and a batch drop would be wrong: a member's measured value is
CONDITIONAL ON THE ROSTER. Measured here 2026-10-02: `lstm_fused3m` read LOO +0.000022 (CI
including zero) while its 0.9454-correlated near-duplicate was present, and +0.008233 once
that duplicate was removed -- a 370x change from removing a DIFFERENT member. A drop list
computed in one pass would have discarded a genuine contributor.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--frame", default="val1")
    ap.add_argument("--protocol", default="modern-v3-9yr")
    ap.add_argument("--seeds", default="501,502,503")
    ap.add_argument("--basin-list", default="data/m1_cohort.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    members = [x for x in a.members.split(",") if x]
    seeds = set(int(x) for x in a.seeds.split(",") if x)
    ids = set(str(x).zfill(8) for x in json.load(open(a.basin_list))["cohort"])

    # Load every member ONCE; the join is then a column selection, so each round is cheap.
    frames = {}
    for m in members:
        d = l51.load_member(m, a.frame, seeds=seeds, protocol=a.protocol)
        frames[m] = d
    base = frames[members[0]][["station_id", "t0", "truth"]]
    for m in members:
        base = base.merge(frames[m][["station_id", "t0", m]], on=["station_id", "t0"], how="inner")
    base = base[base.station_id.isin(ids)]
    print(f"\njoined {len(base):,} rows  {base.station_id.nunique()} basins  seeds={sorted(seeds)}")

    trail = []
    cur = list(members)
    while len(cur) > 2:
        ens = l51.per_basin(base, cur)
        loo = {}
        for m in cur:
            rest = [x for x in cur if x != m]
            sub = l51.per_basin(base, rest)
            d, lo, hi, br, n = l51.paired_ci(ens, sub)
            loo[m] = (d, lo, hi, br)
        med = float(ens.median())
        print(f"\n--- {len(cur)} members: mean {med:.6f} ---")
        for m, (d, lo, hi, br) in sorted(loo.items(), key=lambda x: -x[1][0]):
            flag = "  <- CI incl 0" if lo <= 0 <= hi else ("  <- DROPPABLE" if d < 0 else "")
            print(f"    {m:<16} {d:+.6f} CI[{lo:+.6f},{hi:+.6f}]{flag}")
        # droppable = LOO negative AND CI excludes zero
        cand = [(d, m) for m, (d, lo, hi, br) in loo.items() if d < 0 and hi < 0]
        if not cand:
            print("    no member qualifies for removal -- composition is final")
            break
        d, worst = min(cand)
        sub = l51.per_basin(base, [x for x in cur if x != worst])
        gd, glo, ghi, gbr, _ = l51.paired_ci(sub, ens)
        print(f"    => DROP {worst}: gain {gd:+.6f} CI[{glo:+.6f},{ghi:+.6f}] breadth {gbr:.3f}")
        trail.append({"dropped": worst, "from_n": len(cur), "median_before": med,
                      "gain": gd, "ci": [glo, ghi], "breadth": gbr})
        cur = [x for x in cur if x != worst]

    ens = l51.per_basin(base, cur)
    P = base[cur].to_numpy(float)
    trimmed = ((P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2)) if len(cur) >= 4 else None
    res = {"frame": a.frame, "protocol": a.protocol, "seeds": sorted(seeds),
           "start_members": members, "final_members": cur,
           "final_mean": float(ens.median()),
           "final_trimmed": (float(l51.per_basin(base.assign(_t=trimmed), ["_t"]).median())
                             if trimmed is not None else None),
           "removal_trail": trail, "basins": int(base.station_id.nunique()),
           "rows": int(len(base))}
    print("\n" + "=" * 76)
    print(f"FINAL ({len(cur)}): {', '.join(cur)}")
    print(f"  mean    {res['final_mean']:.6f}")
    if res["final_trimmed"] is not None:
        print(f"  trimmed {res['final_trimmed']:.6f}")
    print(f"  dropped {len(trail)}: {', '.join(t['dropped'] for t in trail) or '(none)'}")
    print("=" * 76)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
