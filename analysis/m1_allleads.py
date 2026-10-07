#!/usr/bin/env python3
"""MODERN-1 — the all-leads 1..14 table (the user's decision 1: day-1 median NSE as
the headline, PLUS a table over every lead).

Day-1 is a rank statistic over 490 basins on ONE lead; the table says whether the
ensemble's advantage is a lead-1 artefact or holds across the horizon. It costs no
extra training -- `queue_m1_withq.sh` already writes val14/test14 dumps with
`--dump-all-leads`.

⭐ BUILT-IN INTEGRITY CHECK (`--crosscheck`): the h==1 slice of the 14-lead dump must
reproduce the day-1 dump's median to ~1e-6. They come from different dump invocations
of the same checkpoint, so a mismatch means one of the two is wrong -- exactly the
class of silent defect that has cost this campaign the most (a stride subsample that
offset grids across corpora; a lead-weighting bug invisible to every mean/bias check).
Nothing here is trustworthy until that check passes.

⚠️ `--h1-weight 0.5` puts half the gradient on lead 1 because the benchmark scores
lead 1 and nothing else. So a DECLINE across leads is expected and is NOT evidence
of a defect; the table exists to quantify it honestly, not to flatter the horizon.

usage:
  m1_allleads.py --frame val14 --protocol modern-v3-9yr \
      --members gridmetm9,nclimgridm9,nhar0m --seeds 501,502,503 --crosscheck
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)
MIN_ROWS = 20
LEADS = list(range(1, 15))


def load_all_leads(member, frame, seeds=None, protocol="modern-v3-9yr", prefix=None):
    """Seed-average a member keeping EVERY lead. Mirrors l51.load_member exactly --
    average the physical quantile slots, THEN sort, THEN clip, readout (ylo+yhi)/2 --
    and shares its provenance guard via l51.check_sidecar so the two cannot drift."""
    _pref = l51.PROTOCOLS[protocol][2] if prefix is None else prefix
    files = sorted(glob.glob(f"{l51.DD}/{_pref}{member}_s*_{frame}.csv.gz"))
    if seeds is not None:
        files = [f for f in files if int(f.split("_s")[-1].split("_")[0]) in seeds]
    if not files:
        sys.exit(f"no {frame} dumps for member {member!r}")
    lo = md = hi = None
    key = truth = None
    used = []
    for f in files:
        s = l51.check_sidecar(f, protocol, True)
        if not s.get("all_leads"):
            sys.exit(f"{f}: sidecar says all_leads={s.get('all_leads')} — this is not a "
                     f"14-lead dump, refusing (scoring a 1-lead dump as 14 would "
                     f"silently report lead 1 fourteen times)")
        # ⚠️ MEMORY. The all-leads frame is 14x the day-1 frame: test14 is ~1.24M rows x 14
        # leads ~= 17M rows PER DUMP. The earlier version of this loader built a concatenated
        # "basin|t0|lead" string key and then np.array([x.split("|") for x in key]) -- 17M Python
        # lists of three strings, several GB per member. With four members that exhausted the
        # 15 GiB box while three trainings were running and took sshd down with it, so the job
        # could not even be stopped. Keep the three key columns as they arrive, in compact
        # dtypes, and never round-trip them through a joined string.
        d = pd.read_csv(f, dtype={"station_id": str},
                        usecols=["station_id", "t0", "h", "truth", "ylo", "ymed", "yhi"])
        d["station_id"] = d.station_id.str.zfill(8)
        d = d.sort_values(["station_id", "t0", "h"], kind="stable")
        k_sid = d.station_id.to_numpy()
        k_t0 = d.t0.to_numpy()
        k_h = d.h.to_numpy(np.int8)
        if key is None:
            key = (k_sid, k_t0, k_h)
            truth = d.truth.to_numpy(np.float32)
            lo, md, hi = (np.zeros(len(d), np.float64) for _ in range(3))
        else:
            # compare the three columns directly rather than a fused string key
            if not (np.array_equal(key[0], k_sid) and np.array_equal(key[1], k_t0)
                    and np.array_equal(key[2], k_h)):
                sys.exit(f"{f}: (basin,t0,lead) grid differs from the first seed of {member}")
        lo += d.ylo.to_numpy(np.float64)
        md += d.ymed.to_numpy(np.float64)
        hi += d.yhi.to_numpy(np.float64)
        used.append(s["seed"])
        del d
    n = len(used)
    q = np.sort(np.stack([lo, md, hi], axis=1) / n, axis=1)
    q = np.clip(q, 0.0, None)
    pred = ((q[:, 0] + q[:, 2]) / 2).astype(np.float32)
    del q, lo, md, hi
    print(f"  {member:12s} {frame:7s} seeds={sorted(used)} rows={len(pred):,}")
    return pd.DataFrame({
        "station_id": pd.Categorical(key[0]),
        "t0": pd.Categorical(key[1]),
        "h": key[2],
        "truth": truth,
        member: pred,
    })


def per_basin_lead(df, cols, lead):
    """Median-ready per-basin NSE at ONE lead, equal weight across cols."""
    sub = df[df.h == lead]
    if not len(sub):
        return pd.Series(dtype=float)
    e = sub[cols].to_numpy(float).mean(axis=1)
    out = {}
    tr = sub.truth.to_numpy(float)
    for s, idx in sub.groupby("station_id").indices.items():
        y, p = tr[idx], e[idx]
        k = np.isfinite(y) & np.isfinite(p)
        y, p = y[k], p[k]
        if len(y) < MIN_ROWS or np.var(y) < 1e-9:
            continue
        out[s] = 1 - np.mean((y - p) ** 2) / np.var(y)
    return pd.Series(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="val14", choices=["val14", "test14"])
    ap.add_argument("--members", required=True)
    ap.add_argument("--seeds", default="501,502,503")
    ap.add_argument("--protocol", default="modern-v3-9yr", choices=sorted(l51.PROTOCOLS))
    ap.add_argument("--basin-list", default="data/m1_cohort.json")
    ap.add_argument("--crosscheck", action="store_true",
                    help="assert the h==1 slice reproduces the day-1 dump's median")
    ap.add_argument("--out", default="benchmarks/m1_allleads.json")
    a = ap.parse_args()
    members = [x for x in a.members.split(",") if x]
    seeds = set(int(x) for x in a.seeds.split(",") if x) or None

    print(f"MODERN-1 all-leads — frame={a.frame} protocol={a.protocol}")
    m = None
    for name in members:
        d = load_all_leads(name, a.frame, seeds=seeds, protocol=a.protocol)
        if m is None:
            m = d
        else:
            m = m.merge(d.drop(columns=["truth"]), on=["station_id", "t0", "h"], how="inner")
    if a.basin_list:
        ids = set(str(x).zfill(8) for x in json.load(open(a.basin_list))["cohort"])
        m = m[m.station_id.isin(ids)]
    print(f"joined {len(m):,} rows  {m.station_id.nunique()} basins  "
          f"leads {sorted(m.h.unique())}")

    rows, res = [], {"frame": a.frame, "protocol": a.protocol, "members": members,
                     "seeds": sorted(seeds) if seeds else "all",
                     "rows": int(len(m)), "basins": int(m.station_id.nunique()),
                     "combiner": "equal-weight mean, zero fitted parameters",
                     "by_lead": {}}
    for h in LEADS:
        ens = per_basin_lead(m, members, h)
        if not len(ens):
            continue
        per_mem = {mem: float(per_basin_lead(m, [mem], h).median()) for mem in members}
        res["by_lead"][h] = {"ensemble": float(ens.median()), "n_basins": int(len(ens)),
                             "members": per_mem}
        rows.append((h, float(ens.median()), len(ens), per_mem))

    w = max(12, max(len(x) for x in members) + 1)
    print("\n" + "=" * (22 + w * len(members)))
    print(f"{'lead':>4} {'ENSEMBLE':>10} {'n':>5}  " + "".join(f"{x:>{w}}" for x in members))
    print("-" * (22 + w * len(members)))
    for h, e, n, pm in rows:
        print(f"{h:>4} {e:>10.6f} {n:>5}  " + "".join(f"{pm[x]:>{w}.6f}" for x in members))
    print("=" * (22 + w * len(members)))
    if rows:
        d1, dN = rows[0][1], rows[-1][1]
        print(f"\nlead 1 {d1:.6f} -> lead {rows[-1][0]} {dN:.6f}   decay {dN - d1:+.6f}")
        print("⚠️ A decline is EXPECTED: --h1-weight 0.5 spends half the gradient on lead 1,")
        print("   which the benchmark scores and the horizon does not.")
        res["lead1"] = d1; res["lead_last"] = dN; res["decay"] = dN - d1

    if a.crosscheck:
        day1_frame = "val1" if a.frame == "val14" else "test1"
        print(f"\nINTEGRITY: h==1 of {a.frame} vs the {day1_frame} dump")
        j = l51.join(members, day1_frame, seeds=seeds, protocol=a.protocol)
        ids = set(str(x).zfill(8) for x in json.load(open(a.basin_list))["cohort"])
        j = j[j.station_id.isin(ids)]
        ref = float(l51.per_basin(j, members).median())
        got = rows[0][1]
        print(f"  {day1_frame} dump      {ref:.9f}")
        print(f"  {a.frame} h==1  {got:.9f}")
        print(f"  difference       {got - ref:+.3g}")
        res["crosscheck"] = {"day1_frame_median": ref, "allleads_h1_median": got,
                             "difference": got - ref}
        assert abs(got - ref) < 1e-6, (
            f"h==1 of {a.frame} ({got:.9f}) does not reproduce {day1_frame} ({ref:.9f}). "
            "One of the two dumps is wrong -- do NOT report this table.")
        print("  ✅ PASS — the two dump paths agree; the table is trustworthy.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
