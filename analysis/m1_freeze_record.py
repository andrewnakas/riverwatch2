#!/usr/bin/env python3
"""MODERN-1 — freeze the modern record. **Reads the test window ONCE and selects
nothing.**

Phase 6 of the plan. Writes `benchmarks/m1_FINAL_withq.json` / `m1_FINAL_noq.json`.

Guards, each because its absence already cost something here:
  * every backing dump's sidecar re-checked for the guarded training window
    (`l51.check_sidecar`) -- the retracted 0.9253 was a model trained on its own
    test decade;
  * ARM separation: the two arms share a protocol and a frame, so a no-q dump could
    otherwise be joined silently into the with-q roster. `--arm noq` requires
    `arm == "noq"` in every sidecar; `--arm withq` refuses any sidecar that has it;
  * ship bar enforced per member (5 seeds unless `--screen-ok`, which also stamps
    the artifact as NOT a record);
  * **the reported median must be reproducible from the per-basin map the artifact
    contains** -- asserted here, so the JSON is self-verifying;
  * **an append-only test-read audit log** (`benchmarks/m1_test_reads.log`). "test is
    read once" is otherwise an unverifiable claim about my own behaviour; with the log
    a second read is visible in the artifact trail. The script WARNS loudly when the
    log already contains a read for this arm and frame.

Both combiners are always reported (mean and trimmed) so the headline can never be an
artifact of a combiner chosen after the fact; `--combiner` only marks which one val
selected.

usage:
  m1_freeze_record.py --arm withq --protocol modern-v3-9yr \
      --members gridmetm9,nclimgridm9,nhar0m --shipbar "gridmetm9:501,502,503,504,505" \
      --combiner mean --out benchmarks/m1_FINAL_withq.json
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, subprocess, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)
NEARING = 0.879          # Nearing 2022 single-model AR, the with-q reference
NOQ_RECORD = 0.836289    # this campaign's own frozen no-q record (legacy window)
AUDIT = "benchmarks/m1_test_reads.log"


def combine(P, how):
    """⚠️ The trimmed mean is UNDEFINED below 4 members and must not silently become
    the mean under the name "trimmed". Inheriting `or P.shape[1] < 4` from the LEDGER
    56 freeze made this tool report trimmed=0.883810 at n=3 where
    analysis/m1_composition.py reports 0.883224 (drop max and min of 3 leaves the
    middle value, i.e. the median). Two tools disagreeing on one number under one name
    is how a wrong claim reaches a write-up, so n<4 now RAISES and the caller records
    the combiner as unavailable."""
    if how == "mean":
        return P.mean(1)
    if P.shape[1] < 4:
        raise ValueError(
            f"trimmed mean is undefined for {P.shape[1]} members: dropping the max and "
            f"min leaves {max(P.shape[1] - 2, 0)} value(s). Use --combiner mean, or add "
            f"members. (m1_composition.py reports the degenerate case as the median.)")
    return (P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2)


def per_basin_pred(df, pred):
    out = {}
    y = df.truth.to_numpy(float)
    for s, idx in df.groupby("station_id").indices.items():
        yy, pp = y[idx], pred[idx]
        k = np.isfinite(yy) & np.isfinite(pp)
        yy, pp = yy[k], pp[k]
        if len(yy) < l51.MIN_ROWS or np.var(yy) < 1e-9:
            continue
        out[s] = 1 - np.mean((yy - pp) ** 2) / np.var(yy)
    return pd.Series(out)


def check_arm(member, frame, protocol, arm, seeds):
    """The arm gate. Both arms share a protocol and a frame; only the sidecar
    separates them, so it is checked for EVERY backing dump, not once per member."""
    pref = l51.PROTOCOLS[protocol][2]
    files = sorted(glob.glob(f"{l51.DD}/{pref}{member}_s*_{frame}.csv.gz"))
    if seeds:
        files = [f for f in files if int(f.split("_s")[-1].split("_")[0]) in seeds]
    if not files:
        sys.exit(f"no {frame} dumps for {member!r}")
    for f in files:
        s = l51.check_sidecar(f, protocol, True)
        got = s.get("arm")
        if arm == "noq" and got != "noq":
            sys.exit(f"{f}: --arm noq but sidecar arm={got!r} — refusing to mix arms")
        if arm == "withq" and got == "noq":
            sys.exit(f"{f}: --arm withq but this is a NO-Q dump — refusing to mix arms")
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["withq", "noq"], required=True)
    ap.add_argument("--members", required=True)
    ap.add_argument("--protocol", default="modern-v3-9yr", choices=sorted(l51.PROTOCOLS))
    ap.add_argument("--frame", default="test1", choices=["test1", "val1"],
                    help="test1 is THE read. val1 allowed only for a dry run.")
    ap.add_argument("--shipbar", default="", help='"member:501,502,..." space-separated, >=5 seeds')
    ap.add_argument("--seeds", default="", help="seed set applied to every member")
    ap.add_argument("--combiner", choices=["mean", "trimmed"], default="mean",
                    help="which combiner val selected; BOTH are reported regardless")
    ap.add_argument("--basin-list", default="data/m1_cohort.json")
    ap.add_argument("--screen-ok", action="store_true",
                    help="permit a sub-ship-bar freeze; stamps the artifact NOT a record")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    members = [x for x in a.members.split(",") if x]
    seeds = set(int(x) for x in a.seeds.split(",") if x) or None
    bar = {}
    for tok in [t for t in a.shipbar.split(" ") if t]:
        m, ss = tok.split(":")
        bar[m] = {int(x) for x in ss.split(",")}

    # ---- ship bar -----------------------------------------------------------
    for m, ss in bar.items():
        if len(ss) < 5 and not a.screen_ok:
            sys.exit(f"SHIP BAR NOT MET: {m} at {len(ss)} seeds, need 5 (or --screen-ok).")
    if not bar and not a.screen_ok:
        sys.exit("no --shipbar given: a record needs an explicit per-member seed bar.")

    # ---- arm separation, every backing dump ---------------------------------
    print(f"MODERN-1 FREEZE — arm={a.arm} frame={a.frame} protocol={a.protocol}")
    for m in members:
        n = check_arm(m, a.frame, a.protocol, a.arm, seeds or bar.get(m))
        print(f"  arm gate OK: {m:14s} {n} dumps")

    # ---- the read -----------------------------------------------------------
    if a.frame == "test1":
        prior = []
        if os.path.exists(AUDIT):
            prior = [l for l in open(AUDIT) if f"arm={a.arm}" in l and "frame=test1" in l]
        if prior:
            print(f"\n⚠️⚠️ {AUDIT} ALREADY RECORDS {len(prior)} test1 read(s) for arm={a.arm}:")
            for l in prior[-3:]:
                print("   " + l.rstrip())
            print("   ⇒ This is NOT the first read. Whatever is reported now must say so;")
            print("      a second read after seeing the first is selection on test.\n")

    m = l51.join(members, a.frame, seeds=seeds, protocol=a.protocol)
    ids = set(str(x).zfill(8) for x in json.load(open(a.basin_list))["cohort"])
    m = m[m.station_id.isin(ids)]
    print(f"joined {len(m):,} rows  {m.station_id.nunique()} basins")

    P = m[members].to_numpy(float)
    out = {}
    for how in ("mean", "trimmed"):
        try:
            ser = per_basin_pred(m, combine(P, how))
        except ValueError as e:
            print(f"  combiner {how!r} unavailable: {e}")
            out[how] = None
            continue
        out[how] = {"median": float(ser.median()), "n_basins": int(len(ser)),
                    "per_basin": {k: float(v) for k, v in ser.items()}}
    if out[a.combiner] is None:
        sys.exit(f"--combiner {a.combiner} is unavailable for {len(members)} members (see above)")
    headline = out[a.combiner]["median"]

    # ---- the artifact must verify itself ------------------------------------
    for how in ("mean", "trimmed"):
        if out[how] is None:
            continue
        vals = np.array(list(out[how]["per_basin"].values()), dtype=float)
        assert abs(float(np.median(vals)) - out[how]["median"]) < 1e-12, \
            f"{how}: median not reproducible from the per-basin map in this artifact"
    print("✅ every reported median is reproducible from its per-basin map in the artifact")

    ref = NEARING if a.arm == "withq" else NOQ_RECORD
    ref_name = ("Nearing 2022 single-model AR (0.879, legacy window)" if a.arm == "withq"
                else "this campaign's frozen no-q record (0.836289, legacy window)")
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = ""

    res = {"_note": "MODERN-1 modern-window record. test read once, selecting nothing. "
                    "Both combiners reported so the headline cannot be a combiner artifact. "
                    "⚠️ The reference is on a DIFFERENT decade, cohort and product set; it is "
                    "context, not a like-for-like comparison.",
           "frozen_utc": datetime.now(timezone.utc).isoformat(),
           "arm": a.arm, "frame": a.frame, "protocol": a.protocol,
           "protocol_window": list(l51.PROTOCOLS[a.protocol][:2]),
           "composition": members, "combiner_selected_on_val": a.combiner,
           "ship_bar": {k: sorted(v) for k, v in bar.items()},
           "is_record": bool(bar) and not a.screen_ok,
           "screen_only": a.screen_ok,
           "cohort": a.basin_list, "basins": int(m.station_id.nunique()), "rows": int(len(m)),
           "combination": "equal weight, zero fitted parameters",
           "readout": "(ylo+yhi)/2",
           "median_nse_mean": out["mean"]["median"] if out["mean"] else None,
           "median_nse_trimmed": out["trimmed"]["median"] if out["trimmed"] else None,
           "trimmed_unavailable_reason": (None if out["trimmed"] else
                                          f"undefined for {len(members)} members (<4)"),
           "headline": headline,
           "reference": ref, "reference_name": ref_name,
           "delta_vs_reference": headline - ref,
           "per_basin_nse": out[a.combiner]["per_basin"],
           "git_commit": git}

    print("\n" + "=" * 74)
    print(f"{'composition':24s} {', '.join(members)}")
    print(f"{'equal-weight MEAN':24s} {out['mean']['median']:.6f}")
    print(f"{'trimmed mean':24s} " + (f"{out['trimmed']['median']:.6f}"
                                      if out["trimmed"] else "n/a (needs >=4 members)"))
    print(f"{'HEADLINE (' + a.combiner + ')':24s} {headline:.6f}")
    print(f"{'vs ' + ref_name:24s} {headline - ref:+.6f}")
    print(f"{'is a record':24s} {res['is_record']}")
    print("=" * 74)
    print("⚠️ The reference is a DIFFERENT decade, cohort and product set. Quote it as context,")
    print("   never as a like-for-like win.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.frame == "test1":
        with open(AUDIT, "a") as fh:
            fh.write(f"{res['frozen_utc']} arm={a.arm} frame=test1 protocol={a.protocol} "
                     f"members={','.join(members)} headline={headline:.6f} "
                     f"is_record={res['is_record']} out={a.out} git={git[:8]}\n")
        print(f"\ntest read appended to {AUDIT} — a later read will be visible there")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
