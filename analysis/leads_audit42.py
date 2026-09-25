#!/usr/bin/env python3
"""LEDGER 42 S0.2 — lead/stride coverage audit. REGISTERED NOT-A-QUERY.

Reads ONLY the metadata columns (station_id, t0, h) of every production-stream
dump, TRAIN and TEST sides. No truth, no predictions, no other columns are
loaded (usecols enforces this). Registered in EXPERIMENTS.md / ledger 42,
NOT-A-QUERY class 1; the column list above is the logged read.

Questions it answers, per stream:
  1. Which leads h exist (and rows per h)?
  2. Does target_date = t0 + h tile the window continuously (day-gap 1) with
     no target_date appearing at more than one lead?  (stride-14 tiling ->
     F2 == F3 by construction)
  3. Which TEST dumps need a B3 re-dump before F2 is defined?

Output: benchmarks/ledger42_leads_audit.json + a printed table.
Runs on the 1080 box (dump root gpu1080/dumps, dhbv_dumps).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))

L = Path("gpu1080/dumps")
H = Path("dhbv_dumps")

MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")

USECOLS = ["station_id", "t0", "h"]  # the registered metadata-only read


def lstm_files(forcing, train):
    """Mirror gate_eval._lstm_seeds naming, incl. the nldas _NEW supersede rule."""
    if train:
        fs = sorted(L.glob(f"camels531ls_{forcing}_nhlstm_TRAIN_s*.csv.gz"))
        if any(p.name.endswith("_TRAIN_s1111.csv.gz") for p in fs):
            fs = [p for p in fs if not p.name.endswith("_TRAIN_s111.csv.gz")]
        return fs
    fs = sorted(p for p in L.glob(f"camels531ls_{forcing}_nhlstm_s*.csv.gz")
                if "_TRAIN" not in p.name)
    if (L / f"camels531ls_{forcing}_nhlstm_s111_NEW.csv.gz").exists():
        fs = [p for p in fs if not p.name.endswith("_s111.csv.gz")]
    return fs


def stream_files():
    """(stream, side, [files]) for the frozen 9-stream production config."""
    out = []
    for f in ("daymet", "nldas", "maurer", "multi"):
        for side in ("TEST", "TRAIN"):
            out.append((f"lstm_{f}", side, lstm_files(f, side == "TRAIN")))
    for f in ("daymet", "nldas", "maurer"):
        out.append((f"dhbv_{f}", "TEST",
                    sorted(p for p in H.glob(f"camels531ls_{f}_dhbv_s*.csv.gz")
                           if "TRAIN" not in p.name)))
        out.append((f"dhbv_{f}", "TRAIN",
                    sorted(H.glob(f"camels531ls_{f}_dhbv_TRAIN_s*.csv.gz"))))
        out.append((f"dhbv_{f}", "TRAIN_ALLH",
                    sorted(H.glob(f"camels531ls_{f}_dhbv_ALLH_TRAIN_s*.csv.gz"))))
    for name, seeds in (("lstm_multi5", MULTI5), ("lstm_multi6", MULTI6)):
        short = name.replace("lstm_", "")
        out.append((name, "TEST",
                    [L / f"camels531ls_{short}_nhlstm_{s}.csv.gz" for s in seeds]))
        out.append((name, "TRAIN",
                    [L / f"camels531ls_{short}_nhlstm_TRAIN_{s}.csv.gz" for s in seeds]))
    return out


def audit_file(path):
    df = pd.read_csv(path, usecols=USECOLS)
    df["t0"] = pd.to_datetime(df["t0"])
    df["target"] = df["t0"] + pd.to_timedelta(df["h"], unit="D")
    h_counts = df["h"].value_counts().sort_index()

    # duplicate target_date at >1 lead? (per station)
    dup = int(df.duplicated(["station_id", "target"]).sum())

    # continuity: per-station day gaps of the sorted unique target dates
    gaps = (df.sort_values(["station_id", "target"])
              .groupby("station_id")["target"].diff().dt.days.dropna())
    frac_gap1 = float((gaps == 1).mean()) if len(gaps) else float("nan")

    # t0 stride (modal spacing of distinct t0s, first station as sample)
    sid = df["station_id"].iloc[0]
    t0s = df.loc[df["station_id"] == sid, "t0"].drop_duplicates().sort_values()
    stride = int(t0s.diff().dt.days.mode().iloc[0]) if len(t0s) > 1 else None

    return {
        "rows": int(len(df)),
        "stations": int(df["station_id"].nunique()),
        "h_values": [int(x) for x in h_counts.index],
        "rows_per_h": {int(k): int(v) for k, v in h_counts.items()},
        "t0_stride_days": stride,
        "dup_station_target_rows": dup,
        "frac_target_gap_1day": round(frac_gap1, 4),
        "target_min": str(df["target"].min().date()),
        "target_max": str(df["target"].max().date()),
    }


def main():
    results = {}
    problems = []
    for stream, side, files in stream_files():
        key = f"{stream}/{side}"
        if not files:
            results[key] = {"files": 0, "status": "NO FILES"}
            problems.append(f"{key}: no files")
            print(f"{key:<28} NO FILES")
            continue
        missing = [str(p) for p in files if not Path(p).exists()]
        if missing:
            results[key] = {"files": len(files), "missing": missing,
                            "status": "MISSING FILES"}
            problems.append(f"{key}: missing {len(missing)}")
            print(f"{key:<28} MISSING {missing}")
            continue
        # audit the first seed fully; verify the others carry the same h set
        first = audit_file(files[0])
        same = True
        for p in files[1:]:
            hs = sorted(pd.read_csv(p, usecols=["h"])["h"].unique().tolist())
            if [int(x) for x in hs] != first["h_values"]:
                same = False
                problems.append(f"{key}: seed {p.name} h-set {hs} != {first['h_values']}")
        first.update({"files": len(files), "seeds_h_consistent": same})
        results[key] = first
        hs = first["h_values"]
        hdesc = f"h=1..{max(hs)}" if hs == list(range(1, max(hs) + 1)) else f"h={hs}"
        print(f"{key:<28} files={len(files)}  {hdesc:<10} stride={first['t0_stride_days']}"
              f"  dup={first['dup_station_target_rows']}"
              f"  gap1={first['frac_target_gap_1day']}"
              f"  {first['target_min']}..{first['target_max']}")

    # ---- the S0.2 verdicts -------------------------------------------------
    def full14(key):
        r = results.get(key, {})
        return r.get("h_values") == list(range(1, 15))

    test_needs_redump = [k for k in results
                         if k.endswith("/TEST") and not full14(k)]
    tiling = all(
        r.get("dup_station_target_rows", 1) == 0
        and r.get("frac_target_gap_1day", 0) > 0.99
        for k, r in results.items()
        if full14(k) and r.get("files", 0) > 0
    )
    verdicts = {
        "F2_test_ready": not test_needs_redump,
        "test_streams_needing_B3_redump": test_needs_redump,
        "stride14_tiles_no_overlap_on_full14_dumps": tiling,
        "F2_equals_F3_if_tiling": tiling,
        "problems": problems,
    }
    print("\nVERDICTS")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")

    out = {"registered": "LEDGER 42 S0.2 NOT-A-QUERY; columns read: "
                         + ",".join(USECOLS),
           "streams": results, "verdicts": verdicts}
    Path("benchmarks/ledger42_leads_audit.json").write_text(json.dumps(out, indent=2))
    print("\nwrote benchmarks/ledger42_leads_audit.json")


if __name__ == "__main__":
    sys.exit(main())
