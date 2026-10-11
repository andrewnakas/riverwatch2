#!/usr/bin/env python3
"""Append recent NLDAS-2 areal days to the published MODERN-1 forcing table.

WHY A SEPARATE ENTRY POINT. `build_nldas2_areal.py` works in whole months and
checkpoints each one, which is right for a corpus build and wrong for a daily
append in two ways: a month is ~744 granules (~550 MB) where an append needs a
handful of days, and once the current month's checkpoint exists `build_month`
SKIPS it, so the month would be frozen at whatever day it was first built on
and never pick up the days that followed.

So this loops over DAYS. It imports the fetch and reduce path from that builder
rather than reimplementing it, because the reduction arithmetic -- and in
particular the mandatory `np.flipud` (NLDAS stores latitude ASCENDING while the
weights are north-first, "or every basin silently gets the wrong cells") -- must
not be able to drift between the corpus the model trained on and the forcing it
is served.

WHAT IT WRITES. The same per-basin table `app/modern_forcing.load_table()`
reads: `date` + the 6 channels + `src`. Rows written here carry
`src=nldas_areal`; the Open-Meteo point builder writes `src=om_point`. Merging
is by the flip rule in build_m1_forcing.py -- areal always wins over point for
the same date, never the reverse -- so a day arrives point-filled while fresh
and is replaced by areal once NLDAS-2 catches up, which is exactly the design
GATE 2 priced at a paired -0.004018.

LATENCY. NLDAS-2 publishes at **today-5**, measured on the response BODY: days
today-0..-4 return a 1,119-byte HTML page while a plain HEAD probe returns
HTTP 200 for every one of them. So this never asks for anything newer than
today-5, and the newest 4 days of any encoder window stay point-filled.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: Measured 2026-10-10 on the response body, not the status code.
PUBLICATION_LATENCY_DAYS = 5

COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
HEADER = ["date"] + COLS + ["src"]
SRC = "nldas_areal"


def _load_builder(root: Path):
    """Import the corpus builder as a module, so its fetch/reduce path is the
    single source of truth for how a granule becomes a basin mean."""
    p = root / "scripts_modern" / "build_nldas2_areal.py"
    spec = importlib.util.spec_from_file_location("_nldas_builder", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_nldas_builder"] = mod
    spec.loader.exec_module(mod)
    return mod


def day_means(mod, y, m, d, W, workers, workdir):
    """One day's 6 basin-mean channels, via the builder's own hour reduction.

    `mod.do_hour` takes one tuple and returns (y, m, d, h, {var: basin_means}),
    with the mandatory np.flipud and the per-thread cookie jar inside it. One
    retry pass over just the failures, mirroring build_month: a day is 24
    fetches and losing all of them to one bad hour is needless.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    vals = {}

    def run(hours):
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(mod.do_hour, (y, m, d, h, workdir, W)): h for h in hours}
            for fu in as_completed(futs):
                h = futs[fu]
                try:
                    _, _, _, _, out = fu.result()
                except Exception as exc:
                    failed.append(h)
                    print(f"    {y}-{m:02d}-{d:02d} {h:02d}Z retryable: {exc}")
                    continue
                vals[h] = out
        return failed

    failed = run(range(24))
    if failed:
        print(f"    {y}-{m:02d}-{d:02d}: retrying {len(failed)} hour(s)")
        time.sleep(10)
        failed = run(failed)
    if len(vals) != 24:
        raise RuntimeError(f"{y}-{m:02d}-{d:02d}: {len(vals)}/24 hours")

    hrs = sorted(vals)
    stack = {v: np.vstack([vals[h][v] for h in hrs]) for v in mod.VARS}
    tair = stack["Tair"] - 273.15
    q, ps = stack["Qair"], stack["PSurf"]
    # Identical arithmetic to build_nldas2_areal.build_month, deliberately.
    return {
        "precipitation_sum": np.nansum(stack["Rainf"], axis=0),
        "temperature_2m_mean": np.nanmean(tair, axis=0),
        "temperature_2m_max": np.nanmax(tair, axis=0),
        "temperature_2m_min": np.nanmin(tair, axis=0),
        "shortwave_radiation_sum": np.nanmean(stack["SWdown"], axis=0) * 86400.0 / 1e6,
        "vapor_pressure": np.nanmean(q * ps / (0.622 + 0.378 * q), axis=0),
    }


def read_table(p: Path):
    if not p.exists():
        return []
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def write_table(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", newline="") as gz:
        w = csv.DictWriter(gz, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)
    # Re-read: a truncated write that still parses is what this catches.
    back = read_table(p)
    if len(back) != len(rows):
        p.unlink(missing_ok=True)
        raise RuntimeError(f"{p.name}: wrote {len(rows)} rows, read back {len(back)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--archive-dir", default="data/m1_forcing")
    ap.add_argument("--weights", default="data/nldas_basin_weights.pkl")
    ap.add_argument("--basins", default="data/m1_cohort.json",
                    help="JSON with a 'cohort' or 'servable' list, or a plain list")
    ap.add_argument("--days", type=int, default=10,
                    help="how many days back from today-latency to ensure")
    ap.add_argument("--through", default="", help="YYYY-MM-DD; default today-5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-days", type=int, default=425)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    mod = _load_builder(root)
    # Point the builder's module globals at this checkout.
    mod.ROOT = root
    wp = Path(a.weights)
    mod.WEIGHTS = wp if wp.is_absolute() else root / wp
    if not mod.WEIGHTS.exists():
        print(f"ERROR: weights pickle absent at {mod.WEIGHTS}", file=sys.stderr)
        return 1

    through = (date.fromisoformat(a.through) if a.through
               else date.today() - timedelta(days=PUBLICATION_LATENCY_DAYS))
    wanted = [through - timedelta(days=i) for i in range(a.days)][::-1]

    raw = json.loads((root / a.basins).read_text()) if (root / a.basins).exists() \
        else json.loads(Path(a.basins).read_text())
    if isinstance(raw, dict):
        ids = raw.get("cohort") or raw.get("servable") or []
    else:
        ids = raw
    ids = [str(x) for x in ids]

    W = mod.load_weights()
    basins = list(W[0])
    index = {b: i for i, b in enumerate(basins)}
    missing = [s for s in ids if s not in index]
    if missing:
        print(f"WARNING: {len(missing)} requested basins are not in the weight "
              f"pickle: {missing[:5]}")
    ids = [s for s in ids if s in index]
    print(f"{len(ids)} basins; ensuring {wanted[0]}..{wanted[-1]} "
          f"(NLDAS-2 publishes at today-{PUBLICATION_LATENCY_DAYS})")

    arch = root / a.archive_dir
    # Which days are already areal in every requested basin's table? A day is
    # only "done" if it is areal EVERYWHERE -- otherwise some basins would be
    # permanently point-filled for a day the rest have areal.
    have = {d.isoformat(): 0 for d in wanted}
    for sid in ids:
        rows = {r["date"]: r.get("src") for r in read_table(arch / f"{sid}.csv.gz")}
        for k in have:
            if rows.get(k) == SRC:
                have[k] += 1
    todo = [d for d in wanted if have[d.isoformat()] < len(ids)]
    print("days already areal everywhere: "
          f"{len(wanted) - len(todo)}/{len(wanted)}; fetching {len(todo)}")
    if not todo:
        print("nothing to do")
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="nldas_app_", dir=os.environ.get("TMPDIR", "/tmp")))
    mod.preflight(workdir)

    produced = {}
    for d in todo:
        try:
            produced[d.isoformat()] = day_means(mod, d.year, d.month, d.day,
                                                W, a.workers, workdir)
            print(f"  {d} ok")
        except Exception as exc:
            # A day that is not published yet, or a transient failure, is not
            # fatal: the point builder keeps that day filled and the next run
            # tries again. Silence here would be the bug.
            print(f"  {d} SKIPPED: {exc}")

    if not produced:
        print("no days produced")
        return 0

    cutoff = (date.today() - timedelta(days=a.keep_days)).isoformat()
    rank = {"om_point": 0, SRC: 1}
    n_written = 0
    for sid in ids:
        p = arch / f"{sid}.csv.gz"
        rows = {r["date"]: dict(r) for r in read_table(p)}
        bi = index[sid]
        for dstr, chans in produced.items():
            new = {"date": dstr, "src": SRC}
            bad = False
            for c in COLS:
                v = float(chans[c][bi])
                if not np.isfinite(v):
                    bad = True
                    break
                new[c] = f"{v:.6g}"
            if bad:
                continue
            old = rows.get(dstr)
            # FLIP RULE: areal replaces point, never the reverse.
            if old is None or rank.get(new["src"], 0) >= rank.get(old.get("src"), 0):
                rows[dstr] = new
        keep = [rows[k] for k in sorted(rows) if k >= cutoff]
        if keep:
            write_table(p, keep)
            n_written += 1

    areal_days = sorted(produced)
    print(f"wrote {n_written} basin tables; areal days added/refreshed: "
          f"{areal_days[0]}..{areal_days[-1]} ({len(areal_days)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
