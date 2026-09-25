#!/usr/bin/env python
"""MODERN-1: build the authoritative discharge store, 1980-01-01 -> today.

WHY THIS EXISTS. `build_daymet_areal_modern.py:102` (`load_q`) reads discharge out
of `camels_corpus_daymet_modern`, the old POINT corpus, which begins in 2015.
Measured consequence, already present in the shipped corpus: the 2014 prefix of
`camels_corpus_daymet_modern_areal` carries forcings but **zero discharge on every
basin** -- its targets start 2015-01-01. Re-running that builder for 1980-2013
would have produced 34 years of forcings with no targets at all, and would have
exited 0.

So discharge is built once, here, from one source, and every corpus builder reads
this store.

SOURCE DECISION (measured 2026-09-23, 30 cohort basins x 2010-2014, 53,356 pairs):
fresh USGS DV vs the frozen CAMELS q_cfs --
    frozen == round(usgs, 0)      73.90%
    median ratio usgs/frozen      1.000000
    log-space r                   0.998481
    differing by >5%              3,848 pairs, but 3,780 of them in 4 basins
                                  (08195000, 03463300, 08164600, 07346045)
i.e. the gap is CAMELS' integer rounding plus genuine USGS record revisions in a
handful of low-flow/ephemeral gauges. There is no systematic level shift, so a
single fresh-USGS series over the whole span is safe, and it avoids putting a seam
at 2015-01-01 -- exactly where the modern training window ends.

The residual cost -- the frozen record's targets are pre-revision values -- is not
assumed away: it is measured separately so a Gate A delta can be split into
"pipeline" and "target revision" instead of confounding the two.
"""
from __future__ import annotations
import argparse, gzip, json, os, sys, threading, time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.usgs import fetch_daily_discharge  # noqa: E402

PRINT_LOCK = threading.Lock()


def log(*a):
    with PRINT_LOCK:
        print(*a, flush=True)


def build_one(gauge: str, start: dt.date, end: dt.date, outdir: Path,
              force: bool) -> tuple[str, str, str]:
    op = outdir / f"{gauge}.csv.gz"
    want = (end - start).days + 1
    if op.exists() and not force:
        try:                              # idempotent: trust the artifact, not an exit code
            with gzip.open(op, "rt") as fh:
                n = sum(1 for _ in fh) - 1
            if n == want:
                return ("skip", gauge, f"{n}d")
        except Exception:
            pass

    df = fetch_daily_discharge(gauge, start, end, max_age_hours=10 ** 9)
    if df is None or len(df) == 0:
        return ("empty", gauge, "no rows returned")

    obs: dict[str, float] = {}
    for d, v in zip(df["date"].astype(str), df["q_cfs"].astype(float)):
        if not np.isfinite(v) or v < 0:
            continue                      # -999999 = USGS NoData: mask, never zero-fill
        obs[d[:10]] = float(v)

    tmp = op.with_suffix(".tmp.gz")
    n_obs = 0
    with gzip.open(tmp, "wt") as fh:
        fh.write("date,q_cfs\n")
        d = start
        while d <= end:
            s = d.isoformat()
            v = obs.get(s)
            if v is None:
                fh.write(f"{s},\n")
            else:
                fh.write(f"{s},{v}\n")
                n_obs += 1
            d += dt.timedelta(days=1)
    with gzip.open(tmp, "rt") as fh:       # verify the artifact before publishing it
        got = sum(1 for _ in fh) - 1
    if got != want:
        tmp.unlink(missing_ok=True)
        return ("bad", gauge, f"wrote {got} rows, wanted {want}")
    tmp.replace(op)
    return ("ok", gauge, f"{n_obs}/{want} obs ({100 * n_obs / want:.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/camels_gauge_ids.json")
    ap.add_argument("--key", default="671")
    ap.add_argument("--start", default="1980-01-01")
    ap.add_argument("--end", default="", help="default: yesterday")
    ap.add_argument("--out", default="data/modern_corpus/q_store")
    ap.add_argument("--workers", type=int, default=6, help="be polite to waterservices")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.chdir(ROOT)
    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today() - dt.timedelta(days=1)
    ids = [str(i).zfill(8) for i in json.loads(Path(a.ids).read_text())[a.key]]
    if a.limit:
        ids = ids[:a.limit]
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"q_store: {len(ids)} basins  {start} .. {end}  "
        f"({(end - start).days + 1} days)  -> {outdir}")
    t0 = time.time()
    tally: dict[str, int] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, g, start, end, outdir, a.force): g for g in ids}
        for fu in as_completed(futs):
            st, g, msg = fu.result()
            tally[st] = tally.get(st, 0) + 1
            done += 1
            if st not in ("ok", "skip") or done % 50 == 0:
                log(f"[{done}/{len(ids)}] {st:5s} {g} {msg}  ({time.time() - t0:.0f}s)")
    log("TALLY " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    log(f"FINISHED in {(time.time() - t0) / 60:.1f} min")
    return 0 if tally.get("bad", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
