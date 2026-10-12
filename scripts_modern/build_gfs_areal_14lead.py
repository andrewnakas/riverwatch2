#!/usr/bin/env python3
"""Areal GFS forcing for ALL 14 decoder lead days, not just lead 1.

WHY THIS MATTERS. `scripts_modern/gfs_to_forcing_schema.py` currently fills
leads 2-14 from OBSERVED corpus values and stamps its own README saying so:
"⛔ Nothing read off leads 2-14 of these files is forecast skill -- it is
perfect forcing." A product channel at the scored step is worth ~3.8x what it
is worth in the encoder (encoder-only +0.003487 vs encoder+decoder +0.013106),
so the decoder is where the remaining forcing substitution actually costs
something.

WHY A SIBLING OF build_gfs_areal_lead1.py RATHER THAN AN EDIT. That builder
produced the registered 115-date artifact behind
`benchmarks/m1_decay_day1_nldasm9.json` (-0.048033, the operational discount
this project quotes). Editing it in place would put that number's
reproducibility at risk for no gain. This imports its Reducer, its idx parser
and its decoder, so the grid handling and the byte-range reads cannot drift,
and adds only the per-lead record selection.

⚠️ PRECIPITATION IS THE WHOLE DIFFICULTY, and the lead-1 builder documents how
it goes wrong: the "0-1 day acc" 24-hour record exists ONLY at f024, and GFS
ships BOTH a 6-hour bucket and a run-total at every step whose index strings
are the SAME LENGTH -- so the old "shortest window" tiebreak was arbitrary and
could double-count. It produced a 300 mm/day basin maximum.

So the window is PARSED, never matched by substring, and selection is
structural: require `acc` with `b - a == 6`, then assert the four selected
windows tile 24(L-1)..24L disjointly and contiguously. A run-total has a == 0
and b > 6 and can never satisfy `b - a == 6`. Probed on gfs.20261001/00z, all
14 lead days tile exactly and every other channel yields exactly 4 records per
lead (scripts_modern/probe_gfs_f336_idx.py).

⭐ --verify-lead1 is the regression that licenses trusting this at all: the new
4-bucket sum must agree with the lead-1 builder's single 24-hour record, or the
registered decay number is no longer reproducible from the same inputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: eccodes/cfgrib is not demonstrably thread-safe, and libhdf5 in the sibling
#: NLDAS builder needed exactly this (measured: segfault). The network GET
#: dominates, so parallelising the fetch and serialising the decode captures
#: nearly all of the win at none of the risk.
GRIB_LOCK = threading.Lock()

#: GRIB2 packing step of the APCP records, MEASURED rather than assumed: at the
#: grid level the 4-bucket sum and the single 24-hour record differ only by
#: exactly 0.0625 mm (= 1/16) or 0.125 mm, 100% of differing cells sit at an
#: integer multiple of 0.0625, there are only TWO distinct difference values,
#: and the max multiple is 2.00 -- which is what summing four independently
#: quantised buckets produces. Measured on gfs.20261001/00z over 1,038,240
#: cells (scripts_modern/diag_gfs_lead1_packing.py).
GRIB_APCP_PACKING_MM = 0.0625
#: Four buckets, each carrying up to half a step of rounding either way, so the
#: sum can differ from the single record by up to 4 half-steps. Areal averaging
#: then ATTENUATES this (a weighted mean of +/- steps), so the basin-level
#: difference is strictly smaller -- 0.0913 mm measured.
#:
#: ⚠️ An earlier version used a flat 0.1 mm and the real run came in at
#: 0.091278, i.e. it passed by 9%. A tolerance tighter than the stored
#: precision does not detect a defect, it manufactures one -- the same trap as
#: the exact-equality truth guard that aborted GATE 1 on float32 storage noise
#: while blaming the join.
LEAD1_TOL_MM = 4 * GRIB_APCP_PACKING_MM / 2

WIN_RE = re.compile(r"^(\d+)-(\d+) (hour|day) (acc|ave|max|min)")


def _field5(idx_line: str) -> str:
    """The window/validity text of a wgrib2 .idx line (colon field 6).

    Lines look like
      `12:7340:d=2026100100:APCP:surface:6-12 hour acc fcst:`
    so the window text is index 5 after splitting on ':'. Guarded because a
    malformed line must not raise inside the selector.
    """
    parts = idx_line.split(":")
    return parts[5] if len(parts) > 5 else idx_line


def load_lead1_builder():
    p = ROOT / "scripts_modern" / "build_gfs_areal_lead1.py"
    spec = importlib.util.spec_from_file_location("_gfs1", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_gfs1"] = m
    spec.loader.exec_module(m)
    return m


def window(field: str):
    """(start_hour, end_hour, kind) for an accumulation/average record, else None."""
    mt = WIN_RE.match(field)
    if not mt:
        return None
    a, b, unit, kind = int(mt.group(1)), int(mt.group(2)), mt.group(3), mt.group(4)
    if unit == "day":
        a, b = a * 24, b * 24
    return a, b, kind


def buckets_for_lead(lead: int):
    """The four 6-hour windows that tile this lead day, in order."""
    lo = 24 * (lead - 1)
    return [(lo, lo + 6), (lo + 6, lo + 12), (lo + 12, lo + 18), (lo + 18, lo + 24)]


def steps_for_lead(lead: int):
    """The four forecast hours whose records cover this lead day."""
    lo = 24 * (lead - 1)
    return [lo + 6, lo + 12, lo + 18, lo + 24]


def pick_bucket(recs, lead, var, kind):
    """The four disjoint 6-hour `kind` records for `var` on this lead day.

    Raises rather than falling back. The lead-1 builder silently substituted
    tmean when TMAX/TMIN were absent; over 14 leads that would flatten the
    diurnal range invisibly, so an absent record fails the lead instead.
    """
    want = buckets_for_lead(lead)
    found = {}
    for h in steps_for_lead(lead):
        for ln, arr in recs.get(h, {}).get(var, []):
            w = window(_field5(ln))
            if w is None:
                continue
            a, b, k = w
            if k == kind and b - a == 6 and (a, b) in want:
                found[(a, b)] = arr
    got = sorted(found)
    if got != want:
        raise RuntimeError(
            f"lead {lead} {var} {kind}: selected {got}, want {want} "
            "(four disjoint 6-hour windows tiling the day; a run-total has "
            "a==0 and b>6 and cannot satisfy b-a==6)")
    return [found[w] for w in want]


def pick_instant(recs, lead, var):
    """The instantaneous records for `var` at this lead day's four hours."""
    out = []
    for h in steps_for_lead(lead):
        cs = recs.get(h, {}).get(var, [])
        inst = [a for ln, a in cs if window(_field5(ln)) is None]
        if inst:
            out.append(inst[0])
    if len(out) != 4:
        raise RuntimeError(f"lead {lead} {var}: {len(out)}/4 instantaneous records")
    return out


def build_day(g1, red, fs, day: str, leads: int, workers: int, verify_lead1: bool):
    """All `leads` lead days for one 00z init. Returns a DataFrame."""
    hours = sorted({h for L in range(1, leads + 1) for h in steps_for_lead(L)})
    recs: dict[int, dict] = {}

    def fetch(h):
        """One step's wanted records, as {var: [(idx_line, global_array), ...]}.

        g1.idx_ranges returns (grb_path, [(var, lev, line, start, nbytes)...])
        and g1.decode does the ranged GET plus the cfgrib open, returning that
        same mapping -- so the byte-range read and the grid handling stay in
        one place. The decode is serialised behind GRIB_LOCK; the GET inside it
        dominates, so the pool still buys nearly all of the parallelism.
        """
        grb, rngs = g1.idx_ranges(fs, day, h)
        if not grb or not rngs:
            return h, None
        with GRIB_LOCK:
            return h, g1.decode(fs, grb, rngs)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, h): h for h in hours}
        for fu in as_completed(futs):
            h, out = fu.result()
            if out:
                recs[h] = out

    missing = [h for h in hours if h not in recs]
    if missing:
        raise RuntimeError(f"{day}: {len(missing)} of {len(hours)} steps unavailable")

    frames = []
    for L in range(1, leads + 1):
        apcp = pick_bucket(recs, L, "APCP", "acc")
        precip = np.sum([red(a) for a in apcp], axis=0)

        tmp = [red(a) for a in pick_instant(recs, L, "TMP")]
        tmean = np.mean(tmp, axis=0) - 273.15
        tmax = np.max([red(a) for a in pick_bucket(recs, L, "TMAX", "max")], axis=0) - 273.15
        tmin = np.min([red(a) for a in pick_bucket(recs, L, "TMIN", "min")], axis=0) - 273.15
        sw = np.mean([red(a) for a in pick_bucket(recs, L, "DSWRF", "ave")], axis=0) * 86400 / 1e6
        q = np.mean([red(a) for a in pick_instant(recs, L, "SPFH")], axis=0)
        pr = np.mean([red(a) for a in pick_instant(recs, L, "PRES")], axis=0)
        vp = q * pr / (0.622 + 0.378 * q)

        # Physical plausibility, per lead day. This is the 300 mm/day catcher,
        # and tmin<=tmean<=tmax is the cheapest check of all and was asserted
        # nowhere before.
        for name, arr, lo, hi in (("precip", precip, 0, 500), ("tmean", tmean, -60, 60),
                                  ("tmax", tmax, -60, 60), ("tmin", tmin, -60, 60),
                                  ("srad", sw, 0, 45), ("vp", vp, 10, 7000)):
            bad = (~np.isfinite(arr)) | (arr < lo) | (arr > hi)
            if bad.any():
                raise RuntimeError(f"{day} lead {L} {name}: {int(bad.sum())} basins "
                                   f"outside [{lo},{hi}] (min {np.nanmin(arr):.3f} "
                                   f"max {np.nanmax(arr):.3f})")
        if not ((tmin <= tmean + 1e-6).all() and (tmean <= tmax + 1e-6).all()):
            raise RuntimeError(f"{day} lead {L}: tmin <= tmean <= tmax violated")

        if verify_lead1 and L == 1:
            cands = recs.get(24, {}).get("APCP", [])
            tot = [a for ln, a in cands
                   if "0-1 day acc" in ln or "0-24 hour acc" in ln]
            if not tot:
                raise RuntimeError("--verify-lead1: no 24h APCP record at f024")
            ref = red(tot[0])
            d = float(np.max(np.abs(precip - ref)))
            print(f"  LEAD-1 REGRESSION: max |4-bucket sum - single 24h record| "
                  f"= {d:.6f} mm over {len(ref)} basins "
                  f"(tolerance {LEAD1_TOL_MM:.4f} mm)", flush=True)
            if d >= LEAD1_TOL_MM:
                raise RuntimeError(
                    f"--verify-lead1 FAILED: {d:.6f} mm >= {LEAD1_TOL_MM:.4f}. The "
                    "new selector moved lead 1 by more than GRIB packing can "
                    "explain, so benchmarks/m1_decay_day1_nldasm9.json "
                    "(-0.048033) is no longer reproducible from these inputs.")

        frames.append(pd.DataFrame({
            "station_id": red.basins, "lead_day": L,
            "temperature_2m_mean": tmean, "temperature_2m_max": tmax,
            "temperature_2m_min": tmin, "precipitation_sum": precip,
            "shortwave_radiation_sum": sw, "vapor_pressure": vp,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--dates", default="", help="comma-separated YYYYMMDD; overrides the range")
    ap.add_argument("--stride", type=int, default=28)
    ap.add_argument("--leads", type=int, default=14)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/mblstm/gfs_fcst_areal14")
    ap.add_argument("--verify-lead1", action="store_true",
                    help="assert the 4-bucket sum reproduces the single 24h record")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    g1 = load_lead1_builder()
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    red = g1.Reducer()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    if a.dates:
        days = [d.strip() for d in a.dates.split(",") if d.strip()]
    else:
        d0 = date.fromisoformat(a.start) if a.start else date(2021, 5, 1)
        d1 = date.fromisoformat(a.end) if a.end else date(2025, 9, 17)
        days, d = [], d0
        while d <= d1:
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(days=a.stride)
    if a.limit:
        days = days[:a.limit]
    print(f"{len(days)} init dates, leads 1..{a.leads}, {len(red.basins)} basins, "
          f"{a.workers} fetch workers -> {out}")

    done = fail = 0
    for day in days:
        dst = out / f"{day}.csv.gz"
        if dst.exists():
            done += 1
            continue
        try:
            df = build_day(g1, red, fs, day, a.leads, a.workers, a.verify_lead1)
        except Exception as exc:
            print(f"  {day}: FAILED {type(exc).__name__}: {exc}", flush=True)
            fail += 1
            continue
        want = a.leads * len(red.basins)
        if len(df) != want:
            print(f"  {day}: {len(df)} rows, want {want}; refusing")
            fail += 1
            continue
        tmp = dst.with_suffix(".tmp.gz")
        df.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(dst)
        back = pd.read_csv(dst, compression="gzip")
        if len(back) != want:
            dst.unlink(missing_ok=True)
            print(f"  {day}: wrote {want} rows, read back {len(back)}; refusing")
            fail += 1
            continue
        done += 1
        print(f"  {day}: ok ({want} rows)", flush=True)
    print(f"\ndone={done} fail={fail} -> {out}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
