#!/usr/bin/env python
"""MODERN-1: assemble a final `_m3` product corpus from its build parts.

Does four things, in this order, per basin:

  1. MERGE the parts on date (the 1980-2013 back-extension + the existing
     2014-2025 areal build). Later parts win on any overlapping date, and the
     overlap is counted and reported rather than assumed to be empty.
  2. SHIFT the named columns by `--shift-days`, so that the row labelled D holds
     the weather of day D.
     ⚠️ For Daymet this is a LEAKAGE fix, not a cosmetic alignment. V4R1 stores at
     date t the rain that physically falls on t+1 (streamflow referee: modal peak
     lag +1, 77% of basins). A decoder predicting day D is handed the row labelled
     D, which natively holds the rain of D+1 -- one day of genuine look-ahead at
     the exact step the model is scored on. Shifting precip +1 removes it. This
     would be required even if CAMELS had never existed.
     Shift PRECIP ONLY: tmin and vp align at r=0.999-1.000, so the file's time
     axis is not uniformly offset -- it is a per-variable day definition -- and
     precip-only is the variant measured at +0.229 NSE.
  3. REFRESH q_cfs from the authoritative store. The 2014-2025 build carries NO
     discharge before 2015 (its builder read a corpus that began in 2015), so a
     naive merge would inherit a year of blank targets.
  4. VERIFY the artifact: contiguous daily calendar, no NaN in any forcing column,
     row count equal to the span. A basin that fails is REFUSED, not written.
"""
from __future__ import annotations
import argparse, gzip, json, os
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORCING_COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]


def read_part(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Rows as {column: value}, NOT positional.

    ⚠️ The parts do NOT share a schema: the 2014-2025 areal build carries 10
    columns (it gained `swe`/`swe_delta` from a later pass) while the 1980-2013
    back-extension writes the 8-column base set. A positional merge silently
    produces ragged rows -- a corrupt corpus that still parses."""
    rows: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        for ln in fh:
            f = ln.rstrip("\n").split(",")
            rows[f[0]] = dict(zip(hdr, f))
    return hdr, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="comma-separated, later wins")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qstore", default="data/modern_corpus/q_store")
    ap.add_argument("--shift-days", type=int, default=0)
    ap.add_argument("--shift-cols", default="precipitation_sum")
    ap.add_argument("--product", required=True)
    ap.add_argument("--ids", default="data/camels_gauge_ids.json")
    ap.add_argument("--key", default="671")
    ap.add_argument("--rationale", default="")
    a = ap.parse_args()
    os.chdir(ROOT)

    parts = [Path(p) for p in a.parts.split(",")]
    for p in parts:
        if not p.exists():
            raise SystemExit(f"GUARD FAIL: part {p} does not exist")
    shift_cols = [c for c in a.shift_cols.split(",") if c] if a.shift_days else []
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    gauges = [str(i).zfill(8) for i in json.loads(Path(a.ids).read_text())[a.key]]

    nok = nbad = nmiss = 0
    overlap_total = 0
    spans: dict[str, int] = {}
    for g in gauges:
        merged: dict[str, dict[str, str]] = {}
        common: set[str] | None = None
        order: list[str] = []
        for p in parts:
            fp = p / f"{g}.csv.gz"
            if not fp.exists():
                continue
            h, rows = read_part(fp)
            # the output schema is the INTERSECTION of the parts, in first-part order
            common = set(h) if common is None else (common & set(h))
            if not order:
                order = list(h)
            overlap_total += len(set(merged) & set(rows))
            merged.update(rows)                     # later part wins
        if not merged:
            nmiss += 1
            continue
        hdr = [c for c in order if c in common]
        if "date" not in hdr or "q_cfs" not in hdr:
            raise SystemExit(f"GUARD FAIL: {g}: parts share no date/q_cfs column")
        dates = sorted(merged)
        # contiguity: the span must be a complete daily calendar
        d0 = dt.date.fromisoformat(dates[0]); d1 = dt.date.fromisoformat(dates[-1])
        want = [(d0 + dt.timedelta(days=k)).isoformat()
                for k in range((d1 - d0).days + 1)]
        if len(want) != len(dates):
            print(f"{g}: {len(want) - len(dates)} missing days in "
                  f"{dates[0]}..{dates[-1]} - REFUSING")
            nbad += 1
            continue

        # shift: new[t] = old[t-1] for the named columns
        if shift_cols:
            for c in shift_cols:
                if c not in hdr:
                    raise SystemExit(f"GUARD FAIL: shift column {c} not in the corpus")
            prev = {c: [merged[d][c] for d in want] for c in shift_cols}
            for k, d in enumerate(want):
                for c in shift_cols:
                    merged[d][c] = prev[c][k - 1] if k > 0 else ""
            want = want[1:]                          # row 0 has no predecessor

        # refresh discharge from the authoritative store
        q = {}
        qp = Path(a.qstore) / f"{g}.csv.gz"
        if qp.exists():
            with gzip.open(qp, "rt") as fh:
                fh.readline()
                for ln in fh:
                    dd, vv = ln.rstrip("\n").split(",")
                    if vv not in ("", "nan"):
                        q[dd] = vv

        bad_cell = False
        out_rows = []
        for d in want:
            row = merged[d]
            row["q_cfs"] = q.get(d, "")
            for c in FORCING_COLS:
                if c in hdr and row.get(c, "") in ("", "nan", "NaN"):
                    bad_cell = True
            out_rows.append(",".join(row.get(c, "") for c in hdr))
        if bad_cell:
            print(f"{g}: NaN in a forcing column - REFUSING")
            nbad += 1
            continue

        op = outdir / f"{g}.csv.gz"
        tmp = op.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt") as fh:
            fh.write(",".join(hdr) + "\n")
            fh.write("\n".join(out_rows) + "\n")
        with gzip.open(tmp, "rt") as fh:             # verify the artifact
            n = sum(1 for _ in fh) - 1
        if n != len(out_rows):
            tmp.unlink(missing_ok=True)
            print(f"{g}: wrote {n} rows, wanted {len(out_rows)} - REFUSING")
            nbad += 1
            continue
        tmp.replace(op)
        spans[f"{want[0]}..{want[-1]}"] = spans.get(f"{want[0]}..{want[-1]}", 0) + 1
        nok += 1

    side = {
        "product": a.product, "parts": [str(p) for p in parts],
        "shift_days": a.shift_days, "shift_cols": shift_cols,
        "shift_rationale": a.rationale or
            ("leakage fix: the native product labels precip one day early, which "
             "would hand the decoder tomorrow's rain at the scored step"),
        "qstore": a.qstore, "n_ok": nok, "n_refused": nbad, "n_missing": nmiss,
        "part_overlap_rows": overlap_total, "spans": spans,
    }
    (outdir / "_BUILD_SIDECAR.json").write_text(json.dumps(side, indent=1))
    print(f"\nASSEMBLE {a.product}: ok={nok} refused={nbad} missing={nmiss} "
          f"overlap_rows={overlap_total} shift={a.shift_days}d on {shift_cols}")
    for s, c in sorted(spans.items(), key=lambda kv: -kv[1]):
        print(f"  span {s}: {c} basins")
    print(f"-> {outdir}")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
