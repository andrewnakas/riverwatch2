#!/usr/bin/env python
"""CAMELS-US subset report from existing benchmark JSONs (no inference).

For each benchmark JSON (which must carry a "per_station" block), intersect
its stations with the CAMELS-US 671 and Kratzert-531 gauge-id lists from
data/camels_gauge_ids.json and report median NSE / KGE over the overlap.

Usage:
    .venv/bin/python scripts/camels_subset_report.py benchmarks/foo.json [more.json ...]
        [--out benchmarks/camels_subset_report.md]

Read-only with respect to models and benchmarks; only writes the report file.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMELS_PATH = ROOT / "data" / "camels_gauge_ids.json"

METRICS = ("nse", "kge")

CAVEATS = """\
Caveats — read before comparing to published CAMELS numbers:

- Our metrics are pooled-horizon 1-14-day *forecast* NSE/KGE on a 2025
  backtest window. Published CAMELS benchmarks (e.g. Kratzert et al.
  median NSE ~0.74-0.76 on the 531 subset) are *simulation* NSE with
  observed/analysis forcing over the full test period — a much easier task.
- Our train/test protocol is not the CAMELS protocol: different training
  years, different test years (2025), and our corpus is 1,758 CONUS gauges
  chosen for operational relevance, not the CAMELS basin set.
- The perfect-forcing run (ens4ft_perfect) is the only quasi-comparable
  config: it removes weather-forecast error, so it is the closest analog to
  published CAMELS *simulation* benchmarks (still pooled over 14-day
  horizons and still a different train/test split).
- Overlap is partial: only the intersection counts below are scored, not
  the full 531/671 lists, so medians are not computed over the same basin
  population as the published numbers.
"""


def load_camels_ids() -> dict[str, set[str]]:
    data = json.loads(CAMELS_PATH.read_text())
    return {
        k: {str(s).strip().zfill(8) for s in v}
        for k, v in data.items()
        if not k.startswith("_")
    }


def median(vals: list[float]) -> float | None:
    finite = sorted(v for v in vals if isinstance(v, (int, float)) and math.isfinite(v))
    if not finite:
        return None
    n = len(finite)
    mid = n // 2
    return finite[mid] if n % 2 else 0.5 * (finite[mid - 1] + finite[mid])


def subset_stats(per_station: dict, ids: set[str] | None) -> dict:
    sub = per_station if ids is None else {
        sid: r for sid, r in per_station.items() if sid in ids
    }
    out = {"n": len(sub)}
    for m in METRICS:
        out[m] = median([r.get(m) for r in sub.values()])
    return out


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmarks", nargs="+", type=Path,
                    help="benchmark JSON(s) with a per_station block")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "benchmarks" / "camels_subset_report.md",
                    help="markdown output path (default benchmarks/camels_subset_report.md)")
    args = ap.parse_args()

    camels = load_camels_ids()
    subsets: list[tuple[str, set[str] | None]] = [
        ("all", None),
        ("671", camels["671"]),
        ("531", camels["531"]),
    ]

    lines: list[str] = []
    lines.append("# CAMELS-US subset report")
    lines.append("")
    lines.append(f"Gauge-id lists: `{CAMELS_PATH.relative_to(ROOT)}` "
                 f"(671 full CAMELS-US, 531 Kratzert et al. subset).")
    lines.append("")

    header = ("| config | subset | n overlap | median NSE | median KGE |")
    sep = "|---|---|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)

    overlap_note: list[str] = []
    for path in args.benchmarks:
        data = json.loads(path.read_text())
        per_station = data.get("per_station")
        if not per_station:
            raise SystemExit(f"{path}: no per_station block")
        label = data.get("label", path.stem)
        for name, ids in subsets:
            st = subset_stats(per_station, ids)
            lines.append(f"| {label} | {name} | {st['n']} "
                         f"| {fmt(st['nse'])} | {fmt(st['kge'])} |")
        if not overlap_note:  # station set is the same across configs
            n671 = sum(1 for s in camels["671"] if s in per_station)
            n531 = sum(1 for s in camels["531"] if s in per_station)
            overlap_note.append(
                f"Overlap: {n531}/531 and {n671}/671 CAMELS basins are in this "
                f"corpus of {len(per_station)} stations.")

    lines.append("")
    lines.extend(overlap_note)
    lines.append("")
    lines.append(CAVEATS)

    text = "\n".join(lines)
    print(text)
    args.out.write_text(text + "\n")
    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
