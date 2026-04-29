#!/usr/bin/env python3
"""Merge per-shard `dist/` outputs into a single `dist/` for Pages deploy.

Each shard uploads its `dist/` directory as an artifact. The deploy job
downloads them all into ./shard_dists/<shard_name>/ and then this script
reconciles them:

  - Asset files (index.html, static/, stations.json) come from shard 0
  - Forecast JSONs are concatenated across all shards into dist/forecasts/
  - Per-shard summaries (index_summary_shard_<n>.json) are merged into one
    top-level dist/index_summary.json with totals + worst-case build time
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True,
                    help="directory containing shard subdirs from artifact downloads")
    args = ap.parse_args()

    in_root = Path(args.input_root)
    if not in_root.exists():
        print(f"input dir not found: {in_root}")
        return 1

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    (DIST / "forecasts").mkdir(parents=True)

    shard_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])
    print(f"merging {len(shard_dirs)} shard dirs from {in_root}")

    # 1) Asset shard (the one with index.html). Take the first that has it.
    asset_shard = None
    for sd in shard_dirs:
        if (sd / "index.html").exists():
            asset_shard = sd
            break
    if asset_shard is None:
        print("ERROR: no shard contains index.html")
        return 1
    for item in asset_shard.iterdir():
        if item.name in {"forecasts"} or item.name.startswith("index_summary_shard_"):
            continue
        dst = DIST / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    print(f"  assets from {asset_shard.name}")

    # 2) Forecast JSONs from every shard
    total_forecasts = 0
    for sd in shard_dirs:
        fcst = sd / "forecasts"
        if not fcst.exists():
            continue
        for f in fcst.iterdir():
            if f.suffix == ".json":
                shutil.copy2(f, DIST / "forecasts" / f.name)
                total_forecasts += 1
    print(f"  total forecasts merged: {total_forecasts}")

    # 3) Merge summaries
    shard_summaries = []
    for sd in shard_dirs:
        for s in sd.glob("index_summary_shard_*.json"):
            try:
                shard_summaries.append(json.loads(s.read_text()))
            except Exception as exc:
                print(f"  bad summary {s}: {exc}")
    shard_summaries.sort(key=lambda s: s.get("shard_id", 0))

    total_in_shards = sum(s.get("stations_in_shard", 0) for s in shard_summaries)
    total_succeeded = sum(s.get("stations_succeeded", 0) for s in shard_summaries)
    failed_ids: list[str] = []
    for s in shard_summaries:
        failed_ids.extend(s.get("stations_failed", []))
    longest_shard = max((s.get("build_seconds", 0) for s in shard_summaries), default=0)

    member_pool: dict[str, list[float]] = {}
    for s in shard_summaries:
        for m, v in (s.get("rolling_mae_mean_by_member") or {}).items():
            if v is not None:
                member_pool.setdefault(m, []).append(float(v))

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stations_total": total_in_shards,
        "stations_succeeded": total_succeeded,
        "stations_failed": failed_ids,
        "rolling_mae_mean_by_member": {
            m: (sum(vs) / len(vs)) if vs else None for m, vs in member_pool.items()
        },
        "build_seconds": longest_shard,
        "shards": len(shard_summaries),
        "shard_summaries": shard_summaries,
    }
    (DIST / "index_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  summary: {total_succeeded}/{total_in_shards} succeeded across {len(shard_summaries)} shards, longest shard {longest_shard}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
