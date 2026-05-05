#!/usr/bin/env python3
"""v14.5c-bf validation: diff fleet/per-shard MAE before vs after the
SNODAS history restore. Compares benchmarks/baseline_v14p5c-bf_pre-history.json
(delta-only) to the live deployed index_summary.json (post-history).

Usage:
    python scripts/compare_snodas_history_mae.py
    python scripts/compare_snodas_history_mae.py --post path/to/post.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "benchmarks" / "baseline_v14p5c-bf_pre-history.json"
LIVE = "https://andrewnakas.github.io/riverwatch2/index_summary.json"


def _load(src: str) -> dict:
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=30) as r:
            return json.loads(r.read())
    return json.loads(Path(src).read_text())


def _delta(pre: float | None, post: float | None) -> str:
    if pre is None or post is None:
        return "n/a"
    d = post - pre
    pct = 100.0 * d / pre if pre else 0.0
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:8.2f} cfs ({sign}{pct:+.2f}%)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pre", default=str(PRE))
    p.add_argument("--post", default=LIVE)
    args = p.parse_args()

    pre = _load(args.pre)
    post = _load(args.post)

    print(f"PRE  {pre.get('generated_at')}  ({pre.get('stations_total')} stations)")
    print(f"POST {post.get('generated_at')}  ({post.get('stations_total')} stations)")
    print()
    print("Fleet rolling MAE (cfs):")
    print(f"  blend (mean h):     pre {pre.get('rolling_mae_blend_mean'):>9.2f}   post {post.get('rolling_mae_blend_mean'):>9.2f}   Δ {_delta(pre.get('rolling_mae_blend_mean'), post.get('rolling_mae_blend_mean'))}")
    print(f"  blend h7:           pre {pre.get('rolling_mae_blend_h7_mean'):>9.2f}   post {post.get('rolling_mae_blend_h7_mean'):>9.2f}   Δ {_delta(pre.get('rolling_mae_blend_h7_mean'), post.get('rolling_mae_blend_h7_mean'))}")
    print(f"  blend h14:          pre {pre.get('rolling_mae_blend_h14_mean'):>9.2f}   post {post.get('rolling_mae_blend_h14_mean'):>9.2f}   Δ {_delta(pre.get('rolling_mae_blend_h14_mean'), post.get('rolling_mae_blend_h14_mean'))}")

    pre_m = pre.get("rolling_mae_mean_by_member") or {}
    post_m = post.get("rolling_mae_mean_by_member") or {}
    members = sorted(set(pre_m) | set(post_m))
    print()
    print("Per-member rolling MAE (cfs):")
    for m in members:
        pv = pre_m.get(m)
        qv = post_m.get(m)
        pv_s = f"{pv:>9.2f}" if isinstance(pv, (int, float)) else "       n/a"
        qv_s = f"{qv:>9.2f}" if isinstance(qv, (int, float)) else "       n/a"
        print(f"  {m:<20s}  pre {pv_s}   post {qv_s}   Δ {_delta(pv, qv)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
