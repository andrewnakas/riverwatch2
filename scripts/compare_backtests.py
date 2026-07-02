#!/usr/bin/env python3
"""Diff two backtest JSONs on the SOTA metric suite — the A/B verdict tool.

Usage:
  .venv/bin/python scripts/compare_backtests.py BASELINE.json CANDIDATE.json [--subset full|camels_531]

Prints a side-by-side of the median metrics with deltas and a PASS/FAIL on the
pre-registered Phase-2 bar: candidate must improve its target metric WITHOUT
regressing the headline NSE/KGE beyond noise. "Higher is better" for
NSE/KGE/log-NSE/r/PICP-toward-0.90; "toward zero" for PBIAS/FHV/FLV; "lower is
better" for approx-CRPS/MPIW.
"""
from __future__ import annotations

import argparse
import json
import sys


def _med(d: dict, subset: str, key: str):
    blk = d.get("metrics", {}).get(subset, {})
    v = blk.get(key)
    return v.get("median") if isinstance(v, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--force", action="store_true",
                    help="compare despite mismatched station subset / window")
    args = ap.parse_args()

    base = json.loads(open(args.baseline).read())
    cand = json.loads(open(args.candidate).read())

    # Comparison guard: a stride-3 candidate scored against the full-set
    # baseline (or mismatched eval windows) produces a meaningless delta —
    # refuse unless --force. Intentional A/B axes (ckpt, forcing plan,
    # anchoring, point policy) are printed so every comparison self-documents.
    hard, soft = [], []
    for key, kind in (("stride_stations", "hard"), ("window", "hard"),
                      ("ckpt", "soft"), ("forcing_plan", "soft"),
                      ("caveat", "soft"), ("anchor_decay", "soft"),
                      ("point_policy", "soft")):
        b, c = base.get(key), cand.get(key)
        if key == "stride_stations":
            b, c = b or 1, c or 1  # legacy JSONs predate the field (= full set)
        if b != c:
            (hard if kind == "hard" else soft).append(f"  {key}: {b!r} -> {c!r}")
    if soft:
        print("A/B axes differing (intentional?):")
        print("\n".join(soft))
    if hard:
        print("MISMATCHED COMPARISON BASIS (delta would be meaningless):")
        print("\n".join(hard))
        if not args.force:
            print("refusing; rerun with --force to override")
            return 1

    # (key, direction): +1 higher-better, -1 lower-better, 0 toward-zero
    spec = [("nse", +1), ("kge", +1), ("log_nse", +1), ("pearson_r", +1),
            ("pct_bias", 0), ("fhv", 0), ("flv", 0),
            ("approx_crps", -1), ("picp90", None), ("mpiw_norm", -1)]

    print(f"subset={args.subset}   baseline={base.get('label')}  candidate={cand.get('label')}")
    print(f"{'metric':>11} {'baseline':>10} {'candidate':>10} {'delta':>9}  verdict")
    nse_d = kge_d = None
    for key, direction in spec:
        b, c = _med(base, args.subset, key), _med(cand, args.subset, key)
        if b is None or c is None:
            continue
        d = c - b
        if key == "picp90":
            # closer to 0.90 is better
            verdict = ("same" if abs(c - 0.90) == abs(b - 0.90)
                       else "better" if abs(c - 0.90) < abs(b - 0.90) else "worse")
        elif direction == 0:
            verdict = ("same" if abs(c) == abs(b)
                       else "better" if abs(c) < abs(b) else "worse")
        elif direction == +1:
            verdict = "better" if d > 0 else ("same" if d == 0 else "WORSE")
        else:
            verdict = "better" if d < 0 else ("same" if d == 0 else "WORSE")
        if key == "nse":
            nse_d = d
        if key == "kge":
            kge_d = d
        print(f"{key:>11} {b:>10.3f} {c:>10.3f} {d:>+9.3f}  {verdict}")

    # Pre-registered bar: don't regress headline NSE/KGE beyond a small noise band.
    NOISE = 0.01
    print()
    if nse_d is not None and kge_d is not None:
        ok = nse_d >= -NOISE and kge_d >= -NOISE
        print(f"HEADLINE GATE: NSE Δ{nse_d:+.3f}, KGE Δ{kge_d:+.3f}  "
              f"→ {'PASS (no headline regression)' if ok else 'FAIL (headline regressed)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
