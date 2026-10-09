#!/usr/bin/env python
"""STAGE 4b — measure the skill of the map's area-ratio transfer.

The click-anywhere feature estimates runoff at an ungauged point by taking the
nearest comparable gauge's hydrograph and scaling it by the drainage-area ratio.
That is a real hydrological method with real limits, so the UI must state its
measured skill rather than imply the model ran at the clicked point. This script
produces the number it states (`data/river_transfer_skill.json`).

HOW IT IS MEASURED. For gauge pairs that USGS NLDI places on the same mainstem,
transfer the DONOR's OBSERVED daily discharge to the TARGET by area ratio and
score NSE against the target's own observed record. Observed-to-observed isolates
the transfer: it is the ceiling the method can reach, with no forecast error
mixed in. Results are binned by |log(area ratio)| because that is the quantity
the UI can show a user before they trust a number.

WHY NOT the forecast-to-observed case too: that would confound the transfer with
the donor's forecast error, which is already reported separately per gauge. The
UI says what this measures -- the transfer step only.

⚠️ This is a NEIGHBOUR-TRANSFER skill, not ungauged-basin model skill. It needs a
gauge on the same river. A clicked point with no comparable gauge gets no number
at all, which is what `select_donor` returning None means.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.river_transfer import area_ratio_penalty, sqmi_to_km2, transfer_by_area_ratio  # noqa: E402
from app.metrics import nse  # noqa: E402

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{}/navigation/{}/nwissite"
# |log ratio| cut points: 1.25x, 2x, 4x, 10x.
BINS = [(math.log(1.25), "1.25x"), (math.log(2.0), "2x"),
        (math.log(4.0), "4x"), (math.log(10.0), "10x")]


def _get(url: str, tries: int = 3, pause: float = 0.4):
    import urllib.error
    import urllib.request
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "riverwatch2-transfer-study"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(pause * (2 ** k))
        except Exception:
            time.sleep(pause * (2 ** k))
    return None


def neighbours(gid: str, distance_km: int, pause: float) -> list[tuple[str, str]]:
    """(neighbour_id, direction) for gauges up/down the same mainstem."""
    out = []
    for code, name in (("UM", "upstream"), ("DM", "downstream")):
        j = _get(NLDI.format(gid, code) + f"?distance={distance_km}")
        time.sleep(pause)
        for f in ((j or {}).get("features") or []):
            ident = (f.get("properties") or {}).get("identifier") or ""
            nid = ident.replace("USGS-", "")
            if nid and nid != gid:
                out.append((nid, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="data/stations_v15.json")
    ap.add_argument("--out", default="data/river_transfer_skill.json")
    ap.add_argument("--cache", default="", help="override the USGS record cache dir")
    ap.add_argument("--sample", type=int, default=220)
    ap.add_argument("--distance-km", type=int, default=100)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--min-days", type=int, default=400)
    ap.add_argument("--max-ratio", type=float, default=10.0)
    ap.add_argument("--pause", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import app.usgs as usgs
    if a.cache:
        usgs.RECORDS_DIR = Path(a.cache)
        usgs.RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    sts = json.loads((ROOT / a.stations).read_text())["stations"]
    by_id = {s["id"]: s for s in sts}
    pool = [s for s in sts if (s.get("drain_area_sqmi") or 0) > 0]
    rng = np.random.default_rng(a.seed)          # fixed: the sample is not a free parameter
    idx = rng.choice(len(pool), size=min(a.sample, len(pool)), replace=False)
    sample = [pool[i] for i in sorted(idx)]
    print(f"sampling {len(sample)} of {len(pool)} gauges with a drainage area", flush=True)

    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    q_cache: dict = {}

    def q_of(gid):
        if gid not in q_cache:
            try:
                df = usgs.fetch_daily_discharge(gid, d0, d1)
                q_cache[gid] = None if df is None or df.empty else df
            except Exception:
                q_cache[gid] = None
        return q_cache[gid]

    pairs, seen = [], set()
    for i, s in enumerate(sample, 1):
        gid = s["id"]
        for nid, direction in neighbours(gid, a.distance_km, a.pause):
            nb = by_id.get(nid)
            if not nb or not (nb.get("drain_area_sqmi") or 0) > 0:
                continue
            key = tuple(sorted((gid, nid)))
            if key in seen:
                continue
            seen.add(key)
            ta = sqmi_to_km2(s["drain_area_sqmi"])
            da = sqmi_to_km2(nb["drain_area_sqmi"])
            pen = area_ratio_penalty(ta, da)
            if pen is None or pen > math.log(a.max_ratio):
                continue
            pairs.append({"target": gid, "donor": nid, "direction": direction,
                          "target_km2": ta, "donor_km2": da, "abs_log_ratio": pen})
        if i % 25 == 0:
            print(f"  [{i}/{len(sample)}] candidate pairs so far: {len(pairs)}", flush=True)
    print(f"{len(pairs)} candidate pairs within {a.max_ratio}x area", flush=True)

    rows = []
    for k, p in enumerate(pairs, 1):
        tq, dq = q_of(p["target"]), q_of(p["donor"])
        if tq is None or dq is None:
            continue
        t = tq.set_index("date")["q_cfs"]
        d = dq.set_index("date")["q_cfs"]
        both = t.index.intersection(d.index)
        if len(both) < a.min_days:
            continue
        obs = t.loc[both].to_numpy(dtype=float)
        don = d.loc[both].to_numpy(dtype=float)
        m = np.isfinite(obs) & np.isfinite(don)
        if m.sum() < a.min_days:
            continue
        sim = np.asarray(transfer_by_area_ratio(don[m].tolist(), p["donor_km2"], p["target_km2"]),
                         dtype=float)
        try:
            v = float(nse(obs[m], sim))
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        rows.append(dict(p, n_days=int(m.sum()), nse=v))
        if k % 25 == 0:
            print(f"  scored {len(rows)} / tried {k}", flush=True)

    if not rows:
        print("no scorable pairs", flush=True)
        return 1

    def summarize(sel):
        v = np.array([r["nse"] for r in sel], dtype=float)
        return {"n_pairs": len(sel), "median_nse": float(np.median(v)),
                "mean_nse": float(np.mean(v)),
                "frac_above_0": float((v > 0).mean()),
                "frac_above_0.5": float((v > 0.5).mean()),
                "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}

    bins = []
    for cut, label in BINS:
        sel = [r for r in rows if r["abs_log_ratio"] <= cut]
        if len(sel) >= 10:
            b = summarize(sel)
            b["max_abs_log_ratio"] = cut
            b["label"] = label
            bins.append(b)

    out = {
        "_note": ("Skill of the map's drainage-area-ratio transfer, measured "
                  "observed-to-observed between gauges NLDI places on the same "
                  "mainstem. This is the TRANSFER step only -- it excludes the "
                  "donor's forecast error, and it is NOT ungauged-model skill."),
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "window": [a.start, a.end],
        "method": "Q_target = Q_donor * (A_target / A_donor); NSE vs target observed",
        "n_gauges_sampled": len(sample),
        "overall": summarize(rows),
        "bins": bins,
        "by_direction": {
            d: summarize([r for r in rows if r["direction"] == d])
            for d in ("upstream", "downstream")
            if len([r for r in rows if r["direction"] == d]) >= 10
        },
        "pairs": rows,
    }
    (ROOT / a.out).write_text(json.dumps(out, indent=1))

    o = out["overall"]
    print(f"\nscored pairs      : {o['n_pairs']}")
    print(f"median NSE        : {o['median_nse']:.4f}   (mean {o['mean_nse']:.4f})")
    print(f"frac NSE > 0      : {o['frac_above_0']:.3f}   > 0.5: {o['frac_above_0.5']:.3f}")
    for b in bins:
        print(f"  within {b['label']:>5s} area: n={b['n_pairs']:4d}  median NSE {b['median_nse']:.4f}"
              f"  >0.5 {b['frac_above_0.5']:.3f}")
    for d, v in (out["by_direction"] or {}).items():
        print(f"  {d:11s}: n={v['n_pairs']:4d}  median NSE {v['median_nse']:.4f}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
