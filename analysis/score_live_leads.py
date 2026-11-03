#!/usr/bin/env python3
"""Per-lead live scoring of the SERVED blend against persistence (PREREG §LIVE-1).

    git fetch origin m1-archive
    mkdir -p /tmp/arch && for f in $(git ls-tree -r --name-only origin/m1-archive archive_m1 | grep csv.gz); do
        git show origin/m1-archive:$f > /tmp/arch/$(basename $f); done
    python analysis/score_live_leads.py /tmp/arch --issued-from 2026-11-04

Reports, per lead: pooled MAE blend/persistence, fraction of basins where the
blend's MAE is lower, and the median relative bias (pred-obs)/obs of each.
All days scored, no stride. Observations from USGS via app.usgs (cached).
"""
import argparse
import glob
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import usgs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_dir")
    ap.add_argument("--generation", default="m1-fcst-2")
    ap.add_argument("--issued-from", default="2000-01-01")
    ap.add_argument("--issued-to", default="2100-01-01")
    a = ap.parse_args()
    df = pd.concat(pd.read_csv(f, dtype={"station_id": str})
                   for f in glob.glob(a.archive_dir + "/*.csv.gz"))
    df = df[(df.schema_version == a.generation)
            & (df.issued_date >= a.issued_from) & (df.issued_date <= a.issued_to)]
    lo = date.fromisoformat(df.target_date.min()) - timedelta(days=2)
    obs = {}
    for s in df.station_id.unique():
        o = usgs.fetch_daily_discharge(s, lo, date.today())
        obs[s] = dict(zip(o["date"].astype(str), o["q_cfs"]))
    df["obs"] = [obs[s].get(t) for s, t in zip(df.station_id, df.target_date)]
    df = df.dropna(subset=["obs"])
    key = ["station_id", "target_date", "horizon_day"]
    p = (df.pivot_table(index=key, columns="series", values="q_cfs")
         .join(df.groupby(key).obs.first())
         .dropna(subset=["ensemble_blend", "persistence_lag1"]).reset_index())
    print(f"rows {len(p)}  basins {p.station_id.nunique()}  "
          f"targets {p.target_date.min()}..{p.target_date.max()}")
    print("lead rows mae_ratio frac_blend_better bias_blend bias_pers")
    for h, g in p.groupby("horizon_day"):
        eb, ep = (g.ensemble_blend - g.obs).abs(), (g.persistence_lag1 - g.obs).abs()
        pb = g.assign(eb=eb, ep=ep).groupby("station_id")[["eb", "ep"]].mean()
        d = g.obs.clip(lower=0.1)
        print(h, len(g), round(eb.sum() / ep.sum(), 3), round(float((pb.eb < pb.ep).mean()), 3),
              round(float(((g.ensemble_blend - g.obs) / d).median()), 3),
              round(float(((g.persistence_lag1 - g.obs) / d).median()), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
