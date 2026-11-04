#!/usr/bin/env python3
"""GATE 4: which forecast product should drive MODERN-1's decoder? (PREREG_v2.md)

Two arms, strictly one variable. For each archived init date D and basin:
  * encoder: the SERVED forcing table (m1-forcing-archive) through D, and USGS
    daily discharge through D -- identical in both arms;
  * decoder, arm "om":  the Open-Meteo point forecast archived at D
    (archive_m1_forcing, lead_day 1..14 = targets D+1..D+14);
  * decoder, arm "gfs": areal GFS from the 00z run of D, GFS lead days 2..15
    (= the same targets D+1..D+14; see the GATE 4 alignment deviation).
Member nldasm9, seeds 501-505, mean, readout (ylo+yhi)/2 -- app.mblstm_modern
exactly as served. No retraining.

    python analysis/gate4_run.py predict --om-dir A --gfs-dir G --tables T --out preds.csv.gz
    python analysis/gate4_run.py score preds.csv.gz [--allow-thin]

`score` REFUSES to print any skill number below MIN_INIT_DATES (30) init dates:
the registration forbids reading earlier. Below that it reports only coverage
and the no-op check (do the two arms' decoder inputs, and outputs, differ?) --
the check GATE 2 needed to rule out a silent splice that reads as a null.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("RW2_ENABLE_MBLSTM_MODERN", "1")
from app import camels_attrs, mblstm_modern as M, usgs  # noqa: E402

COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
MIN_INIT_DATES = 30
HORIZON = 14


def _om_arm(om_dir: str, d: str) -> pd.DataFrame:
    f = glob.glob(f"{om_dir}/**/{d}.csv.gz", recursive=True)
    if not f:
        return pd.DataFrame()
    df = pd.read_csv(f[0], dtype={"station_id": str})
    df = df[(df.lead_day >= 1) & (df.lead_day <= HORIZON)]
    return df.assign(date=pd.to_datetime(df.target_date))[["station_id", "date"] + COLS]


def _gfs_arm(gfs_dir: str, d: str) -> pd.DataFrame:
    p = Path(gfs_dir) / f"{d.replace('-', '')}.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"station_id": str})
    df = df[(df.lead_day >= 2) & (df.lead_day <= HORIZON + 1)]
    init = pd.Timestamp(d)
    return df.assign(date=init + pd.to_timedelta(df.lead_day - 1, unit="D"))[["station_id", "date"] + COLS]


def predict(a) -> int:
    assert M._try_load(), M._load_reason
    inits = sorted(Path(p).name[:10] for p in glob.glob(f"{a.om_dir}/**/*.csv.gz", recursive=True))
    inits = [d for d in inits if (Path(a.gfs_dir) / f"{d.replace('-', '')}.csv.gz").exists()]
    print(f"{len(inits)} init dates with both arms")
    tables, qs, rows = {}, {}, []
    for d in inits:
        D = pd.Timestamp(d)
        arms = {"om": _om_arm(a.om_dir, d), "gfs": _gfs_arm(a.gfs_dir, d)}
        basins = sorted(set(arms["om"].station_id) & set(arms["gfs"].station_id))
        for sid in basins[: a.limit or None]:
            if sid not in tables:
                p = Path(a.tables) / f"{sid}.csv.gz"
                tables[sid] = pd.read_csv(p, parse_dates=["date"]) if p.exists() else None
                qs[sid] = usgs.fetch_daily_discharge(sid, date(2024, 9, 1), date.today())
            t, q = tables[sid], qs[sid]
            if t is None or q is None or q.empty:
                continue
            qh = q[pd.to_datetime(q["date"]) <= D]
            if qh.empty or pd.Timestamp(qh["date"].iloc[-1]) != D:
                continue          # encoder must end at D in both arms, or neither
            enc = t[t.date <= D][["date"] + COLS]
            for arm, f in arms.items():
                dec = f[f.station_id == sid][["date"] + COLS]
                out = M.forecast(qh, enc, dec, {"id": sid}, HORIZON)
                for i, r in enumerate(out or []):
                    rows.append((sid, d, r["date"], i + 1, arm, r["q_cfs"]))
        print(f"  {d}: {len(rows)} rows so far", flush=True)
    pd.DataFrame(rows, columns=["station_id", "init_date", "target_date", "lead", "arm", "q"]) \
        .to_csv(a.out, index=False, compression="gzip")
    print(f"-> {a.out}")
    return 0


def _nse(sim, obs):
    d = np.sum((obs - obs.mean()) ** 2)
    return np.nan if d <= 0 else 1 - np.sum((sim - obs) ** 2) / d


def score(a) -> int:
    p = pd.read_csv(a.preds, dtype={"station_id": str})
    n_init = p.init_date.nunique()
    w = p.pivot_table(index=["station_id", "init_date", "target_date", "lead"],
                      columns="arm", values="q").dropna().reset_index()
    # Rows where BOTH arms clip to 0 cfs (dry basins) are identical by
    # construction and say nothing about whether the splice took; exclude them.
    nz = (w.gfs > 0) | (w.om > 0)
    moved = float((~np.isclose(w.gfs[nz], w.om[nz], rtol=1e-6)).mean())
    w["tier"] = np.where(w.station_id.isin(camels_attrs.cohort()), 1, 2)
    print(f"init dates {n_init}, paired rows {len(w)}, basins {w.station_id.nunique()}; "
          f"rows where the arms' outputs differ: {moved:.4f}")
    if moved < 0.5:
        print("⛔ the arms barely differ -- a silent no-op splice reads as a null. Check inputs.")
    if n_init < MIN_INIT_DATES and not a.allow_thin:
        print(f"⛔ {n_init} < {MIN_INIT_DATES} init dates: the registration forbids reading "
              "skill this early. Coverage only.")
        return 0
    obs = {}
    for s in w.station_id.unique():
        o = usgs.fetch_daily_discharge(s, date(2026, 10, 1), date.today())
        obs[s] = dict(zip(o["date"].astype(str), o["q_cfs"]))
    w["obs"] = [obs[s].get(t) for s, t in zip(w.station_id, w.target_date)]
    w = w.dropna(subset=["obs"])
    rng = np.random.default_rng(0)
    for tier, wt in w.groupby("tier"):
        print(f"\n=== tier {tier} {'(HEADLINE: the registered decision reads this table)' if tier == 1 else '(reported separately; NOT part of the decision)'}")
        _table(wt, rng)
    return 0


def _table(w, rng) -> None:
    print("lead  median_dNSE(gfs-om)  CI95  breadth  basins")
    for L, g in w.groupby("lead"):
        per = g.groupby("station_id").apply(
            lambda b: _nse(b.gfs.values, b.obs.values) - _nse(b.om.values, b.obs.values)).dropna()
        v = per.values
        boot = [np.median(rng.choice(v, len(v))) for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{L:>4}  {np.median(v):+.6f}  [{lo:+.6f}, {hi:+.6f}]  {np.mean(v > 0):.3f}  {len(v)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    pr = sp.add_parser("predict")
    pr.add_argument("--om-dir", required=True)
    pr.add_argument("--gfs-dir", required=True)
    pr.add_argument("--tables", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--limit", type=int, default=0, help="basins per init (smoke test)")
    sc = sp.add_parser("score")
    sc.add_argument("preds")
    sc.add_argument("--allow-thin", action="store_true",
                    help="NOT for GATE 4's verdict; debugging the scorer only")
    a = ap.parse_args()
    return predict(a) if a.cmd == "predict" else score(a)


if __name__ == "__main__":
    raise SystemExit(main())
