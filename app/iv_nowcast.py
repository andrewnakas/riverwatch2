"""Close the daily-value latency gap with USGS instantaneous values.

PREREG §V16 H-LAG. A build's lead 1 is the day after the last USGS *daily*
value, which is typically yesterday or the day before, while instantaneous
values (IV) run to the present. A pre-matrix job (`scripts/build_iv_nowcast.py`)
reduces the last three days of IV to per-gauge LOCAL-day means and hands every
shard one small table; `extend()` appends the complete days a gauge's daily
record does not have yet.

Only COMPLETE local days are used (>= MIN_COVERAGE_H hours of samples spanning
the day), never today's partial day: a mean over the first hours of a rising
limb is not a daily mean. The appended rows are in-memory only -- the persistent
USGS record cache is never written, so the official daily value replaces them
when it posts. Off unless RW2_IV_NOWCAST=1.
"""
from __future__ import annotations

import io
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

TABLE = Path(__file__).resolve().parents[1] / "data" / "iv_nowcast" / "nowcast.json"
MIN_COVERAGE_H = 22.0
MIN_SAMPLES = 20
_table: Optional[Dict[str, Dict[str, float]]] = None


def enabled() -> bool:
    return os.environ.get("RW2_IV_NOWCAST") == "1"


def daily_means_from_rdb(text: str) -> Dict[str, Dict[str, float]]:
    """IV RDB (one block per column layout) -> {site: {local_date: mean cfs}},
    complete days only. `datetime` is already station-local."""
    out: Dict[str, Dict[str, float]] = {}
    block: list = []

    def flush() -> None:
        if len(block) < 3:
            return
        df = pd.read_csv(io.StringIO("\n".join([block[0]] + block[2:])), sep="\t", dtype=str)
        col = next((c for c in df.columns if c.endswith("_00060")), None)
        if col is None or "site_no" not in df or "datetime" not in df:
            return
        df = pd.DataFrame({"site": df["site_no"], "t": pd.to_datetime(df["datetime"], errors="coerce"),
                           "q": pd.to_numeric(df[col], errors="coerce")}).dropna()
        df = df[df.q >= 0]
        if df.empty:
            return
        df["d"] = df.t.dt.date
        g = df.groupby(["site", "d"]).agg(q=("q", "mean"), n=("q", "size"),
                                          t0=("t", "min"), t1=("t", "max"))
        span_h = (g.t1 - g.t0).dt.total_seconds() / 3600.0
        g = g[(g.n >= MIN_SAMPLES) & (span_h >= MIN_COVERAGE_H)]
        for (site, d), r in g.iterrows():
            out.setdefault(site, {}).setdefault(d.isoformat(), round(float(r.q), 3))

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            block = []
        elif line.strip():
            block.append(line)
    flush()
    return out


def _load() -> Dict[str, Dict[str, float]]:
    global _table
    if _table is None:
        try:
            _table = json.loads(TABLE.read_text()).get("sites", {})
        except Exception:
            _table = {}
    return _table


def extend(site_no: str, q_hist: pd.DataFrame, today: Optional[date] = None,
           table: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """Append IV-derived daily means for the days right after `q_hist`'s last
    row, contiguously, up to yesterday. Returns `q_hist` unchanged when there
    is nothing to add. Never raises."""
    try:
        days = (table if table is not None else _load()).get(str(site_no))
        if not days or q_hist is None or q_hist.empty:
            return q_hist
        today = today or date.today()
        last = pd.to_datetime(q_hist["date"].iloc[-1]).date()
        add = []
        d = last + timedelta(days=1)
        while d < today and d.isoformat() in days:
            add.append({"date": d, "q_cfs": float(days[d.isoformat()])})
            d += timedelta(days=1)
        if not add:
            return q_hist
        extra = pd.DataFrame(add)
        if pd.api.types.is_datetime64_any_dtype(q_hist["date"]):
            extra["date"] = pd.to_datetime(extra["date"])
        out = pd.concat([q_hist, extra], ignore_index=True)
        out.attrs["iv_days"] = len(add)
        return out
    except Exception:
        return q_hist
