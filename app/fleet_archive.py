"""Fleet verification archive rows (PREREG §V16-0, generation `fleet-1`).

One row per (gauge, lead) with everything the build SERVED, plus one
provenance row per gauge saying how old its inputs were. Pure functions over a
`StationForecast`-shaped object, so the schema is unit-testable without a build.

The m1 archive (`scripts/snapshot_m1_archive.py`) covers the 620 MODERN-1
basins in long format and is left untouched: PREREG §LIVE-1 and GATE 4 read it.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

SCHEMA_VERSION = "fleet-1"

#: Fixed column order. A member a gauge does not have is an empty cell, never 0.
MEMBERS = (
    "persistence_lag1", "runoff_ridge", "chronos_bolt", "ttm", "timesfm",
    "timesfm_xreg", "lgbm_pooled", "nwm", "nwm_residual", "mblstm",
    "mblstm_modern",
)
HEADER = (["issued_date", "station_id", "target_date", "horizon_day",
           "blend", "band_lo", "band_hi", "challenger", "rule"]
          + list(MEMBERS) + ["nwm_raw", "schema_version"])
META_HEADER = ["issued_date", "issued_at", "station_id", "last_obs_date",
               "q_obs_last", "data_age_days", "wx_fcst_start", "wx_fcst_end",
               "wx_hist_end", "nbm_start", "iv_days", "wx_hist_rows", "stale", "degraded", "chosen",
               "schema_version"]


def _num(v: Any) -> str:
    """4 significant digits: forecast error is orders larger, and the archive
    is ~136k rows a day on a git branch."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:.4g}"


def _by_date(pts: Optional[List[dict]]) -> dict:
    return {str(p.get("date")): p for p in (pts or [])}


def rows_for(f: Any) -> Tuple[List[list], list]:
    """(lead rows, provenance row) for one served forecast."""
    issued = str(f.issued_at)[:10]
    sid = str(f.station_id)
    members = {m: _by_date((f.members or {}).get(m)) for m in MEMBERS}
    nwm_raw = _by_date(getattr(f, "nwm_raw_forecast", None))
    chal = _by_date(getattr(f, "challenger", None))
    rules = getattr(f, "blend_strategy_per_h", None) or {}
    rows: List[list] = []
    for h, pt in enumerate(f.blend or [], start=1):
        td = str(pt.get("date"))
        rows.append(
            [issued, sid, td, h, _num(pt.get("q_cfs")), _num(pt.get("band_lo")),
             _num(pt.get("band_hi")), _num((chal.get(td) or {}).get("q_cfs")),
             str(rules.get(h, rules.get(str(h), "")))]
            + [_num((members[m].get(td) or {}).get("q_cfs")) for m in MEMBERS]
            + [_num((nwm_raw.get(td) or {}).get("q_cfs")), SCHEMA_VERSION])
    prov = getattr(f, "provenance", None) or {}
    hist = getattr(f, "history", None) or []
    last = hist[-1] if hist else {}
    meta = [issued, str(f.issued_at), sid, str(last.get("date", "")),
            _num(last.get("q_cfs")),
            "" if getattr(f, "data_age_days", None) is None else int(f.data_age_days),
            prov.get("wx_fcst_start") or "", prov.get("wx_fcst_end") or "",
            prov.get("wx_hist_end") or "", prov.get("nbm_start") or "",
            prov.get("iv_days") or "0",
            prov.get("wx_hist_rows") or "0",
            int(bool(getattr(f, "stale", False))),
            int(bool(getattr(f, "degraded", False))),
            str(getattr(f, "chosen", "") or ""), SCHEMA_VERSION]
    return rows, meta
