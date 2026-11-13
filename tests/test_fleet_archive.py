"""Fleet verification archive (PREREG §V16-0).

The properties that matter: every lead of every gauge lands on one row with a
fixed column order; a member the gauge does not have is an EMPTY cell, never a
zero (a zero would be scored as a forecast of a dry river); provenance carries
the age of the weather the forecast ran on; a re-run replaces a day rather than
double-counting; and an empty payload is refused.
"""
import csv
import gzip
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import fleet_archive as fa  # noqa: E402


def _fcst(sid="01010000", issued="2026-11-13T05:00:00+00:00", n=14):
    dates = [f"2026-11-{12 + h:02d}" for h in range(n)]
    return SimpleNamespace(
        station_id=sid, issued_at=issued,
        history=[{"date": "2026-11-11", "q_cfs": 1540.0}],
        blend=[{"date": d, "q_cfs": 1500.123456 + i, "band_lo": 0.0, "band_hi": 2000.0}
               for i, d in enumerate(dates)],
        members={"persistence_lag1": [{"date": d, "q_cfs": 1540.0} for d in dates],
                 "nwm": [{"date": d, "q_cfs": float("nan")} for d in dates]},
        nwm_raw_forecast=[{"date": dates[0], "q_cfs": 1394.6}],
        blend_strategy_per_h={h: "lgbm_stacker" for h in range(1, n + 1)},
        provenance={"wx_fcst_start": "2026-10-11", "wx_fcst_end": "2026-10-27",
                    "wx_hist_end": "2026-10-10", "nbm_start": None},
        data_age_days=2, stale=False, degraded=False, chosen="nwm_residual",
    )


def test_one_row_per_lead_with_fixed_columns():
    rows, meta = fa.rows_for(_fcst())
    assert len(rows) == 14 and all(len(r) == len(fa.HEADER) for r in rows)
    assert len(meta) == len(fa.META_HEADER)
    r = dict(zip(fa.HEADER, rows[0]))
    assert (r["issued_date"], r["target_date"], r["horizon_day"]) == ("2026-11-13", "2026-11-12", 1)
    assert r["blend"] == "1500" and r["rule"] == "lgbm_stacker"
    assert r["persistence_lag1"] == "1540" and r["nwm_raw"] == "1395"


def test_missing_or_nonfinite_member_is_empty_not_zero():
    rows, _ = fa.rows_for(_fcst())
    r = dict(zip(fa.HEADER, rows[3]))
    assert r["mblstm"] == "" and r["nwm"] == "" and r["nwm_raw"] == "" and r["challenger"] == ""
    assert r["band_lo"] == "0"  # a real zero stays a zero


def test_provenance_row_carries_input_age():
    _, meta = fa.rows_for(_fcst())
    m = dict(zip(fa.META_HEADER, meta))
    assert m["last_obs_date"] == "2026-11-11" and m["wx_fcst_start"] == "2026-10-11"
    assert m["nbm_start"] == "" and m["data_age_days"] == 2


def test_input_provenance_never_raises_and_reads_edges():
    from app.forecast import _input_provenance
    hist = pd.DataFrame({"date": pd.to_datetime(["2026-10-09", "2026-10-10", "2026-10-11"]),
                         "precipitation_sum": [1.0, 0.0, None]})
    fc = pd.DataFrame({"date": ["2026-10-11", "2026-10-27"], "precipitation_sum": [0, 0]})
    p = _input_provenance(hist, fc, pd.DataFrame(columns=["date"]))
    assert p == {"wx_fcst_start": "2026-10-11", "wx_fcst_end": "2026-10-27",
                 "wx_hist_end": "2026-10-10", "nbm_start": None}
    assert _input_provenance(None, object(), 3) == dict.fromkeys(p)



@pytest.fixture()
def sandbox(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    for rel in ("scripts/snapshot_fleet_archive.py", "app/fleet_archive.py"):
        (tmp_path / rel).write_text((ROOT / rel).read_text())
    return tmp_path


def _stage(sandbox, forecasts, shard=0):
    rows, metas = [], []
    for f in forecasts:
        r, m = fa.rows_for(f)
        rows += r
        metas.append(m)
    for sub, hdr, payload in (("_fleet", fa.HEADER, rows), ("_fleet_meta", fa.META_HEADER, metas)):
        d = sandbox / "archive_staging_fleet" / sub
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / f"shard_{shard}.csv.gz", "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(hdr)
            w.writerows(payload)


def _run(sandbox):
    return subprocess.run([sys.executable, str(sandbox / "scripts" / "snapshot_fleet_archive.py")],
                          cwd=sandbox, capture_output=True, text=True)


def _read(path):
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.reader(fh))


def test_snapshot_consolidates_and_rerun_replaces(sandbox):
    _stage(sandbox, [_fcst("01010000")], shard=0)
    _stage(sandbox, [_fcst("02020000")], shard=1)
    assert _run(sandbox).returncode == 0
    out = sandbox / "archive_fleet" / "2026" / "11" / "2026-11-13.csv.gz"
    got = _read(out)
    assert got[0] == fa.HEADER and len(got) == 1 + 28
    meta = _read(sandbox / "archive_fleet_meta" / "2026" / "11" / "2026-11-13.csv.gz")
    assert meta[0] == fa.META_HEADER and len(meta) == 3
    assert _run(sandbox).returncode == 0
    assert len(_read(out)) == 1 + 28, "a re-run must replace the day, not append"


def test_snapshot_refuses_empty_payload(sandbox):
    _stage(sandbox, [])
    r = _run(sandbox)
    assert r.returncode == 1 and not (sandbox / "archive_fleet").exists()
