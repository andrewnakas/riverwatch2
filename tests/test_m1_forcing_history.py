"""The forecast forcing must be archived as it passes, or a comparison between
decoder products can never be made.

Areal GFS forecast forcing can be rebuilt from S3 for any past date. Open-Meteo's
forecast cannot -- its API serves the current run -- and `m1-forcing-archive` is
a single-commit force-pushed branch, so yesterday's forecast rows are gone the
moment today's land. Unless they are captured per build, the paired
"which decoder product is better" question is permanently unanswerable, not
merely expensive.
"""
import csv
import gzip
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "scripts" / "snapshot_m1_forcing_history.py"
CHANNELS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]


def _sandbox(n_basins=3, past=400, fut=14, src="om_point"):
    sand = Path(tempfile.mkdtemp())
    (sand / "scripts").mkdir()
    (sand / "scripts" / "snapshot_m1_forcing_history.py").write_text(SNAP.read_text())
    td = sand / "data" / "m1_forcing"
    td.mkdir(parents=True)
    today = date.today()
    for i in range(n_basins):
        sid = f"0000000{i}"
        with gzip.open(td / f"{sid}.csv.gz", "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(["date"] + CHANNELS + ["src"])
            for d in range(past, 0, -1):
                w.writerow([(today - timedelta(days=d)).isoformat()] + [1.0] * 6 + ["nldas_areal"])
            w.writerow([today.isoformat()] + [1.0] * 6 + [src])
            for d in range(1, fut + 1):
                w.writerow([(today + timedelta(days=d)).isoformat()] + [2.0] * 6 + [src])
    # a label file must be ignored, not treated as a basin
    (td / "_served_skill.json").write_text("{}")
    return sand


def _run(sand):
    return subprocess.run([sys.executable, "scripts/snapshot_m1_forcing_history.py"],
                          cwd=sand, capture_output=True, text=True)


def _read(sand):
    fs = list((sand / "archive_m1_forcing").rglob("*.csv.gz"))
    assert len(fs) == 1, fs
    with gzip.open(fs[0], "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def test_captures_every_basin_and_lead():
    sand = _sandbox(n_basins=3, fut=14)
    assert _run(sand).returncode == 0
    rows = _read(sand)
    assert len(rows) == 3 * 14
    assert sorted({int(r["lead_day"]) for r in rows}) == list(range(1, 15))
    assert len({r["station_id"] for r in rows}) == 3


def test_observed_history_is_excluded():
    """It is reproducible from the corpora, and including it would multiply the
    volume for nothing."""
    sand = _sandbox(past=400, fut=5)
    _run(sand)
    rows = _read(sand)
    today = date.today().isoformat()
    assert all(r["target_date"] > today for r in rows)
    assert len(rows) == 3 * 5


def test_lead_day_is_relative_to_the_init_date():
    sand = _sandbox(fut=3)
    _run(sand)
    for r in _read(sand):
        lead = (date.fromisoformat(r["target_date"])
                - date.fromisoformat(r["init_date"])).days
        assert int(r["lead_day"]) == lead


def test_the_label_file_is_not_mistaken_for_a_basin():
    sand = _sandbox()
    _run(sand)
    assert all(not r["station_id"].startswith("_") for r in _read(sand))


def test_src_is_recorded_so_the_product_is_identifiable():
    """The whole point is comparing products, so a row that does not say which
    product it came from is useless."""
    sand = _sandbox(src="om_point")
    _run(sand)
    assert {r["src"] for r in _read(sand)} == {"om_point"}


def test_a_table_with_no_forecast_rows_is_refused_not_published_empty():
    sand = _sandbox(fut=0)
    r = _run(sand)
    assert r.returncode == 1
    assert not (sand / "archive_m1_forcing").exists() or \
        not list((sand / "archive_m1_forcing").rglob("*.csv.gz"))


def test_missing_tables_dir_is_refused():
    sand = Path(tempfile.mkdtemp())
    (sand / "scripts").mkdir()
    (sand / "scripts" / "snapshot_m1_forcing_history.py").write_text(SNAP.read_text())
    assert _run(sand).returncode == 1
