"""The served-forecast archive: the thing that makes "it is live" checkable.

Nothing in this project scores the forecasts the site actually ISSUED against
what the rivers then did -- every skill figure comes from a research window.
These rows are the input to that, so the properties that matter are: the three
series land on IDENTICAL keys (a paired comparison needs identical rows, and
this project has already voided a screen whose arms differed in more than one
way), a re-run cannot double-count, and an empty payload is refused rather
than published as a zero-row day.
"""
import csv
import gzip
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "scripts" / "snapshot_m1_archive.py"
HEADER = ["issued_date", "station_id", "target_date", "horizon_day", "series",
          "q_cfs", "q_lo_raw", "q_hi_raw", "schema_version"]


def _write_shard(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow(HEADER)
        w.writerows(rows)


def _run(sandbox: Path):
    """Run the SANDBOX's copy, not the real one.

    The first version passed the real script path, so `ROOT` resolved to the
    real repo and every assertion was being made about the real archive while
    the sandbox sat untouched -- the harness was testing another checkout.
    It failed loudly here only because the real staging dir did not exist; had
    it existed, these tests would have passed while proving nothing.
    """
    script = sandbox / "scripts" / "snapshot_m1_archive.py"
    assert script.exists(), "sandbox copy missing"
    return subprocess.run([sys.executable, str(script)], cwd=sandbox,
                          capture_output=True, text=True)


@pytest.fixture()
def sandbox(tmp_path):
    """A scratch repo root, so a test can never touch the real archive."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "snapshot_m1_archive.py").write_text(SNAP.read_text())
    return tmp_path


def _rows(issued="2026-10-11", sid="12114500"):
    out = []
    for h in range(1, 15):
        td = f"2026-10-{10 + h:02d}"
        out.append([issued, sid, td, h, "mblstm_modern", 14.8, 10.9, 18.8, "m1-fcst-1"])
        out.append([issued, sid, td, h, "ensemble_blend", 15.2, "", "", "m1-fcst-1"])
        out.append([issued, sid, td, h, "persistence_lag1", 18.5, "", "", "m1-fcst-1"])
    return out


def _read(p: Path):
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def test_consolidates_shards_into_one_file_per_issue_date(sandbox):
    _write_shard(sandbox / "archive_staging_m1" / "shard_0.csv.gz", _rows())
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr
    out = sandbox / "archive_m1" / "2026" / "10" / "2026-10-11.csv.gz"
    assert out.exists()
    assert len(_read(out)) == 42


def test_rerunning_replaces_rather_than_appends(sandbox):
    """A re-run of the same build must not double-count: a day's file is
    authoritative for that day."""
    _write_shard(sandbox / "archive_staging_m1" / "shard_0.csv.gz", _rows())
    _run(sandbox)
    _run(sandbox)
    out = sandbox / "archive_m1" / "2026" / "10" / "2026-10-11.csv.gz"
    assert len(_read(out)) == 42


def test_the_three_series_share_identical_keys(sandbox):
    """Whatever the archive is later used to compare, the comparison must be
    paired on exactly the same (station, target_date, horizon) rows."""
    _write_shard(sandbox / "archive_staging_m1" / "shard_0.csv.gz", _rows())
    _run(sandbox)
    rows = _read(sandbox / "archive_m1" / "2026" / "10" / "2026-10-11.csv.gz")
    keys = {}
    for r in rows:
        keys.setdefault(r["series"], set()).add(
            (r["station_id"], r["target_date"], r["horizon_day"]))
    assert len(keys) == 3
    ref = next(iter(keys.values()))
    for series, ks in keys.items():
        assert ks == ref, f"{series} does not share the common key set"


def test_an_empty_payload_is_refused(sandbox):
    """A zero-row day published as a file would later read as 'the model served
    nothing that day', which is not the same claim as 'the build did not run'."""
    _write_shard(sandbox / "archive_staging_m1" / "shard_0.csv.gz", [])
    r = _run(sandbox)
    assert r.returncode == 1
    assert not (sandbox / "archive_m1").exists() or \
        not list((sandbox / "archive_m1").rglob("*.csv.gz"))


def test_missing_staging_is_refused_not_silently_ok(sandbox):
    r = _run(sandbox)
    assert r.returncode == 1


def test_schema_version_is_restamped_at_consolidation(sandbox):
    """Stamped here, not at shard-write time, so a backfill can always tell
    which generation of the pipeline produced a row."""
    stale = [r[:-1] + ["m1-fcst-OLD"] for r in _rows()]
    _write_shard(sandbox / "archive_staging_m1" / "shard_0.csv.gz", stale)
    _run(sandbox)
    rows = _read(sandbox / "archive_m1" / "2026" / "10" / "2026-10-11.csv.gz")
    assert {r["schema_version"] for r in rows} == {"m1-fcst-1"}


def test_two_issue_dates_in_one_payload_split(sandbox):
    """A build straddling UTC midnight emits two days; they must not merge."""
    _write_shard(sandbox / "archive_staging_m1" / "shard_0.csv.gz",
                 _rows("2026-10-11") + _rows("2026-10-12"))
    _run(sandbox)
    base = sandbox / "archive_m1" / "2026" / "10"
    assert (base / "2026-10-11.csv.gz").exists()
    assert (base / "2026-10-12.csv.gz").exists()
    assert len(_read(base / "2026-10-11.csv.gz")) == 42
