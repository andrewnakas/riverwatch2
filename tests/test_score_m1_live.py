"""The live scorer's arithmetic, on a synthetic archive with a KNOWN answer.

This script will eventually produce the first honest statement about how the
served model does on forecasts this site issued, which makes it the one piece
of code here whose correctness nobody will be able to eyeball from the output.
So its arithmetic is pinned against constructed errors: the member is made
exactly 2.0 cfs wrong, the blend 3.0 and persistence 8.0, and the scorer must
recover those and the paired differences between them exactly.

It also pins the REFUSAL: below a floor of matched pairs the per-basin NSE is
withheld rather than published, because a median over a handful of basin-days
is noise with a decimal point -- and a live NSE is not comparable to 0.905829,
to ~0.858, or to GATE 1's 0.808659 in any case.
"""
import csv
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "scripts" / "score_m1_live.py"
HDR = ["issued_date", "station_id", "target_date", "horizon_day", "series",
       "q_cfs", "q_lo_raw", "q_hi_raw", "schema_version"]

MEMBER_ERR, BLEND_ERR, PERSIST_ERR = 2.0, 3.0, 8.0


def _sandbox(n_basins: int, n_days: int) -> tuple[Path, dict]:
    """A scratch root with a stubbed USGS, so the truth is known and offline."""
    sand = Path(tempfile.mkdtemp())
    (sand / "scripts").mkdir()
    (sand / "scripts" / "score_m1_live.py").write_text(SCORER.read_text())
    (sand / "app").mkdir()
    (sand / "app" / "__init__.py").write_text("")

    basins = [f"{i:08d}" for i in range(1, n_basins + 1)]
    dates = [f"2026-09-{d:02d}" for d in range(1, n_days + 1)]
    rng = np.random.default_rng(0)
    truth = {(b, d): float(50 + 40 * rng.random()) for b in basins for d in dates}

    flat = {f"{b}|{d}": v for (b, d), v in truth.items()}
    (sand / "app" / "usgs.py").write_text(
        "import json\nimport pandas as pd\n"
        f"TRUTH = json.loads({json.dumps(json.dumps(flat))})\n"
        "def fetch_daily_discharge(sid, lo, hi, max_age_hours=0):\n"
        "    rows=[(k.split('|')[1], v) for k,v in TRUTH.items() if k.split('|')[0]==sid]\n"
        "    rows.sort()\n"
        "    return pd.DataFrame({'date':[r[0] for r in rows],"
        " 'q_cfs':[r[1] for r in rows]})\n")

    out = sand / "archive_m1" / "2026" / "09"
    out.mkdir(parents=True)
    per_day = {}
    for b in basins:
        for d in dates:
            t = truth[(b, d)]
            per_day.setdefault(d, []).extend([
                [d, b, d, 1, "mblstm_modern", t + MEMBER_ERR, t - 5, t + 9, "m1-fcst-1"],
                [d, b, d, 1, "ensemble_blend", t + BLEND_ERR, "", "", "m1-fcst-1"],
                [d, b, d, 1, "persistence_lag1", t + PERSIST_ERR, "", "", "m1-fcst-1"],
            ])
    for d, rs in per_day.items():
        with gzip.open(out / f"{d}.csv.gz", "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(HDR)
            w.writerows(rs)
    return sand, truth


def _run(sand: Path):
    r = subprocess.run([sys.executable, "scripts/score_m1_live.py"], cwd=sand,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((sand / "benchmarks" / "m1_live_verification.json").read_text())


def test_pooled_mae_recovers_the_constructed_errors():
    sand, _ = _sandbox(40, 28)
    res = _run(sand)
    s = res["series"]
    assert abs(s["mblstm_modern"]["mae"] - MEMBER_ERR) < 1e-9
    assert abs(s["ensemble_blend"]["mae"] - BLEND_ERR) < 1e-9
    assert abs(s["persistence_lag1"]["mae"] - PERSIST_ERR) < 1e-9


def test_paired_deltas_are_exact_and_signed_the_right_way():
    """Negative means the member is better. Getting this sign backwards would
    invert every conclusion drawn from the archive."""
    sand, _ = _sandbox(40, 28)
    pv = _run(sand)["paired_vs_member"]
    assert abs(pv["ensemble_blend"]["paired_median_mae_delta"]
               - (MEMBER_ERR - BLEND_ERR)) < 1e-9
    assert abs(pv["persistence_lag1"]["paired_median_mae_delta"]
               - (MEMBER_ERR - PERSIST_ERR)) < 1e-9
    assert pv["ensemble_blend"]["paired_median_mae_delta"] < 0
    assert pv["ensemble_blend"]["breadth_member_better"] == 1.0


def test_nse_is_withheld_below_the_pair_floor():
    sand, _ = _sandbox(40, 28)          # 1,120 pairs, under the floor
    res = _run(sand)
    assert "_withheld" in res["per_basin_nse"]
    assert "mblstm_modern" not in res["per_basin_nse"]


def test_nse_is_published_once_there_is_enough():
    sand, _ = _sandbox(80, 28)          # 2,240 pairs, over the floor
    res = _run(sand)
    assert "mblstm_modern" in res["per_basin_nse"]
    assert "_withheld" not in res["per_basin_nse"]
    # And it must carry the non-comparability note wherever it is published.
    assert "not comparable" in res["per_basin_nse"]["mblstm_modern"]["_note"].lower()


def test_only_rows_every_series_has_are_scored():
    """Strictly paired: a row one series is missing must be dropped from all,
    or the comparison is between different samples."""
    sand, _ = _sandbox(40, 28)
    # Remove one series' rows for a single day.
    p = sand / "archive_m1" / "2026" / "09" / "2026-09-05.csv.gz"
    with gzip.open(p, "rt", newline="") as fh:
        rows = list(csv.reader(fh))
    kept = [rows[0]] + [r for r in rows[1:] if r[4] != "mblstm_modern"]
    with gzip.open(p, "wt", newline="") as gz:
        csv.writer(gz).writerows(kept)
    res = _run(sand)
    assert res["n_pairs"] == 40 * 27, "the unpaired day must be dropped entirely"


def test_an_empty_archive_exits_nonzero_rather_than_publishing():
    sand = Path(tempfile.mkdtemp())
    (sand / "scripts").mkdir()
    (sand / "scripts" / "score_m1_live.py").write_text(SCORER.read_text())
    r = subprocess.run([sys.executable, "scripts/score_m1_live.py"], cwd=sand,
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert not (sand / "benchmarks").exists()
