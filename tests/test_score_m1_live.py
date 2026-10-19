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


# ------------------------------------------------- generations may not be pooled

def test_the_scorer_refuses_to_pool_served_generations():
    """The archive's schema_version marks the SERVED CONFIGURATION, and on
    2026-10-17 that changed materially: before it the forcing table was frozen,
    the newest "observed" encoder days were point forecasts up to five days
    old, leads past about 11 were served at the model's TRAINING MEAN, and the
    NLDAS-2 areal append had never run.

    Averaging across that boundary would fold a degraded configuration into the
    first live skill number -- and the first number is the one that gets
    quoted. So the scorer reports the mix and refuses, rather than quietly
    averaging two things that are not the same product.
    """
    src = (ROOT / "scripts" / "score_m1_live.py").read_text()
    assert 'r.get("schema_version")' in src, \
        "the scorer must READ the archive's generation stamp, not just stamp its own"
    assert "REFUSING to pool them" in src
    assert "--allow-mixed-generations" in src
    assert "--generation" in src
    # The refusal must be a non-zero exit, not a warning: a warning in a log
    # nobody reads is how the missing areal-append script cost three days.
    i = src.index("REFUSING to pool them")
    assert "return 1" in src[i:i + 400], \
        "the refusal must exit non-zero, not merely print"


def test_mixed_generations_is_opt_in_and_named_as_a_comparison():
    """The escape hatch exists for a deliberate comparison BETWEEN generations.
    It must not read like a convenience flag."""
    src = (ROOT / "scripts" / "score_m1_live.py").read_text()
    i = src.index("--allow-mixed-generations")
    blurb = src[i:i + 600]
    assert "never for a headline" in blurb


def test_the_live_scorer_still_stamps_its_own_schema():
    """Reading the archive's generation must not have displaced the scorer's
    own output version."""
    src = (ROOT / "scripts" / "score_m1_live.py").read_text()
    assert 'SCHEMA = "m1-live-' in src
    assert '"schema_version": SCHEMA' in src


# ------------------------------------------------------- each tier, separately

def test_the_live_scorer_reports_each_tier():
    """The served domain is 620 basins in two tiers whose measured skill
    differs -- 0.876335 for the 490 trained on, 0.854214 for the 130 GATE 3
    authorised, same window. Pooling them into one live number understates the
    first and overstates the second, and the FIRST live number is the one that
    gets quoted.

    The identical blindness had just been found in the published label, where a
    minority tier three days behind on forcing could not move a pooled median
    at all. 130 of 620 cannot.
    """
    src = (ROOT / "scripts" / "score_m1_live.py").read_text()
    assert 'result["by_tier"]' in src
    assert "m1_cohort_tier2.json" in src
    assert "tier1_trained" in src and "tier2_gate3" in src
    # Reported, not substituted: the pooled figure must survive.
    assert 'result["series"][s]' in src, \
        "the pooled per-series figure must still be reported"


def test_a_basin_in_neither_tier_is_surfaced_not_folded_in():
    """An archived basin in neither cohort file is either a stale archive row
    or a domain change nobody propagated. Quietly counting it in one tier would
    corrupt that tier's number and hide the cause."""
    src = (ROOT / "scripts" / "score_m1_live.py").read_text()
    assert "_basins_in_no_tier" in src
    i = src.index("_basins_in_no_tier")
    assert "served domain changed" in src[i:i + 500]


def test_an_empty_tier_says_so_rather_than_reading_as_zero_skill():
    """Tier 2 was served but never archived until 2026-10-19, so an empty tier
    is a real state. It must not render as a tier that scored badly."""
    src = (ROOT / "scripts" / "score_m1_live.py").read_text()
    i = src.index('"n_pairs": 0')
    assert "no archived rows for this tier" in src[i:i + 300]
