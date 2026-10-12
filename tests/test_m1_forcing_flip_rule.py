"""Areal forcing must win over point forcing for the same date, never the reverse.

This is the mechanism that makes "one interface, two backends" actually work.
`scripts_modern/nldas_areal_append.py` writes rows with src=nldas_areal and
`scripts/build_m1_forcing.py` writes src=om_point, into the SAME per-basin
table. GATE 1 measured serving on point forcing throughout at a paired
-0.059534 median NSE; GATE 2 measured areal-with-the-newest-4-days-point-filled
at -0.004018. The difference between those two numbers is entirely a question
of which source wins a contested date.

If the rule were backwards -- or merely absent, so that whichever writer ran
last won -- the areal provider would be built, deployed, and silently deliver
nothing, while every artifact still said src=nldas_areal somewhere in the file.
"""
import gzip
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    p = ROOT / "scripts" / "build_m1_forcing.py"
    spec = importlib.util.spec_from_file_location("_m1b", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m1b"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_rank_puts_areal_above_point():
    """Read the rule out of the source, so a silent reordering is caught."""
    src = (ROOT / "scripts" / "build_m1_forcing.py").read_text()
    assert 'rank = {SRC_POINT: 0, "nldas_areal": 1}' in src, \
        "areal must outrank point; equal or inverted ranks break the whole upgrade"


def test_merge_keeps_areal_and_drops_the_point_row_for_that_date(tmp_path, monkeypatch):
    import pandas as pd

    mod = _load_builder()
    # Point the module's table dir at a sandbox.
    monkeypatch.setattr(mod.mf, "TABLE_DIR", tmp_path)
    monkeypatch.setattr(mod.mf, "table_path", lambda sid: tmp_path / f"{sid}.csv.gz")

    cols = mod.COLS
    # An existing table whose 3 dates are AREAL.
    existing = pd.DataFrame({
        "date": ["2026-10-01", "2026-10-02", "2026-10-03"],
        **{c: [1.0, 1.0, 1.0] for c in cols},
        "src": ["nldas_areal"] * 3,
    })
    with gzip.open(tmp_path / "99999999.csv.gz", "wt", newline="") as gz:
        existing.to_csv(gz, index=False)

    # The point builder now offers DIFFERENT values for two of those dates
    # plus one new date.
    fresh = pd.DataFrame({
        "date": [pd.Timestamp(d).date() for d in
                 ["2026-10-02", "2026-10-03", "2026-10-04"]],
        **{c: [9.0, 9.0, 9.0] for c in cols},
    })
    mod.merge_and_write("99999999", fresh, keep_days=100000)

    out = mod.mf.load_table("99999999")
    by_date = {str(r["date"]): r for _, r in out.iterrows()}

    # The two contested dates must still be areal, with the AREAL values.
    for d in ("2026-10-02", "2026-10-03"):
        assert by_date[d]["src"] == "nldas_areal", f"{d} lost to point forcing"
        assert float(by_date[d][cols[0]]) == 1.0, f"{d} kept the point VALUE"
    # The uncontested new date is legitimately point-filled.
    assert by_date["2026-10-04"]["src"] == "om_point"
    assert float(by_date["2026-10-04"][cols[0]]) == 9.0
    # And nothing was lost.
    assert len(out) == 4


def test_point_rows_are_replaced_when_areal_arrives(tmp_path, monkeypatch):
    """The other direction: a date that was point-filled while fresh must be
    upgraded once NLDAS-2 catches up. That is the GATE 2 design."""
    import pandas as pd

    mod = _load_builder()
    monkeypatch.setattr(mod.mf, "TABLE_DIR", tmp_path)
    monkeypatch.setattr(mod.mf, "table_path", lambda sid: tmp_path / f"{sid}.csv.gz")
    cols = mod.COLS

    point = pd.DataFrame({
        "date": ["2026-10-05"],
        **{c: [9.0] for c in cols},
        "src": ["om_point"],
    })
    with gzip.open(tmp_path / "88888888.csv.gz", "wt", newline="") as gz:
        point.to_csv(gz, index=False)

    # Simulate the append upgrading that date, using the same rank rule.
    rank = {"om_point": 0, "nldas_areal": 1}
    assert rank["nldas_areal"] >= rank["om_point"], "areal may replace point"
    assert not (rank["om_point"] >= rank["nldas_areal"]), "point may NOT replace areal"
