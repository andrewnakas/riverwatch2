"""Unit tests for USGS NWIS daily-value parsing (app.usgs._parse_dv).

This is the front door for all observed discharge; bad parsing silently
corrupts every downstream member (AUDIT Phase 2 / Phase 5)."""
import pandas as pd

from app.usgs import _parse_dv


def _payload(values):
    """Wrap a list of (dateTime, value[, qualifiers]) into NWIS dv JSON shape."""
    entries = []
    for item in values:
        dt, val = item[0], item[1]
        entry = {"dateTime": dt, "value": val}
        if len(item) > 2:
            entry["qualifiers"] = item[2]
        entries.append(entry)
    return {"value": {"timeSeries": [{"values": [{"value": entries}]}]}}


def test_parse_basic():
    df = _parse_dv(_payload([
        ("2024-01-01T00:00:00", "100.0"),
        ("2024-01-02T00:00:00", "150.5"),
    ]))
    assert list(df["q_cfs"]) == [100.0, 150.5]
    assert len(df) == 2


def test_parse_drops_negative_flows():
    df = _parse_dv(_payload([
        ("2024-01-01T00:00:00", "100.0"),
        ("2024-01-02T00:00:00", "-999.0"),  # USGS no-data sentinel
    ]))
    assert list(df["q_cfs"]) == [100.0]


def test_parse_skips_malformed_values():
    df = _parse_dv(_payload([
        ("2024-01-01T00:00:00", "100.0"),
        ("2024-01-02T00:00:00", "Ice"),     # non-numeric
        ("2024-01-03T00:00:00", None),      # null
    ]))
    assert list(df["q_cfs"]) == [100.0]


def test_parse_dedupes_and_sorts_by_date():
    df = _parse_dv(_payload([
        ("2024-01-03T00:00:00", "30.0"),
        ("2024-01-01T00:00:00", "10.0"),
        ("2024-01-01T00:00:00", "11.0"),   # duplicate date
    ]))
    dates = [str(d) for d in df["date"]]
    assert dates == ["2024-01-01", "2024-01-03"]
    assert df["q_cfs"].iloc[0] in (10.0, 11.0)  # one kept, deterministic order


def test_parse_empty_payload_returns_empty_frame():
    df = _parse_dv({"value": {"timeSeries": []}})
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["date", "q_cfs"]
