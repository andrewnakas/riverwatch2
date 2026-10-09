"""Unit tests for analysis/river_transfer.py -- the click-anywhere arithmetic.

These pin the values that `app/static/river_transfer.js` must reproduce. The
`test` job in pages.yml runs pytest only, so this file is the CI-gated copy of
the math; the JS mirror is cross-checked against the fixture emitted by
`test_emit_js_crosscheck_fixture`.
"""
import json
import math
from pathlib import Path

import pytest

from analysis import river_transfer as rt

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- distance

def test_haversine_known_distance():
    # 1 degree of latitude is ~111.19 km on a sphere of this radius.
    assert rt.haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, rel=0.001)
    # A degree of longitude shrinks with cos(lat).
    assert rt.haversine_km(60.0, 0.0, 60.0, 1.0) == pytest.approx(55.6, rel=0.01)


def test_haversine_zero():
    assert rt.haversine_km(44.6, -67.9, 44.6, -67.9) == pytest.approx(0.0, abs=1e-9)


def test_point_on_polyline_is_zero():
    line = [[-68.0, 44.0], [-67.0, 44.0]]
    assert rt.point_to_polyline_km(44.0, -67.5, line) == pytest.approx(0.0, abs=0.02)


def test_point_off_polyline_perpendicular():
    """0.1 deg of latitude off a due-east line is ~11.06 km."""
    line = [[-68.0, 44.0], [-67.0, 44.0]]
    d = rt.point_to_polyline_km(44.1, -67.5, line)
    assert d == pytest.approx(11.06, rel=0.02)


def test_point_beyond_segment_end_clamps_to_vertex():
    """Past the end of a segment the nearest point is the endpoint, not the
    infinite line -- otherwise a click far off the end of a reach would read as
    being on it."""
    line = [[-68.0, 44.0], [-67.0, 44.0]]
    d = rt.point_to_polyline_km(44.0, -66.0, line)          # 1 deg east of the end
    assert d == pytest.approx(rt.haversine_km(44.0, -66.0, 44.0, -67.0), rel=0.02)


def test_polyline_empty_and_single_point():
    assert rt.point_to_polyline_km(44.0, -67.0, []) is None
    assert rt.point_to_polyline_km(44.0, -67.0, [[-67.0, 44.1]]) == pytest.approx(11.12, rel=0.02)


def test_polyline_takes_the_nearest_of_many_segments():
    line = [[-68.0, 44.0], [-67.0, 44.0], [-67.0, 45.0]]
    # Nearest to the second (north-going) segment.
    assert rt.point_to_polyline_km(44.5, -66.99, line) < 2.0


# ----------------------------------------------------------------------- area

def test_one_degree_box_at_equator():
    """A 1x1 degree box on the equator is ~12,363 km^2."""
    geom = {"type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    assert rt.geojson_area_km2(geom) == pytest.approx(12363.0, rel=0.01)


def test_area_shrinks_with_latitude():
    lo = rt.geojson_area_km2({"type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]})
    hi = rt.geojson_area_km2({"type": "Polygon",
        "coordinates": [[[0, 60], [1, 60], [1, 61], [0, 61], [0, 60]]]})
    assert hi < lo
    assert hi / lo == pytest.approx(math.cos(math.radians(60.5)), rel=0.02)


def test_area_is_orientation_independent():
    cw = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
    ccw = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    assert rt.geojson_area_km2(cw) == pytest.approx(rt.geojson_area_km2(ccw), rel=1e-9)


def test_hole_is_subtracted():
    outer = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    hole = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25], [0.25, 0.25]]
    solid = rt.geojson_area_km2({"type": "Polygon", "coordinates": [outer]})
    holed = rt.geojson_area_km2({"type": "Polygon", "coordinates": [outer, hole]})
    assert holed == pytest.approx(solid * 0.75, rel=0.01)


def test_multipolygon_sums():
    a = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
    b = [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]
    one = rt.geojson_area_km2({"type": "Polygon", "coordinates": a})
    two = rt.geojson_area_km2({"type": "MultiPolygon", "coordinates": [a, b]})
    assert two == pytest.approx(2 * one, rel=0.01)


@pytest.mark.parametrize("geom", [
    None, {}, {"type": "Point", "coordinates": [0, 0]},
    {"type": "Polygon", "coordinates": []},
])
def test_area_rejects_non_areal(geom):
    assert rt.geojson_area_km2(geom) is None


# ------------------------------------------------------------------ unit math

def test_cfs_to_mm_per_day_reference_value():
    """1 cfs over 1 km^2 for a day = 2.4466 mm."""
    assert rt.cfs_to_mm_per_day(1.0, 1.0) == pytest.approx(2.446575, rel=1e-5)


def test_mm_per_day_roundtrip():
    for q, a in ((50.9, 588.0), (1.0, 1.0), (12345.0, 2500.0)):
        mm = rt.cfs_to_mm_per_day(q, a)
        assert rt.mm_per_day_to_cfs(mm, a) == pytest.approx(q, rel=1e-9)


def test_specific_discharge_is_scale_free():
    """The same runoff depth in two catchments of different size -- this is the
    property that makes area-ratio transfer meaningful at all."""
    mm_a = rt.cfs_to_mm_per_day(100.0, 100.0)
    mm_b = rt.cfs_to_mm_per_day(200.0, 200.0)
    assert mm_a == pytest.approx(mm_b, rel=1e-12)


@pytest.mark.parametrize("area", [0.0, -5.0, None])
def test_unit_math_rejects_bad_area(area):
    assert rt.cfs_to_mm_per_day(10.0, area) is None
    assert rt.mm_per_day_to_cfs(10.0, area) is None


def test_sqmi_conversion():
    assert rt.sqmi_to_km2(1.0) == pytest.approx(2.58998811, rel=1e-9)
    assert rt.sqmi_to_km2(227.0) == pytest.approx(587.93, rel=1e-4)
    assert rt.sqmi_to_km2(None) is None


# ----------------------------------------------------------- donor selection

def test_area_ratio_penalty_is_symmetric():
    assert rt.area_ratio_penalty(100.0, 100.0) == pytest.approx(0.0)
    assert rt.area_ratio_penalty(200.0, 100.0) == pytest.approx(
        rt.area_ratio_penalty(100.0, 200.0), rel=1e-12)


def test_donor_prefers_closest_area_match():
    cands = [
        {"id": "far", "area_km2": 2000.0},
        {"id": "near", "area_km2": 520.0},
        {"id": "small", "area_km2": 60.0},
    ]
    assert rt.select_donor(cands, 500.0)["id"] == "near"


def test_donor_has_no_directional_preference():
    """Measured, not assumed: over 322 pairs upstream donors reached median NSE
    0.6880 vs downstream 0.7459, so the old upstream bonus is unsupported. The
    contrast is unpaired, so a downstream bonus is equally unsupported --
    direction is simply not used, and area ratio decides."""
    cands = [
        {"id": "down", "area_km2": 500.0, "direction": "downstream"},
        {"id": "up", "area_km2": 440.0, "direction": "upstream"},
    ]
    assert rt.select_donor(cands, 500.0)["id"] == "down"      # exact area match wins
    flipped = [
        {"id": "down", "area_km2": 440.0, "direction": "downstream"},
        {"id": "up", "area_km2": 500.0, "direction": "upstream"},
    ]
    assert rt.select_donor(flipped, 500.0)["id"] == "up"      # and again, on area alone


def test_direction_can_still_be_weighted_if_a_paired_study_justifies_it():
    """The parameter survives so a PAIRED measurement could set it later."""
    cands = [
        {"id": "down", "area_km2": 500.0, "direction": "downstream"},
        {"id": "up", "area_km2": 440.0, "direction": "upstream"},
    ]
    assert rt.select_donor(cands, 500.0, upstream_bonus=0.5)["id"] == "up"


def test_donor_area_match_beats_a_bad_match_in_either_direction():
    for bad_dir, good_dir in (("upstream", "downstream"), ("downstream", "upstream")):
        cands = [
            {"id": "good", "area_km2": 500.0, "direction": good_dir},
            {"id": "bad", "area_km2": 50.0, "direction": bad_dir},
        ]
        assert rt.select_donor(cands, 500.0)["id"] == "good"


def test_donor_refuses_when_everything_is_too_far_off():
    """A refusal is honest; a 20x extrapolation is not."""
    assert rt.select_donor([{"id": "x", "area_km2": 10.0}], 500.0) is None
    assert rt.select_donor([], 500.0) is None
    assert rt.select_donor([{"id": "x"}], 500.0) is None


def test_donor_distance_only_breaks_ties():
    cands = [
        {"id": "a", "area_km2": 500.0, "distance_km": 80.0},
        {"id": "b", "area_km2": 500.0, "distance_km": 3.0},
    ]
    assert rt.select_donor(cands, 500.0)["id"] == "b"


# ------------------------------------------------------------------- transfer

def test_transfer_scales_linearly():
    out = rt.transfer_by_area_ratio([100.0, 50.0], donor_km2=200.0, target_km2=100.0)
    assert out == pytest.approx([50.0, 25.0])


def test_transfer_identity_when_areas_match():
    series = [1.0, 2.0, 3.0]
    assert rt.transfer_by_area_ratio(series, 123.4, 123.4) == pytest.approx(series)


def test_transfer_preserves_specific_discharge():
    """The defining property: the transferred hydrograph has the donor's runoff
    depth, which is the entire assumption the method makes."""
    donor_q, donor_a, target_a = 120.0, 300.0, 75.0
    tq = rt.transfer_by_area_ratio([donor_q], donor_a, target_a)[0]
    assert rt.cfs_to_mm_per_day(tq, target_a) == pytest.approx(
        rt.cfs_to_mm_per_day(donor_q, donor_a), rel=1e-12)


def test_transfer_passes_through_none_gaps():
    out = rt.transfer_by_area_ratio([10.0, None, 30.0], 100.0, 100.0)
    assert out[1] is None and out[0] == pytest.approx(10.0)


@pytest.mark.parametrize("d,t", [(0.0, 10.0), (10.0, 0.0), (None, 10.0), (10.0, None)])
def test_transfer_rejects_bad_areas(d, t):
    assert rt.transfer_by_area_ratio([1.0], d, t) is None


# ------------------------------------------------ fixture for the JS mirror

def test_emit_js_crosscheck_fixture(tmp_path):
    """Emit the canonical input->output pairs the JS implementation must match.

    Written to the repo so it is reviewable and diffable; the JS is checked
    against it by hand/node, NOT by this gate. That limitation is deliberate and
    stated: pages.yml runs pytest only.
    """
    box = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    fixture = {
        "_note": ("Canonical values for app/static/river_transfer.js. Produced by "
                  "tests/test_river_transfer.py::test_emit_js_crosscheck_fixture "
                  "from analysis/river_transfer.py. The JS must reproduce these "
                  "to within 1e-6 relative."),
        "haversine_km": [
            {"args": [0, 0, 1, 0], "want": rt.haversine_km(0, 0, 1, 0)},
            {"args": [60, 0, 60, 1], "want": rt.haversine_km(60, 0, 60, 1)},
            {"args": [44.608, -67.935, 44.7, -68.0], "want": rt.haversine_km(44.608, -67.935, 44.7, -68.0)},
        ],
        "point_to_polyline_km": [
            {"lat": 44.0, "lon": -67.5, "coords": [[-68.0, 44.0], [-67.0, 44.0]],
             "want": rt.point_to_polyline_km(44.0, -67.5, [[-68.0, 44.0], [-67.0, 44.0]])},
            {"lat": 44.1, "lon": -67.5, "coords": [[-68.0, 44.0], [-67.0, 44.0]],
             "want": rt.point_to_polyline_km(44.1, -67.5, [[-68.0, 44.0], [-67.0, 44.0]])},
            {"lat": 44.0, "lon": -66.0, "coords": [[-68.0, 44.0], [-67.0, 44.0]],
             "want": rt.point_to_polyline_km(44.0, -66.0, [[-68.0, 44.0], [-67.0, 44.0]])},
        ],
        "geojson_area_km2": [
            {"geometry": {"type": "Polygon", "coordinates": [box]},
             "want": rt.geojson_area_km2({"type": "Polygon", "coordinates": [box]})},
            {"geometry": {"type": "MultiPolygon", "coordinates": [[box]]},
             "want": rt.geojson_area_km2({"type": "MultiPolygon", "coordinates": [[box]]})},
        ],
        "cfs_to_mm_per_day": [
            {"args": [1.0, 1.0], "want": rt.cfs_to_mm_per_day(1.0, 1.0)},
            {"args": [50.9, 587.93], "want": rt.cfs_to_mm_per_day(50.9, 587.93)},
        ],
        "sqmi_to_km2": [{"args": [227.0], "want": rt.sqmi_to_km2(227.0)}],
        "transfer_by_area_ratio": [
            {"series": [100.0, 50.0], "donor_km2": 200.0, "target_km2": 100.0,
             "want": rt.transfer_by_area_ratio([100.0, 50.0], 200.0, 100.0)},
        ],
        "select_donor": [
            {"candidates": [{"id": "far", "area_km2": 2000.0},
                            {"id": "near", "area_km2": 520.0}],
             "target_km2": 500.0, "want": "near"},
            {"candidates": [{"id": "down", "area_km2": 500.0, "direction": "downstream"},
                            {"id": "up", "area_km2": 440.0, "direction": "upstream"}],
             "target_km2": 500.0, "want": "down"},
            {"candidates": [{"id": "x", "area_km2": 10.0}],
             "target_km2": 500.0, "want": None},
        ],
    }
    out = ROOT / "tests" / "river_transfer_fixture.json"
    out.write_text(json.dumps(fixture, indent=1))
    assert out.exists()
