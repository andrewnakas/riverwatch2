"""Guards the AUDIT Phase 3 regime-aware backtest windows. The whole point of
Phase 3 is that validation (weights, rule selection, conformal bands) spans
multiple seasons/flow regimes rather than only the recent ~8 months, so we pin
that the offset tuples actually reach out-of-regime and stay consistent."""
from app.forecast import _FOUNDATION_HOLDOUT_OFFSETS, _RIDGE_HOLDOUT_OFFSETS


def test_offsets_reach_out_of_regime():
    # At least one window must end more than a year back so weights/bands are
    # validated across seasons, not just the trailing few months.
    assert max(_FOUNDATION_HOLDOUT_OFFSETS) >= 365
    assert any(off >= 365 for off in _RIDGE_HOLDOUT_OFFSETS)


def test_offsets_are_unique_and_sorted_nonneg():
    for offsets in (_FOUNDATION_HOLDOUT_OFFSETS, _RIDGE_HOLDOUT_OFFSETS):
        assert list(offsets) == sorted(offsets)
        assert len(set(offsets)) == len(offsets)
        assert all(o >= 0 for o in offsets)


def test_ridge_offsets_subset_of_foundation():
    # Ridge MAEs are compared apples-to-apples against foundation members, so
    # every ridge window must also be a foundation window.
    assert set(_RIDGE_HOLDOUT_OFFSETS).issubset(set(_FOUNDATION_HOLDOUT_OFFSETS))
