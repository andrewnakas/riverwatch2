"""The committed NLDAS basin-weight pickle must be usable by the append.

It is the ONE file that was blocking the secret-gated areal append in CI: the
areal reduction is precomputed, so the job needs numpy + netCDF4 and this
pickle, and nothing else -- no geopandas, no exactextract. It lived only on the
1080 until a reboot made the point that single-machine files are a liability.

These checks are cheap and they pin the two things that would silently corrupt
every basin if wrong: the grid the weights were computed against, and the
row ordering.
"""
import pickle
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "data" / "nldas_basin_weights.pkl"

#: Asserted from the NLDAS-2 documentation by scripts_modern/build_nldas_weights.py
NX, NY, RES = 464, 224, 0.125


def _load():
    if not PKL.exists():
        pytest.skip("weight pickle not committed in this checkout")
    return pickle.loads(PKL.read_bytes())


def test_the_pickle_loads_and_has_the_expected_keys():
    w = _load()
    assert "grid" in w and "weights" in w


def test_the_grid_matches_the_nldas2_grid_it_was_computed_against():
    """A weight pickle computed on a different grid would silently put the
    wrong cells on every basin."""
    g = _load()["grid"]
    assert (g["nx"], g["ny"]) == (NX, NY), (g["nx"], g["ny"])
    assert abs(g["res"] - RES) < 1e-9


def test_row0_is_north_because_the_reducer_flips_on_that():
    """⚠️ NLDAS granules store latitude ASCENDING while these weights are
    north-first, so build_nldas2_areal.reduce_field does np.flipud before
    ravel -- 'or every basin silently gets the wrong cells'. If this ever says
    anything but 'north', that flip is wrong and nothing downstream would
    complain."""
    assert _load()["grid"]["row0"] == "north"


def test_every_camels_basin_has_weights():
    w = _load()["weights"]
    assert len(w) == 671, len(w)
    for b, (cells, cov) in list(w.items())[:20]:
        assert len(cells) == len(cov) and len(cells) > 0, b
        assert all(c >= 0 for c in cov), b


def test_it_stays_small_enough_to_commit():
    """The whole reason the areal append can run in CI rather than depending on
    the box being up."""
    assert PKL.stat().st_size < 512 * 1024, PKL.stat().st_size
