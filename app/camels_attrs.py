"""Lazy loader for the 27 Addor-2017 CAMELS static attributes.

The MODERN-1 record members were trained with `--static-set camels`, i.e. the
27-attribute Addor set, NOT the 14 registry+GAGES-II features that
`app/mblstm.py:STATIC_FEATS` uses for the production member. Those attributes
live in `data/camels_attrs.json` (671 basins, built by
`scripts/build_camels_attrs.py` from the CAMELS v2.0 attribute tables).

Same lazy-load-and-cache shape as `app/gages2.py`, and the same failure
contract: a missing file or a missing basin degrades to an empty dict rather
than raising, because `app/mblstm.py:static_vector()` median-imputes any
attribute the dict does not carry (from `cfg["static_median"]`, which is the
training-set median, so an imputed value is the same one training saw).

Coverage, measured 2026-10-09: all 27 attributes present for 490/490 basins of
the MODERN-1 cohort, except `root_depth_50` which is missing for 12 basins and
therefore median-imputed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
ATTRS_PATH = ROOT / "data" / "camels_attrs.json"
COHORT_PATH = ROOT / "data" / "m1_cohort.json"
#: TIER 2 -- CAMELS basins the member was NOT trained on, authorised by GATE 3
#: and served under their OWN measured median (0.854214 on val1 over 130
#: basins) rather than the cohort's. Kept in a SEPARATE file from the training
#: cohort on purpose: the two have different measured skill, and a single list
#: would make it impossible for the label to tell a reader which number applies
#: to the basin in front of them.
TIER2_PATH = ROOT / "data" / "m1_cohort_tier2.json"

_attrs: Optional[dict] = None
_cohort: Optional[frozenset] = None
_tier2: Optional[frozenset] = None


def _load_attrs() -> dict:
    global _attrs
    if _attrs is None:
        try:
            _attrs = json.loads(ATTRS_PATH.read_text())
        except Exception:
            _attrs = {}
    return _attrs


def cohort() -> frozenset:
    """The 490 basin ids the MODERN-1 members were trained and scored on.

    Empty frozenset if the file is unreadable, which makes `in_cohort()` False
    for everything and so disables the member rather than serving it out of
    domain.
    """
    global _cohort
    if _cohort is None:
        try:
            _cohort = frozenset(json.loads(COHORT_PATH.read_text())["cohort"])
        except Exception:
            _cohort = frozenset()
    return _cohort


def tier2() -> frozenset:
    """The GATE 3 out-of-cohort basins the member may serve, labelled.

    Empty on any failure, exactly as `cohort()` is: the fallback must be to
    serve FEWER basins, never to serve one whose skill was never measured.
    """
    global _tier2
    if _tier2 is None:
        try:
            _tier2 = frozenset(json.loads(TIER2_PATH.read_text())["cohort"])
        except Exception:
            _tier2 = frozenset()
    return _tier2


def in_cohort(station_id: str) -> bool:
    """TIER 1 only -- the basins the member was trained on.

    Deliberately unchanged in meaning. Callers that need "may the member run
    here" want `in_served_domain`; callers that need "was this basin trained
    on" want this. Widening `in_cohort` to cover tier 2 would have silently
    re-pointed every existing caller, including the ones that pick which
    measured number to quote.
    """
    return str(station_id) in cohort()


def in_served_domain(station_id: str) -> bool:
    """May the member run on this basin at all -- tier 1 or tier 2."""
    sid = str(station_id)
    return sid in cohort() or sid in tier2()


def domain_tier(station_id: str) -> int:
    """1 = trained on, 2 = GATE 3 out-of-cohort, 0 = out of domain.

    The number a label quotes depends on this, so it is returned explicitly
    rather than inferred by the caller from two membership tests.
    """
    sid = str(station_id)
    if sid in cohort():
        return 1
    if sid in tier2():
        return 2
    return 0


def attrs_for(station_id: str) -> dict:
    """The 27 Addor attributes for one basin, or {} if absent."""
    return dict(_load_attrs().get(str(station_id), {}))


def enrich_station_attrs(station_id: str, base: Optional[dict] = None) -> dict:
    """Merge the Addor attributes onto a station registry record.

    The registry keys (lat/lon/drain_area_sqmi/...) are kept because
    `static_vector()` reads `drain_area_sqmi` for `log_drain_area`, and other
    members in the blend read lat/lon off the same dict. CAMELS attributes win
    on a key collision: they are what the checkpoint's normalizers were fitted
    against.
    """
    out = dict(base or {})
    out.update(attrs_for(station_id))
    return out
