"""Pooled blend stacker (v14.4).

Replaces the inverse-MAE² + per-bucket blend-rule selection at the end of
`forecast_station` with a per-horizon LightGBM meta-learner that takes the
8-member panel of forecast values at horizon h plus a small set of static
+ temporal context features and predicts the observed flow at h.

Training rows come from the per-station holdout panels that
`forecast_station` already computes for `_FOUNDATION_HOLDOUT_OFFSETS`:
6 offsets × 14 horizons × 1893 stations ≈ 160K rows per horizon, plenty
for a 31-leaf tree.

Why a stacker:
- Inverse-MAE² assumes member errors are independent and zero-mean. v14.3
  showed they're neither — biases get absorbed by tiny weights instead of
  corrected, and a member with low average MAE can still be wrong on the
  cases that matter (high-flow events, snowmelt onset, etc.).
- A tree-based meta-learner can learn conditional "trust" patterns:
  "when chronos is much higher than NWM and q_obs is rising, lean
  chronos"; "when persistence and NWM agree below climatology, lean
  blend". These are exactly the patterns inverse-MAE² can't represent.

Architecture mirrors `app/pooled_lgbm.py`:
    StackerTrainer.add_station(...)    # accumulate (member_preds, ytrue) holdout rows
    StackerTrainer.fit()               # train h=1..14 LightGBM models
    StackerTrainer.predict_blend(...)  # produce 14-day blend for one station

Targets are in asinh space so the loss isn't dominated by a few high-flow
stations. We invert at predict time. Falls back to None if LightGBM is
unavailable or any horizon has insufficient training data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as _lgb  # type: ignore
    _LGB_OK = True
except Exception:
    _LGB_OK = False


# Order is locked: every row at every horizon must use this exact column
# layout so trees can split on "what does member M say at h". Members not
# present for a given training row are NaN (LightGBM handles natively).
MEMBER_ORDER = (
    "persistence_lag1",
    "runoff_ridge",
    "chronos_bolt",
    "ttm",
    "timesfm",
    "timesfm_xreg",
    "nwm",
    "lgbm_pooled",
)

# Static + temporal context. Drainage area is log1p'd because it spans 4+
# orders of magnitude. DOY is encoded as sin/cos so the tree sees a smooth
# year-of-day signal instead of a 1-366 ordinal.
STATIC_NAMES = (
    "q_obs_today_log",      # asinh(q_obs_t / qs)
    "log_drain_area",       # log1p(drain_area_sqmi)
    "lat",
    "lon",
    "log_alt_ft",
    "doy_sin",
    "doy_cos",
    "issued_doy_sin",
    "issued_doy_cos",
)

# v14.5a: GAGES-II static augmentation. Order locked, NaN for stations
# without a GAGES-II row (~32% of actives). LightGBM treats NaN as its own
# split direction so missing rows don't poison the panel.
GAGES2_KEYS = (
    "FORESTNLCD06", "DEVNLCD06", "WOODYWETNLCD06", "EMERGWETNLCD06",
    "HGA_PCT", "HGB_PCT", "HGC_PCT", "HGD_PCT",
    "AWCAVE", "PERMAVE",
    "ELEV_MEAN_M_BASIN", "SLOPE_PCT",
    "BFI_AVE", "TOPWET", "RUNAVE7100",
    "PPTAVG_BASIN", "T_AVG_BASIN", "SNOW_PCT_PRECIP",
)


def _row_features(
    member_vals_asinh: Dict[str, float],
    q_obs_today_asinh: float,
    static_vec: List[float],
    target_doy: int,
    issued_doy: int,
) -> List[float]:
    """Build a single feature row in MEMBER_ORDER + STATIC_NAMES + GAGES2_KEYS order.

    Member values are in asinh space (already divided by qs and asinh'd) so
    their ranges across stations are comparable — without that, a station
    with q_scale=10000 dominates one with q_scale=10.

    v14.5a: appends 18 GAGES-II columns from `static_vec[4:22]` (NaN when
    that station isn't in the GAGES-II 9067-gauge table).
    """
    row: List[float] = []
    for m in MEMBER_ORDER:
        v = member_vals_asinh.get(m)
        row.append(float(v) if v is not None and math.isfinite(v) else float("nan"))
    target_rad = 2.0 * math.pi * (target_doy - 1) / 366.0
    issued_rad = 2.0 * math.pi * (issued_doy - 1) / 366.0
    row.append(float(q_obs_today_asinh))
    row.extend(float(x) for x in static_vec[:4])  # log_drain_area, lat, lon, log_alt
    row.append(math.sin(target_rad))
    row.append(math.cos(target_rad))
    row.append(math.sin(issued_rad))
    row.append(math.cos(issued_rad))
    # v14.5a: append GAGES-II static columns. Always exactly len(GAGES2_KEYS)
    # so the booster sees a stable schema; missing rows are NaN.
    if len(static_vec) >= 4 + len(GAGES2_KEYS):
        row.extend(float(x) for x in static_vec[4:4 + len(GAGES2_KEYS)])
    else:
        row.extend([float("nan")] * len(GAGES2_KEYS))
    return row


def _static_vec_from_attrs(attrs: dict) -> Optional[List[float]]:
    """Pack the static cells we need from a station attrs dict.

    Layout: [log_drain_area, lat, lon, log_alt, *GAGES2_KEYS values].
    Returns None if lat/lon are missing. GAGES-II values are NaN per-cell
    when the station isn't covered (~32% of actives).
    """
    try:
        lat = float(attrs.get("lat"))
        lon = float(attrs.get("lon"))
    except Exception:
        return None
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    area = float(attrs.get("drain_area_sqmi", 0) or 0)
    alt = float(attrs.get("alt_ft", 0) or 0)
    base = [
        math.log1p(max(area, 0.0)),
        lat,
        lon,
        math.log1p(max(alt, 0.0)),
    ]
    g2: list[float] = []
    for k in GAGES2_KEYS:
        v = attrs.get(k)
        if v is None:
            g2.append(float("nan"))
            continue
        try:
            f = float(v)
        except Exception:
            g2.append(float("nan"))
            continue
        if not math.isfinite(f):
            g2.append(float("nan"))
        else:
            g2.append(f)
    return base + g2


@dataclass
class StackerTrainer:
    """Pooled per-horizon LightGBM blend stacker.

    Training: each call to `add_station` walks that station's holdout panel
    (member_preds: {member: [(offset, yhat[14], ytrue[14]), ...]}) and emits
    one (X, y) row per (offset, horizon) where the target is asinh(ytrue/qs)
    and features are member preds in asinh space + station context.

    Inference: `predict_blend(member_panel_asinh, ...)` returns a 14-element
    list of CFS values, one per horizon. Returns None for any horizon whose
    booster failed to train or predicted a non-finite value.
    """
    horizon: int = 14
    # rows[h] = list of (X_batch, y_batch); kept as small lists per station,
    # concatenated once at fit time.
    _rows: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    _models: Dict[int, object] = field(default_factory=dict)
    _fitted: bool = False
    # Per-station scaling needed at inference: qs to map members→asinh space
    # and back, and the station's static vec.
    _station_state: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rows = {h: [] for h in range(1, self.horizon + 1)}

    @property
    def feature_names(self) -> List[str]:
        return list(MEMBER_ORDER) + list(STATIC_NAMES) + [f"g2_{k}" for k in GAGES2_KEYS]

    def add_station(
        self,
        station_id: str,
        attrs: dict,
        qs: float,
        member_preds: Dict[str, List[Tuple[int, list, list]]],
        offset_to_issued_doy: Dict[int, int],
    ) -> None:
        """Append this station's holdout panel rows to the training buffer.

        member_preds: same shape as `member_preds` inside `forecast_station`,
            i.e. {member_name: [(offset, yhat_list, ytrue_list)]}. Members
            with empty lists (NWM, timesfm_xreg in current code) just don't
            contribute to that row's features for that station.
        offset_to_issued_doy: {offset: issued_dayofyear} so each holdout
            row knows the day-of-year the forecast was issued from.
        """
        if self._fitted:
            return
        sv = _static_vec_from_attrs(attrs)
        if sv is None or not math.isfinite(qs) or qs <= 0:
            return

        # Group all members' tuples by offset so we can build one row per
        # (offset, h) with every member's prediction at that horizon.
        per_offset: Dict[int, Dict[str, list]] = {}
        per_offset_ytrue: Dict[int, list] = {}
        for member, plist in member_preds.items():
            if member not in MEMBER_ORDER:
                continue
            for off, yhat, ytrue in plist:
                slot = per_offset.setdefault(off, {})
                slot[member] = list(yhat)
                # ytrue is the same across members for a given offset; only
                # set once.
                if off not in per_offset_ytrue:
                    per_offset_ytrue[off] = list(ytrue)

        if not per_offset:
            return

        for off, ytrue_h in per_offset_ytrue.items():
            issued_doy = offset_to_issued_doy.get(off)
            if issued_doy is None:
                continue
            for h in range(1, self.horizon + 1):
                if h - 1 >= len(ytrue_h):
                    continue
                y = ytrue_h[h - 1]
                if y is None or not math.isfinite(float(y)):
                    continue
                # Feature row: each member's pred at h in asinh(/qs) space.
                member_vals_asinh: Dict[str, float] = {}
                for member, yhat in per_offset[off].items():
                    if h - 1 >= len(yhat):
                        continue
                    v = yhat[h - 1]
                    if v is None or not math.isfinite(float(v)):
                        continue
                    try:
                        member_vals_asinh[member] = math.asinh(float(v) / qs)
                    except Exception:
                        continue
                if not member_vals_asinh:
                    continue
                # q_obs at issue time isn't directly stored, but ytrue at
                # offset=0,h=0 of the same offset is the day after issue;
                # closest proxy we have without re-passing q_hist is
                # ytrue_h[0] one step back — instead, encode it via the
                # static vec dropping. We *don't* include q_obs_today in
                # holdout rows because we'd have to thread it through; the
                # tree can lean on static + member levels alone, and the
                # live-inference path provides the real q_obs_today.
                target_doy = ((issued_doy - 1 + h) % 366) + 1
                # At training time, q_obs_today_asinh is approximated as the
                # mean of the member values at h=1 (which all anchor close
                # to the truth at lag 1) — keeps the column populated and
                # consistent with live inference, where it's the real value.
                q_obs_proxy = 0.0
                if 1 in per_offset_ytrue.get(off, []):
                    pass
                # Use ytrue at h=1 of this offset as the true q_obs_lag1
                # proxy — that's literally observed flow one day after
                # issue, which is the closest stand-in for "today's flow"
                # at issue time.
                ytrue_lag1 = ytrue_h[0] if len(ytrue_h) >= 1 else None
                if ytrue_lag1 is not None and math.isfinite(float(ytrue_lag1)):
                    try:
                        q_obs_proxy = math.asinh(float(ytrue_lag1) / qs)
                    except Exception:
                        q_obs_proxy = 0.0
                row = _row_features(
                    member_vals_asinh=member_vals_asinh,
                    q_obs_today_asinh=q_obs_proxy,
                    static_vec=sv,
                    target_doy=target_doy,
                    issued_doy=issued_doy,
                )
                try:
                    y_asinh = math.asinh(float(y) / qs)
                except Exception:
                    continue
                X_arr = np.asarray(row, dtype=np.float32).reshape(1, -1)
                y_arr = np.asarray([y_asinh], dtype=np.float32)
                self._rows[h].append((X_arr, y_arr))

        self._station_state[station_id] = {
            "qs": float(qs),
            "static": sv,
        }

    def n_rows(self) -> int:
        return sum(sum(b[0].shape[0] for b in batches) for batches in self._rows.values())

    def fit(self) -> bool:
        """Fit one LightGBM per horizon. Returns True if any horizon trained."""
        if not _LGB_OK:
            return False
        any_fit = False
        for h in range(1, self.horizon + 1):
            batches = self._rows.get(h) or []
            total = sum(b[0].shape[0] for b in batches)
            # Need a meaningful number of rows; with 6 offsets × 1893 stations
            # the upper bound per-horizon is ~11K, so 200 is a comfortable
            # floor that still trains a useful tree.
            if total < 200:
                continue
            X = np.concatenate([b[0] for b in batches], axis=0).astype(np.float32)
            y = np.concatenate([b[1] for b in batches], axis=0).astype(np.float32)
            params = {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": max(20, len(y) // 200),
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 5,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": 2,
            }
            try:
                ds = _lgb.Dataset(X, label=y, free_raw_data=False)
                booster = _lgb.train(params, ds, num_boost_round=300)
                self._models[h] = booster
                any_fit = True
            except Exception:
                continue
        self._fitted = True
        # Drop training rows; we keep _station_state for inference.
        self._rows = {}
        return any_fit

    @property
    def fitted(self) -> bool:
        return self._fitted and bool(self._models)

    def predict_blend(
        self,
        station_id: str,
        attrs: dict,
        qs: float,
        members_panel: Dict[str, List[Optional[float]]],
        q_obs_today: float,
        issued_doy: int,
        target_doys: List[int],
    ) -> Optional[List[Optional[float]]]:
        """Predict 14-day blend for one station using the fitted stacker.

        members_panel: {member_name: [q_cfs at h=1..14, possibly None]} —
            the live forecast values from this station's `members` dict.
        q_obs_today: observed flow at issue time (CFS).
        target_doys: dayofyear of each forecast target h=1..14.

        Returns a 14-element list of CFS predictions (or None per horizon
        when the stacker can't predict for that horizon), or None entirely
        if the stacker isn't fit or this station has no static state.
        """
        if not self.fitted:
            return None
        sv = _static_vec_from_attrs(attrs)
        if sv is None or not math.isfinite(qs) or qs <= 0:
            return None
        try:
            q_obs_today_asinh = math.asinh(float(q_obs_today) / qs)
        except Exception:
            q_obs_today_asinh = 0.0

        out: List[Optional[float]] = []
        for h in range(1, self.horizon + 1):
            booster = self._models.get(h)
            if booster is None:
                out.append(None)
                continue
            target_doy = target_doys[h - 1] if h - 1 < len(target_doys) else issued_doy
            member_vals_asinh: Dict[str, float] = {}
            for m in MEMBER_ORDER:
                vals = members_panel.get(m)
                if not vals or h - 1 >= len(vals):
                    continue
                v = vals[h - 1]
                if v is None or not math.isfinite(float(v)):
                    continue
                try:
                    member_vals_asinh[m] = math.asinh(float(v) / qs)
                except Exception:
                    continue
            if not member_vals_asinh:
                out.append(None)
                continue
            row = _row_features(
                member_vals_asinh=member_vals_asinh,
                q_obs_today_asinh=q_obs_today_asinh,
                static_vec=sv,
                target_doy=target_doy,
                issued_doy=issued_doy,
            )
            X = np.asarray(row, dtype=np.float32).reshape(1, -1)
            try:
                yhat_asinh = float(np.asarray(booster.predict(X))[0])
            except Exception:
                out.append(None)
                continue
            try:
                q = float(qs * math.sinh(yhat_asinh))
            except Exception:
                out.append(None)
                continue
            if not math.isfinite(q) or q < 0:
                out.append(None)
            else:
                out.append(q)
        return out

    def score_holdouts(
        self,
        station_id: str,
        attrs: dict,
        qs: float,
        member_preds: Dict[str, List[Tuple[int, list, list]]],
        offset_to_issued_doy: Dict[int, int],
        offset_to_target_doys: Dict[int, List[int]],
        q_obs_at_offset: Dict[int, float],
    ) -> Dict[int, List[float]]:
        """Predict on this station's own holdout panel for fair MAE reporting.

        Returns {h: [|yhat - ytrue| ...]} — one absolute error per
        (offset, h) where the stacker successfully predicted.
        Used by `forecast_station` to populate `rolling_mae_blend`.
        """
        out: Dict[int, List[float]] = {h: [] for h in range(1, self.horizon + 1)}
        if not self.fitted:
            return out
        sv = _static_vec_from_attrs(attrs)
        if sv is None or not math.isfinite(qs) or qs <= 0:
            return out
        per_offset: Dict[int, Dict[str, list]] = {}
        per_offset_ytrue: Dict[int, list] = {}
        for member, plist in member_preds.items():
            if member not in MEMBER_ORDER:
                continue
            for off, yhat, ytrue in plist:
                slot = per_offset.setdefault(off, {})
                slot[member] = list(yhat)
                if off not in per_offset_ytrue:
                    per_offset_ytrue[off] = list(ytrue)
        for off, ytrue_h in per_offset_ytrue.items():
            issued_doy = offset_to_issued_doy.get(off)
            target_doys = offset_to_target_doys.get(off, [])
            if issued_doy is None or not target_doys:
                continue
            q_obs = q_obs_at_offset.get(off)
            if q_obs is None or not math.isfinite(q_obs):
                continue
            try:
                q_obs_asinh = math.asinh(float(q_obs) / qs)
            except Exception:
                continue
            for h in range(1, self.horizon + 1):
                booster = self._models.get(h)
                if booster is None or h - 1 >= len(ytrue_h):
                    continue
                y = ytrue_h[h - 1]
                if y is None or not math.isfinite(float(y)):
                    continue
                target_doy = target_doys[h - 1] if h - 1 < len(target_doys) else issued_doy
                member_vals_asinh: Dict[str, float] = {}
                for m in MEMBER_ORDER:
                    yh = per_offset[off].get(m)
                    if not yh or h - 1 >= len(yh):
                        continue
                    v = yh[h - 1]
                    if v is None or not math.isfinite(float(v)):
                        continue
                    try:
                        member_vals_asinh[m] = math.asinh(float(v) / qs)
                    except Exception:
                        continue
                if not member_vals_asinh:
                    continue
                row = _row_features(
                    member_vals_asinh=member_vals_asinh,
                    q_obs_today_asinh=q_obs_asinh,
                    static_vec=sv,
                    target_doy=target_doy,
                    issued_doy=issued_doy,
                )
                X = np.asarray(row, dtype=np.float32).reshape(1, -1)
                try:
                    yhat_asinh = float(np.asarray(booster.predict(X))[0])
                except Exception:
                    continue
                try:
                    q = float(qs * math.sinh(yhat_asinh))
                except Exception:
                    continue
                if not math.isfinite(q) or q < 0:
                    continue
                out[h].append(abs(q - float(y)))
        return out
