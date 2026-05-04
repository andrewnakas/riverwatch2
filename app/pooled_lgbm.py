"""Pooled-LGBM cross-station regressor (v13.2).

Trains a single LightGBM model per horizon h=1..14 over rows from EVERY station
in the build shard. Static basin attributes (lat/lon/alt/drainage area/HUC) act
as splitters so the tree learns rainfall→runoff transfer functions conditioned
on basin character. The per-station LGBM in `forecast.py` doesn't see static
features (they're constant within a station), so this is the architectural
complement: it learns *across* stations.

Exposed API:
    PooledTrainer.add_station(...)    # accumulate training rows from one station
    PooledTrainer.fit()               # train h=1..14 models
    PooledTrainer.predict(...)        # per-station live prediction
    PooledTrainer.holdout_score(...)  # per-station MAE estimate via tail-holdout

The pooled model is fit per-shard (118 stations × ~3000 days × 14 horizons = ~5M
rows). Pre-train cost: ~5-10 minutes/shard with LightGBM at num_leaves=31. Adds
to the cold-cache build path; warm builds reuse the cached model artifact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as _lgb  # type: ignore
    _LGB_OK = True
except Exception:
    _LGB_OK = False


# Static basin attributes pulled from stations_40_enriched.json. drain_area
# and alt are log-scaled at training time because they span 4+ orders of
# magnitude (small headwater creeks to mainstems).
STATIC_KEYS = ("lat", "lon", "alt_ft", "drain_area_sqmi", "huc_cd")

# v14.5a: GAGES-II augmentation. ~68% coverage; missing columns become NaN
# at row-build time and LightGBM splits on the missingness mask. Order is
# locked so the booster sees a stable feature schema across stations and
# across pooled vs stacker (the stacker reuses the same suffix block).
GAGES2_STATIC_KEYS = (
    "FORESTNLCD06", "DEVNLCD06", "WOODYWETNLCD06", "EMERGWETNLCD06",
    "HGA_PCT", "HGB_PCT", "HGC_PCT", "HGD_PCT",
    "AWCAVE", "PERMAVE",
    "ELEV_MEAN_M_BASIN", "SLOPE_PCT",
    "BFI_AVE", "TOPWET", "RUNAVE7100",
    "PPTAVG_BASIN", "T_AVG_BASIN", "SNOW_PCT_PRECIP",
)


def _huc2(huc: Optional[str]) -> int:
    """First two HUC digits = HUC-2 region (1-21 in CONUS, treat unknown as 0)."""
    if not huc:
        return 0
    try:
        return int(str(huc)[:2])
    except Exception:
        return 0


def _static_vec(attrs: dict) -> Optional[np.ndarray]:
    """Pack static features into a fixed-length vector. Missing static fields
    return None so we don't pollute the pooled training set with zeros.

    v14.5a: appends 18 GAGES-II columns when present (else NaN). LightGBM
    handles NaN as its own split, so missing GAGES-II rows on ~32% of our
    stations don't poison the training panel.
    """
    try:
        lat = float(attrs.get("lat"))
        lon = float(attrs.get("lon"))
        alt = float(attrs.get("alt_ft", 0) or 0)
        area = float(attrs.get("drain_area_sqmi", 0) or 0)
        huc2 = _huc2(attrs.get("huc_cd"))
    except Exception:
        return None
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    base = [
        lat, lon,
        np.log1p(max(alt, 0.0)),
        np.log1p(max(area, 0.0)),
        float(huc2),
    ]
    g2: list[float] = []
    for k in GAGES2_STATIC_KEYS:
        v = attrs.get(k)
        if v is None:
            g2.append(np.float32("nan"))
            continue
        try:
            f = float(v)
        except Exception:
            g2.append(np.float32("nan"))
            continue
        if not math.isfinite(f):
            g2.append(np.float32("nan"))
        else:
            g2.append(f)
    return np.array(base + g2, dtype=np.float32)


_STATIC_NAMES = (
    ["s_lat", "s_lon", "s_log_alt", "s_log_area", "s_huc2"]
    + [f"g2_{k}" for k in GAGES2_STATIC_KEYS]
)


@dataclass
class PooledTrainer:
    """Accumulates training rows across stations, then fits one LGBM per horizon.

    Train rows are (station_features × static_features, target_log_anom_h{h}).
    target = asinh(q_t+h / q_scale) - q_log_clim(t+h). Each station contributes
    its q_scale and q_log_clim alignment so the pooled target is unitless and
    comparable across basins.
    """
    horizon: int = 14
    # Master column schema, locked at first add_station call. Subsequent
    # stations are projected onto these columns (missing -> NaN, LightGBM
    # handles NaN natively); columns the first station didn't have are dropped.
    feature_cols: List[str] = field(default_factory=list)
    _master_dyn_cols: Optional[List[str]] = None
    # rows[h] = list of (X_batch, y_batch) per (station, horizon)
    _rows: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    _models: Dict[int, object] = field(default_factory=dict)
    # cached per-station data needed at predict time
    _station_state: Dict[str, dict] = field(default_factory=dict)
    _seen_static_dim: Optional[int] = None
    _fitted: bool = False
    # Cap rows per (station, horizon) so 119 stations × 100yr histories don't
    # produce 30M-row training matrices per horizon. With cap=5000 we get
    # ~600K rows/horizon — plenty for a 31-leaf tree to learn rainfall→runoff
    # transfer functions, and CI memory + time stays bounded.
    max_rows_per_station_per_h: int = 5000

    def __post_init__(self) -> None:
        self._rows = {h: [] for h in range(1, self.horizon + 1)}

    def add_station(
        self,
        station_id: str,
        feats_df: pd.DataFrame,
        attrs: dict,
        cols: List[str],
        qs: float,
        has_clim: bool,
    ) -> None:
        """Append this station's training rows to the pooled buffer.

        feats_df: output of forecast._build_features (already has q_log,
            q_log_clim, lags, covariates).
        cols: result of forecast._feature_columns(feats_df).
        qs: per-station asinh scale.
        """
        if self._fitted:
            return  # don't accumulate after fit
        sv = _static_vec(attrs)
        if sv is None:
            return
        if self._seen_static_dim is None:
            self._seen_static_dim = len(sv)
        elif len(sv) != self._seen_static_dim:
            return  # malformed

        # Lock master schema on first station: every later station gets
        # projected onto these columns (missing -> NaN). _feature_columns drops
        # cols with <10 non-null, so different stations have different col sets;
        # naive concat then fails with shape mismatch.
        if self._master_dyn_cols is None:
            self._master_dyn_cols = list(cols)
            self.feature_cols = self._master_dyn_cols + _STATIC_NAMES
        master = self._master_dyn_cols

        # Extract target series first (they may also be in master as features).
        # q_log is the asinh-transformed observed flow; q_log_clim is its
        # day-of-year climatology. Used to build per-h target = q_log.shift(-h)
        # - q_log_clim.shift(-h).
        if "q_log" not in feats_df.columns:
            return  # malformed feats — can't build pooled targets
        q_log_series = feats_df["q_log"]
        q_log_clim_series = feats_df["q_log_clim"] if has_clim and "q_log_clim" in feats_df.columns else None
        # Reindex feats to master schema. Pandas creates NaN for missing cols.
        feats_master = feats_df.reindex(columns=master)

        # Use the same per-h target construction as the per-station LGBM:
        # target_anom = q_log.shift(-h) - q_log_clim.shift(-h) (or raw q_log).
        # v13.2 mem fix: store one (X, y) batch per (station, horizon) — NOT one
        # row per tuple. With 118 stations × 14 horizons × ~50K rows/station,
        # row-level tuples overflow RAM (~25 GB). Batched arrays keep it ~1.8 GB.
        rng = np.random.default_rng(seed=hash(station_id) & 0xFFFFFFFF)
        for h in range(1, self.horizon + 1):
            target_log = q_log_series.shift(-h)
            if q_log_clim_series is not None:
                target = target_log - q_log_clim_series.shift(-h)
            else:
                target = target_log
            df_h = feats_master.copy()
            df_h["__target__"] = target.values
            # Drop only on target NaN; keep NaN feature cells (LightGBM handles).
            train = df_h.dropna(subset=["__target__"])
            if len(train) < 60:
                continue
            # Subsample to cap rows per (station, horizon).
            cap = self.max_rows_per_station_per_h
            if len(train) > cap:
                idx = rng.choice(len(train), size=cap, replace=False)
                train = train.iloc[np.sort(idx)]
            X = train[master].values.astype(np.float32)
            y = train["__target__"].values.astype(np.float32)
            # Append static-feature columns broadcast across all rows.
            X = np.concatenate([X, np.tile(sv, (X.shape[0], 1))], axis=1)
            self._rows[h].append((X, y))

        # Cache predict-time state for this station (live inference + holdout).
        self._station_state[station_id] = {
            "static": sv,
            "qs": float(qs),
            "has_clim": bool(has_clim),
            "cols": list(cols),
        }

    def n_rows(self) -> int:
        # _rows[h] is a list of (X_batch, y_batch); count rows in batches.
        return sum(sum(b[0].shape[0] for b in batches) for batches in self._rows.values())

    def fit(self) -> bool:
        """Train per-horizon LGBM models on the accumulated rows. Returns True
        if at least one horizon model trained, False if no horizon had enough
        data or LightGBM is unavailable."""
        if not _LGB_OK:
            return False
        any_fit = False
        for h in range(1, self.horizon + 1):
            batches = self._rows.get(h) or []
            total = sum(b[0].shape[0] for b in batches)
            if total < 200:  # need enough cross-station rows to be worthwhile
                continue
            X = np.concatenate([b[0] for b in batches], axis=0).astype(np.float32)
            y = np.concatenate([b[1] for b in batches], axis=0).astype(np.float32)
            params = {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": max(50, len(y) // 200),
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": 2,
            }
            try:
                ds = _lgb.Dataset(X, label=y, free_raw_data=False)
                booster = _lgb.train(params, ds, num_boost_round=200)
                self._models[h] = booster
                any_fit = True
            except Exception:
                continue
        self._fitted = True
        # Drop training rows to free memory after fit.
        self._rows = {}
        return any_fit

    def predict(
        self,
        station_id: str,
        feats_now: pd.Series,
        cols: List[str],
        last_q: float,
        clim_at_target_per_h: Dict[int, float],
    ) -> Optional[List[float]]:
        """Per-station live forecast using the pooled models."""
        if not self._fitted or not self._models or self._master_dyn_cols is None:
            return None
        st = self._station_state.get(station_id)
        if st is None:
            return None
        sv = st["static"]
        qs = st["qs"]
        has_clim = st["has_clim"]

        # Project per-station feats_now onto master schema. Cols this station
        # is missing become NaN (LightGBM handles natively, same as training).
        master = self._master_dyn_cols
        feats_now_dict = {c: float(feats_now.get(c, np.nan)) for c in master}
        x_dyn = np.array([feats_now_dict[c] for c in master], dtype=np.float32)
        x_live = np.concatenate([x_dyn, sv]).reshape(1, -1)

        out: List[float] = []
        for h in range(1, self.horizon + 1):
            booster = self._models.get(h)
            if booster is None:
                out.append(float("nan"))
                continue
            try:
                yhat_anom = float(np.asarray(booster.predict(x_live))[0])
            except Exception:
                out.append(float("nan"))
                continue
            yhat_z = yhat_anom + clim_at_target_per_h.get(h, 0.0) if has_clim else yhat_anom
            # Inverse asinh: q = qs * sinh(z)
            try:
                q = float(qs * math.sinh(yhat_z))
            except Exception:
                q = last_q
            if not math.isfinite(q) or q < 0:
                q = last_q
            out.append(q)
        return out

    def holdout_score_one(
        self,
        station_id: str,
        feats_df: pd.DataFrame,
        cols: List[str],
        q_hist: pd.DataFrame,
        end_offset: int,
        horizon: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Score the pooled model on a single station's tail-holdout window.

        Reuses train-time feats_df: at the as-of date `len(q_hist) - end_offset`
        we extract the feature row + apply each horizon model, vs the actual
        observed q at t+h. This is mildly leaky (the pooled model was trained
        on rows from this station's full history), but the per-station LGBM
        ridge has the same property and we score them apples-to-apples.
        """
        if not self._fitted or not self._models or self._master_dyn_cols is None:
            return None
        st = self._station_state.get(station_id)
        if st is None:
            return None
        sv = st["static"]
        qs = st["qs"]
        has_clim = st["has_clim"]

        end_idx = len(q_hist) - end_offset
        ctx_len = end_idx - horizon
        if ctx_len < 64 or end_idx > len(q_hist):
            return None
        last_date = pd.to_datetime(q_hist["date"].iloc[ctx_len - 1])
        ytrue = q_hist["q_cfs"].iloc[ctx_len:ctx_len + horizon].values.astype(float)
        if len(ytrue) < horizon:
            return None
        last_q = float(q_hist["q_cfs"].iloc[ctx_len - 1])

        if last_date not in feats_df.index:
            return None
        # Project per-station feats onto master schema (NaN for missing cols).
        master = self._master_dyn_cols
        row = feats_df.loc[last_date]
        x_dyn = np.array(
            [float(row[c]) if c in row.index else float("nan") for c in master],
            dtype=np.float32,
        )
        x_live = np.concatenate([x_dyn, sv]).reshape(1, -1)

        last_doy = int(last_date.dayofyear)
        yhat = np.empty(horizon, dtype=float)
        for h in range(1, horizon + 1):
            booster = self._models.get(h)
            if booster is None:
                yhat[h - 1] = last_q
                continue
            try:
                yhat_anom = float(np.asarray(booster.predict(x_live))[0])
            except Exception:
                yhat[h - 1] = last_q
                continue
            clim_at_target = 0.0
            if has_clim:
                target_doy = ((last_doy - 1 + h) % 366) + 1
                doy_match = feats_df.index.dayofyear == target_doy
                if doy_match.any():
                    clim_at_target = float(feats_df.loc[doy_match, "q_log_clim"].iloc[0])
            yhat_z = yhat_anom + clim_at_target if has_clim else yhat_anom
            try:
                q = float(qs * math.sinh(yhat_z))
            except Exception:
                q = last_q
            if not math.isfinite(q) or q < 0:
                q = last_q
            yhat[h - 1] = q
        return yhat, ytrue

    @property
    def fitted(self) -> bool:
        return self._fitted and bool(self._models)
