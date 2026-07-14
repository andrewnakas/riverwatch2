#!/usr/bin/env python
"""Grand-ensemble combiner for heterogeneous MB-LSTM / δHBV member dumps.

The backtest's --ckpt a:b:c ensembler requires all members to share ONE cfg
(same enc_vars / normalization), so it cannot combine models trained on
different corpora (per-forcing single-forcing LSTMs, or a δHBV member with a
different input recipe). This combiner does that combination OFFLINE from the
per-window dumps (--dump-windows), which are already in physical cfs space.

Each dump is the --dump-windows schema:
    station_id, t0, h, truth, ylo, ymed, yhi, ymean, persist

Combination (default): inner-join all dumps on (station_id, t0, h) and average
the point column across members — Vincentization in physical space, matching the
z-space quantile-averaging the seed ensembler uses internally (app/mblstm.py).
Point column is `ymed` (the served point; for CMAL dumps ymean carries the
distribution mean — use --point ymean for those). Optional --weights fits a
single LSTM-vs-δHBV mixing weight, but ONLY on a held-out slice (--weight-val-
end) so the test decade is never used to tune the combination.

Scoring reuses app.metrics exactly as backtest_mblstm.py does: per-station
pooled all_point_metrics → aggregate (median/mean/scorable), plus a day-1
(h==1) slice and tercile guard. Emits a JSON in the mblstm_backtest_*.json
shape so it drops into the comparison table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app import metrics  # noqa: E402

KEY = ["station_id", "t0", "h"]
CAMELS_ATTRS_PATH = REPO / "data" / "camels_attrs.json"

# The 27 Addor static attributes used to condition the ensemble gate. A basin's
# member-mix should depend on its hydrology type (δHBV wins snow/arid, LSTM wins
# flashy humid), which a single global weight vector throws away.
GATE_FEATS = [
    "p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2", "soil_depth_pelletier",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
    "max_water_content", "sand_frac", "silt_frac", "clay_frac",
    "frac_forest", "lai_max", "gvf_max", "gvf_diff", "root_depth_50",
    "carbonate_rocks_frac", "geol_permeability",
]


def load_dump(path: Path, point: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_id": str})
    miss = [c for c in KEY + ["truth", point] if c not in df.columns]
    if miss:
        raise SystemExit(f"{path}: missing columns {miss}")
    return df[KEY + ["truth", point]].rename(columns={point: "sim"})


def combine(dumps: list[pd.DataFrame], weights: list[float] | None) -> pd.DataFrame:
    """Inner-join members on KEY; weighted-average their sim columns. truth is
    taken from the first member (identical across members by construction)."""
    base = dumps[0].rename(columns={"sim": "sim_0"})
    for i, d in enumerate(dumps[1:], 1):
        base = base.merge(d.drop(columns="truth").rename(columns={"sim": f"sim_{i}"}),
                          on=KEY, how="inner")
    sim_cols = [c for c in base.columns if c.startswith("sim_")]
    w = np.asarray(weights if weights else [1.0] * len(sim_cols), dtype=float)
    w = w / w.sum()
    base["sim"] = (base[sim_cols].to_numpy() * w[None, :]).sum(axis=1)
    return base[KEY + ["truth", "sim"]]


def score(df: pd.DataFrame, day1_only: bool = False) -> dict:
    """Per-station pooled metrics → aggregate, matching backtest_mblstm.py."""
    d = df[df["h"] == 1] if day1_only else df
    per_station: dict[str, dict] = {}
    for sid, g in d.groupby("station_id", sort=False):
        obs = g["truth"].to_numpy(dtype=float)
        sim = g["sim"].to_numpy(dtype=float)
        m = np.isfinite(obs) & np.isfinite(sim)
        if m.sum() < 5:
            continue
        per_station[sid] = metrics.all_point_metrics(obs[m], sim[m])
    return metrics.aggregate(per_station)


def fit_weights(dumps, val_end: str):
    """Optimize per-member weights to maximize median NSE on the VAL slice ONLY
    (t0 <= val_end), so the test decade is never used to tune the combination.
    scipy differential-evolution (the paper's 'GA') on simplex weights."""
    from scipy.optimize import differential_evolution
    # merge members once; split by t0 date
    merged = dumps[0].rename(columns={"sim": "sim_0"})
    for i, d in enumerate(dumps[1:], 1):
        merged = merged.merge(d.drop(columns="truth").rename(columns={"sim": f"sim_{i}"}),
                              on=KEY, how="inner")
    sim_cols = [c for c in merged.columns if c.startswith("sim_")]
    val = merged[merged["t0"] <= val_end]
    if len(val) < 100:
        print(f"WARN: only {len(val)} val rows (t0<={val_end}) — weights may be noisy",
              file=sys.stderr)
    S = val[sim_cols].to_numpy(); y = val["truth"].to_numpy()
    fin = np.isfinite(y) & np.isfinite(S).all(axis=1)
    S, y = S[fin], y[fin]
    codes, _ = pd.factorize(val["station_id"].to_numpy()[fin])   # 0..G-1
    G = codes.max() + 1
    cnt = np.bincount(codes, minlength=G)
    # per-station obs variance (SST/n) is candidate-independent — precompute once
    ybar = np.bincount(codes, weights=y, minlength=G) / np.maximum(cnt, 1)
    sst = np.bincount(codes, weights=(y - ybar[codes]) ** 2, minlength=G)
    ok = (cnt >= 5) & (sst > 1e-9)

    def neg_med_nse(w):
        w = np.abs(w); w = w / (w.sum() + 1e-9)
        sim = S @ w
        sse = np.bincount(codes, weights=(y - sim) ** 2, minlength=G)
        nse = 1.0 - sse[ok] / sst[ok]
        return -(np.median(nse) if nse.size else -1.0)

    n = len(sim_cols)
    res = differential_evolution(neg_med_nse, [(0.0, 1.0)] * n, seed=0,
                                 maxiter=40, tol=1e-4, polish=True)
    w = np.abs(res.x); w = w / w.sum()
    print(f"fitted weights (val medNSE={-res.fun:.4f}): "
          + ", ".join(f"{c}={wi:.3f}" for c, wi in zip(sim_cols, w)))
    return list(w)


def _load_gate_features(station_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """z-normalized 27-attr static matrix for the given basins.

    Returns (X, mask) where X is (n_basins, 27) with NaN attrs imputed to 0 (the
    z-mean) and mask flags basins that had usable attrs. Normalization stats are
    computed over the basins present (median/IQR-robust would be overkill here;
    plain mean/std with NaN-omit is fine and matches the trainer's static z-score).
    """
    attrs = json.loads(CAMELS_ATTRS_PATH.read_text())
    raw = np.full((len(station_ids), len(GATE_FEATS)), np.nan)
    for i, sid in enumerate(station_ids):
        a = attrs.get(sid, {})
        for j, f in enumerate(GATE_FEATS):
            v = a.get(f)
            if v is not None and np.isfinite(v):
                raw[i, j] = float(v)
    mask = np.isfinite(raw).any(axis=1)
    mu = np.nanmean(raw, axis=0)
    sd = np.nanstd(raw, axis=0)
    sd[sd < 1e-9] = 1.0
    X = (raw - mu[None, :]) / sd[None, :]
    X = np.where(np.isfinite(X), X, 0.0)   # impute missing attr → z-mean (0)
    return X, mask


def _soft_membership(X: np.ndarray, k: int, tau: float = 1.0,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cluster basins by static signature (k-means) → soft membership (n_basins,k).

    Membership_ik = softmax over clusters of -dist(basin_i, centroid_k)/tau, so a
    basin blends the cluster weight-vectors of the static regimes it resembles.
    k-means gives stable, interpretable regimes and keeps the fit low-dimensional
    (k×n_members params, not 531×n_members) to resist overfitting the val slice.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    d2 = ((X[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)  # (n,k)
    logits = -d2 / max(tau, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    M = np.exp(logits)
    M /= M.sum(axis=1, keepdims=True)
    return M, km.cluster_centers_


def fit_static_gate(dumps, val_end: str, k: int, seed: int = 0, gate_reg=None):
    """Per-basin ensemble weights conditioned on the 27 static attrs, fit on the
    VAL slice only (t0 <= val_end). Learns k per-cluster weight vectors; a basin's
    weights = its soft-membership blend of them, L2-regularized toward the shared
    (global) weight so unhelpful clusters collapse to the robust baseline. Returns
    a dict {station_id: w} for ALL basins in the merged grid (test basins get
    weights from their statics, never their test data — no leakage)."""
    from scipy.optimize import differential_evolution
    merged = dumps[0].rename(columns={"sim": "sim_0"})
    for i, d in enumerate(dumps[1:], 1):
        merged = merged.merge(d.drop(columns="truth").rename(columns={"sim": f"sim_{i}"}),
                              on=KEY, how="inner")
    sim_cols = [c for c in merged.columns if c.startswith("sim_")]
    n_mem = len(sim_cols)
    station_ids = list(pd.unique(merged["station_id"]))
    sid_index = {s: i for i, s in enumerate(station_ids)}
    X, _ = _load_gate_features(station_ids)
    M, _ = _soft_membership(X, k, seed=seed)                     # (n_basins, k)

    val = merged[merged["t0"] <= val_end]
    # The val slice can be ~0.6M rows; DE calls the objective thousands of times.
    # Subsample HORIZONS (not just h==1) so the fit optimizes the same POOLED
    # metric we report — fitting on day-1 only overfits day-1 and hurts pooled.
    # Keep a few spread leads (1,4,7,10,14) for a ~3x speedup with pooled-faithful
    # signal.
    if "h" in val.columns:
        val = val[val["h"].isin([1, 4, 7, 10, 14])]
    S = val[sim_cols].to_numpy(); y = val["truth"].to_numpy()
    b_codes = np.array([sid_index[s] for s in val["station_id"].to_numpy()])
    fin = np.isfinite(y) & np.isfinite(S).all(axis=1)
    S, y, b_codes = S[fin], y[fin], b_codes[fin]
    # per-station NSE bookkeeping on the val slice
    codes, uniq = pd.factorize(b_codes)
    G = codes.max() + 1
    cnt = np.bincount(codes, minlength=G)
    ybar = np.bincount(codes, weights=y, minlength=G) / np.maximum(cnt, 1)
    sst = np.bincount(codes, weights=(y - ybar[codes]) ** 2, minlength=G)
    ok = (cnt >= 5) & (sst > 1e-9)
    Mrow = M[b_codes]                                          # (n_rows, k), M is
    #    indexed by sid_index == b_codes (NOT the factorized `codes`)

    # --- Per-cluster weights by median-NSE differential-evolution (same objective +
    # speed as the global fit_weights, run once per cluster with membership-weighted
    # station NSE). reg blends each cluster's weight toward the GLOBAL optimum so a
    # cluster only deviates where its basin type has real signal (val→test guard).
    # reg blends each cluster toward the global optimum (0=free per-cluster,
    # 1=global only). MEASURED (2026-07-13, 7-member no-q set): the gate ~ties the
    # global fit-weights (pooled ±0.002, day-1 +0.0015) — the current members don't
    # decorrelate enough by static type to exploit. Kept for when the decorrelated
    # combined-loss δHBV members land. Default 0.4 = safe (near-global) blend.
    reg = 0.4 if gate_reg is None else gate_reg

    def fit_one(cluster_w):
        """DE over simplex weights maximizing the membership-weighted median NSE.
        cluster_w=None → the global fit (all stations weight 1)."""
        sw = np.ones(G) if cluster_w is None else np.bincount(
            codes, weights=cluster_w, minlength=G) / np.maximum(cnt, 1)   # per-station

        def neg(wv):
            wv = np.abs(wv); wv = wv / (wv.sum() + 1e-9)
            sim = S @ wv
            sse = np.bincount(codes, weights=(y - sim) ** 2, minlength=G)
            nse = 1.0 - sse[ok] / sst[ok]
            # membership-weighted median NSE over this cluster's stations
            w_ok = sw[ok]
            order = np.argsort(nse)
            cw = np.cumsum(w_ok[order])
            if cw[-1] <= 0:
                return 0.0
            med = nse[order][np.searchsorted(cw, 0.5 * cw[-1])]
            return -med

        r = differential_evolution(neg, [(0.0, 1.0)] * n_mem, seed=seed,
                                   maxiter=25, tol=1e-4, polish=True)
        w = np.abs(r.x); return w / w.sum()

    w_glob = fit_one(None)                                      # global optimum
    W = np.empty((k, n_mem))
    for c in range(k):
        wc = fit_one(M[b_codes, c])
        W[c] = (1 - reg) * wc + reg * w_glob                   # ridge toward global
        W[c] /= W[c].sum()

    print(f"static-gate k={k} reg={reg}, per-cluster weights "
          f"(global: {', '.join(f'{w:.2f}' for w in w_glob)}):")
    for c in range(k):
        print(f"  cluster{c}: " + ", ".join(f"{sc}={W[c,j]:.3f}"
                                             for j, sc in enumerate(sim_cols)))
    # per-basin weights for the FULL basin set
    W_basin = M @ W                                             # (n_basins, n_mem)
    return {s: W_basin[i] for i, s in enumerate(station_ids)}, sim_cols


def combine_gated(dumps, gate: dict) -> pd.DataFrame:
    """Inner-join members; blend sim columns with PER-BASIN weights from the gate."""
    base = dumps[0].rename(columns={"sim": "sim_0"})
    for i, d in enumerate(dumps[1:], 1):
        base = base.merge(d.drop(columns="truth").rename(columns={"sim": f"sim_{i}"}),
                          on=KEY, how="inner")
    sim_cols = [c for c in base.columns if c.startswith("sim_")]
    Smat = base[sim_cols].to_numpy()
    default = np.ones(len(sim_cols)) / len(sim_cols)
    Wrows = np.array([gate.get(s, default) for s in base["station_id"].to_numpy()])
    base["sim"] = (Smat * Wrows).sum(axis=1)
    return base[KEY + ["truth", "sim"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="member dump csv.gz paths (grand-ensemble members)")
    ap.add_argument("--point", default="ymed",
                    help="point column to combine (ymed default; ymean for CMAL)")
    ap.add_argument("--weights", default="",
                    help="comma-separated per-member weights (default equal)")
    ap.add_argument("--fit-weights", action="store_true",
                    help="optimize member weights on the val slice (the beat-0.83 "
                         "lever); overrides --weights. Fit is t0<=--val-end only.")
    ap.add_argument("--static-gate", type=int, default=0, metavar="K",
                    help="per-basin weights conditioned on the 27 static attrs via "
                         "K soft static-clusters, fit on the val slice only "
                         "(overrides --fit-weights/--weights). Try K=3-4.")
    ap.add_argument("--gate-seed", type=int, default=0,
                    help="seed for the static-gate k-means + DE fit")
    ap.add_argument("--gate-reg", type=float, default=None,
                    help="static-gate L2 pull toward the global weight (default: "
                         "auto-select on a held-out val half). 0 = free per-cluster.")
    ap.add_argument("--val-end", default="1998-09-30",
                    help="fit-weights uses windows with t0 on/before this date")
    ap.add_argument("--label", default="grand_ensemble")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = [Path(p) for p in args.dumps]
    for p in paths:
        if not p.exists():
            print(f"FATAL: missing dump {p}", file=sys.stderr)
            return 1
    weights = ([float(x) for x in args.weights.split(",")] if args.weights else None)
    if weights and len(weights) != len(paths):
        print(f"FATAL: {len(weights)} weights for {len(paths)} dumps", file=sys.stderr)
        return 1

    dumps = [load_dump(p, args.point) for p in paths]
    gate = None
    if args.static_gate:
        gate, _ = fit_static_gate(dumps, args.val_end, args.static_gate,
                                  seed=args.gate_seed, gate_reg=args.gate_reg)
        combined = combine_gated(dumps, gate)
    else:
        if args.fit_weights:
            weights = fit_weights(dumps, args.val_end)   # val-only fit (no test leakage)
        combined = combine(dumps, weights)
    n_windows = combined[["station_id", "t0"]].drop_duplicates().shape[0]
    full = score(combined, day1_only=False)
    day1 = score(combined, day1_only=True)

    def med(block, k):
        return round(block[k]["median"], 4) if k in block else None
    print(f"grand ensemble '{args.label}' ({len(paths)} members, {n_windows} windows):")
    print(f"  pooled : NSE {med(full,'nse')}  KGE {med(full,'kge')}  "
          f"log-NSE {med(full,'log_nse')}  FHV {med(full,'fhv')}  "
          f"(scorable {full.get('nse',{}).get('scorable')})")
    print(f"  day-1  : NSE {med(day1,'nse')}  KGE {med(day1,'kge')}")

    out_path = Path(args.out) if args.out else (
        REPO / "benchmarks" / f"combine_{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "label": args.label,
        "members": [str(p) for p in paths],
        "point": args.point,
        "weights": (f"static-gate-k{args.static_gate}" if gate
                    else (weights or "equal")),
        "n_windows": int(n_windows),
        "metrics": {"full": full, "day1": day1},
    }, indent=2))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
