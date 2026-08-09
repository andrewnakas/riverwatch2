#!/usr/bin/env python3
"""Deployable per-basin ensemble combination for the 0.84 push.

The oracle audit (beat_it.py) showed global weighting is dead (NNLS 0.8321 vs
plain mean 0.8298) but the PER-BASIN best-member ceiling is 0.8478. That gap is
the headroom. The paper's per-basin GA weights reach ~0.844 but are fit on TEST
observations -> pure oracle. This script fits per-basin combination rules on
TRAIN-period (1980-95) member predictions only, so the result is deployable.

Combination rules (all fit on train, scored by train-CV, then ONE test eval):
  plain      : equal-weight mean of all streams (the paper's rule)
  topk       : per-basin mean of the k streams with best train NSE   [trimming]
  invmse     : per-basin w_i ∝ train-MSE_i^(-theta), shrunk toward equal by lam
               [Bates-Granger + shrinkage: the 'forecast combination puzzle' fix]
  gate       : soft k-means over the 27 CAMELS statics -> per-cluster weights
  ridge      : per-basin-pooled ridge stacker on member preds

Train-side model selection uses a temporal sub-split of the TRAIN window
(fit 1980-1990 / validate 1990-1995) so hyper-parameters never see test data.

Usage:
  python gate_eval.py                 # train-CV bake-off only (no test eval)
  python gate_eval.py --test-eval RULE # spend one test-metric query on RULE
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_c import load_seed_avg, median_nse

L = Path("gpu1080/dumps")
H = Path("dhbv_dumps")
SUBSET = "data/camels_gauge_ids.json"

# Train-window sub-split for honest hyper-parameter selection.
CV_FIT_END = "1990-09-30"
CV_VAL_START = "1990-10-01"


# ---------------------------------------------------------------- data loading

# Superseded/quarantined dumps that must NEVER enter a stream average.
# `_OLD642` is the discarded weak nldas s111 (medNSE 0.642) that the retrained
# `_NEW` member (0.734) replaced; a bare glob silently averages it back in.
_EXCLUDE_MARKERS = ("_OLD", "_test_", "_bak", "_broken")


def _clean(paths):
    return [p for p in paths if not any(m in p.name for m in _EXCLUDE_MARKERS)]


def _lstm_seeds(forcing, train):
    """All seed dumps for a single-forcing LSTM stream.

    The retrained nldas member is named `_s111_NEW` on the test side but comes
    from run seed 1111, so its TRAIN dump is `_TRAIN_s1111`. Both supersede the
    weak original s111 (medNSE 0.642) and the original must be dropped on BOTH
    sides -- otherwise the train and test streams average different members and
    every train->test comparison is silently invalid.
    """
    if train:
        seeds = _clean(sorted(L.glob(f"camels531ls_{forcing}_nhlstm_TRAIN_s*.csv.gz")))
        if any(p.name.endswith("_TRAIN_s1111.csv.gz") for p in seeds):
            seeds = [p for p in seeds if not p.name.endswith("_TRAIN_s111.csv.gz")]
        return seeds
    seeds = _clean(sorted(p for p in L.glob(f"camels531ls_{forcing}_nhlstm_s*.csv.gz")
                          if "_TRAIN" not in p.name))
    if (L / f"camels531ls_{forcing}_nhlstm_s111_NEW.csv.gz").exists():
        seeds = [p for p in seeds if not p.name.endswith("_s111.csv.gz")]
    return seeds


# LSTMmulti seeds excluded by the TRAIN-SIDE gate: final training-period
# validation NSE (each run's own output.log, 1994-95 slice inside the train
# window) must be >= ~0.90. Healthy runs cluster at 0.911-0.917; these fall far
# outside that band and drag the seed average.
#   s333  val NSE 0.6007  (collapsed; was replaced by s3334)
#   s444  val NSE 0.7646  (weak)
# Decided on train-side evidence only -- excluding a seed for a poor TEST score
# would be test-set selection.
_MULTI_WEAK_SEEDS = ("s333", "s444")


def _multi_seeds(train):
    pat = ("camels531ls_multi_nhlstm_TRAIN_s*.csv.gz" if train
           else "camels531ls_multi_nhlstm_s*.csv.gz")
    seeds = _clean(sorted(p for p in L.glob(pat)
                          if ("_TRAIN" in p.name) == bool(train)))
    return [p for p in seeds
            if not any(p.name.endswith(f"_{s}.csv.gz") for s in _MULTI_WEAK_SEEDS)]


def _aorc_seeds(train):
    """AORC LSTM seed dumps.

    AORC is the 5th forcing: hourly at ~800 m, so tmax/tmin come from the real
    temperature curve (mean diurnal range 11.8 C across our 530 basins) rather
    than being inherited from whatever a daily product supplied. It covers 530 of
    the benchmark 531 -- 13235000 is absent from the HydroShare catchment
    aggregate, and backfilling it from the raw zarr was rejected because
    gauge-point sampling runs ~53% dry against the catchment areal average in
    steep basins (measured on neighbour 13240000).

    The missing basin needs no special handling here: build_merged inner-joins
    the streams, so including AORC drops 13235000 from EVERY stream at once and
    all members stay scored on an identical basin set.
    """
    pat = ("camels531ls_aorc_nhlstm_TRAIN_s*.csv.gz" if train
           else "camels531ls_aorc_nhlstm_s*.csv.gz")
    return _clean(sorted(q for q in L.glob(pat)
                         if ("_TRAIN" in q.name) == bool(train)))


def _dhbv_seeds(forcing, train):
    pat = (f"camels531ls_{forcing}_dhbv_TRAIN_s*.csv.gz" if train
           else f"camels531ls_{forcing}_dhbv_s[0-9]*.csv.gz")
    return _clean(sorted(p for p in H.glob(pat)
                         if ("_TRAIN" in p.name) == bool(train)))


def build_merged(train=False, drop=()):
    """Seed-average each stream, inner-join on (station,date), restrict to 531."""
    streams = {}
    manifest = {}
    for f in ("daymet", "nldas", "maurer"):
        s = _lstm_seeds(f, train)
        if s:
            streams[f"lstm_{f}"] = load_seed_avg(s)
            manifest[f"lstm_{f}"] = [p.name for p in s]
    m = _multi_seeds(train)
    if m:
        streams["lstm_multi"] = load_seed_avg(m)
        manifest["lstm_multi"] = [p.name for p in m]
    if os.environ.get("GATE_WITH_AORC") == "1":
        a = _aorc_seeds(train)
        if a:
            streams["lstm_aorc"] = load_seed_avg(a)
            manifest["lstm_aorc"] = [q.name for q in a]
    for f in ("daymet", "nldas", "maurer"):
        s = _dhbv_seeds(f, train)
        if s:
            streams[f"dhbv_{f}"] = load_seed_avg(s, shift_days=1)
            manifest[f"dhbv_{f}"] = [p.name for p in s]
    # Always show exactly which files back each stream: a stray superseded dump
    # silently averaged into a stream is the easiest way to corrupt every number
    # downstream, and it is invisible unless printed.
    print(f"--- member manifest ({'TRAIN' if train else 'TEST'}) ---", file=sys.stderr)
    for k, v in manifest.items():
        print(f"  {k:14s} n={len(v)}  {v}", file=sys.stderr)
    for d in drop:
        streams.pop(d, None)
    if not streams:
        raise SystemExit(f"no streams found (train={train})")

    merged = None
    for name, df in streams.items():
        d = df[["station_id", "date", "pred"]].rename(columns={"pred": name})
        merged = (df[["station_id", "date", "truth"]].merge(d, on=["station_id", "date"])
                  if merged is None else merged.merge(d, on=["station_id", "date"]))
    ids = set(str(x).zfill(8) for x in json.load(open(SUBSET)).get("531", []))
    merged = merged[merged.station_id.isin(ids)].reset_index(drop=True)
    return merged, list(streams)


# ------------------------------------------------------------ combination rules

def per_basin_stats(df, cols):
    """Per-basin MSE and NSE of each stream (used to fit weights)."""
    recs = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        var = np.var(y)
        mse = np.array([np.mean((y - g[c].to_numpy(float)) ** 2) for c in cols])
        recs[sid] = {"mse": mse, "nse": 1.0 - mse / var}
    return recs


def w_plain(stats, cols, **kw):
    n = len(cols)
    return {s: np.ones(n) / n for s in stats}


def w_topk(stats, cols, k=4, **kw):
    out = {}
    for s, r in stats.items():
        w = np.zeros(len(cols))
        w[np.argsort(-r["nse"])[:k]] = 1.0 / k
        out[s] = w
    return out


def w_invmse(stats, cols, theta=1.0, lam=0.5, **kw):
    """Bates-Granger inverse-MSE weights shrunk toward equal weighting."""
    n = len(cols)
    eq = np.ones(n) / n
    out = {}
    for s, r in stats.items():
        mse = np.maximum(r["mse"], 1e-12)
        w = mse ** (-theta)
        w = w / w.sum()
        out[s] = lam * eq + (1 - lam) * w
    return out


def w_invmse_z(stats, cols, theta=1.0, lam=0.5, **kw):
    """Inverse-MSE weights on WITHIN-MEMBER z-normalized skill.

    Members inflate differently in-sample (δHBV fits the train window at ~0.906
    but generalizes to ~0.723; the LSTMs inflate less). Ranking members by raw
    train MSE therefore partly ranks them by memorization capacity rather than
    generalization, which biases weights toward whichever family overfits hardest.

    Fix: z-score each member's per-basin skill ACROSS basins first. That removes
    every member's own level (its inflation) and keeps only the relative pattern
    of which basins it handles comparatively well -- the part that can transfer.
    """
    n = len(cols)
    eq = np.ones(n) / n
    ids = sorted(stats)
    if not ids:
        return {}
    M = np.array([stats[s]["nse"] for s in ids])          # (basins, members)
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)               # per-member z across basins
    P = np.exp(theta * Z)                                 # softmax-style positive weights
    P = P / P.sum(1, keepdims=True)
    return {s: lam * eq + (1 - lam) * P[i] for i, s in enumerate(ids)}


def w_gate(stats, cols, k=6, lam=0.5, seed=0, **kw):
    """Cluster basins by static attributes; each cluster gets its own weights
    (mean of member-optimal inverse-MSE weights of the basins in it), then blend
    per basin by soft cluster membership. Basins with no statics fall back to
    their own shrunk inverse-MSE weights."""
    X, ids = _static_matrix(sorted(stats))
    if X is None:
        return w_invmse(stats, cols, lam=lam)
    base = w_invmse(stats, cols, lam=0.0)
    from scipy.cluster.vq import kmeans2
    cent, lab = kmeans2(X, k, minit="++", seed=seed)
    # cluster weight = mean of its basins' inverse-MSE weights
    cw = np.zeros((k, len(cols)))
    for c in range(k):
        sel = [ids[i] for i in range(len(ids)) if lab[i] == c]
        cw[c] = (np.mean([base[s] for s in sel], axis=0) if sel
                 else np.ones(len(cols)) / len(cols))
    # soft membership from distance to centroids
    d = np.linalg.norm(X[:, None, :] - cent[None], axis=2) + 1e-9
    memb = (1.0 / d) / (1.0 / d).sum(1, keepdims=True)
    eq = np.ones(len(cols)) / len(cols)
    out = {}
    for i, s in enumerate(ids):
        out[s] = lam * eq + (1 - lam) * (memb[i] @ cw)
    for s in stats:
        out.setdefault(s, eq)
    return out


_STATIC_CACHE = {}


def _static_matrix(station_ids):
    """z-normalized CAMELS static attributes for the given basins."""
    key = tuple(station_ids)
    if key in _STATIC_CACHE:
        return _STATIC_CACHE[key]
    path = Path("data/camels_attrs.json")   # {station_id: {27 Addor attrs}}
    if not path.exists():
        print("WARN: no camels_attrs.json -> gate falls back to invmse", file=sys.stderr)
        _STATIC_CACHE[key] = (None, None)
        return None, None
    a = pd.DataFrame(json.loads(path.read_text())).T
    a.index = a.index.astype(str).str.zfill(8)
    a = a.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    ids = [s for s in station_ids if s in a.index]
    X = a.loc[ids].to_numpy(float)
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd[sd < 1e-9] = 1.0
    X = np.nan_to_num((X - mu) / sd)
    _STATIC_CACHE[key] = (X, ids)
    return X, ids


RULES = {"plain": w_plain, "topk": w_topk, "invmse": w_invmse,
         "invmse_z": w_invmse_z, "gate": w_gate}


def apply_weights(df, cols, weights):
    W = np.array([weights.get(s, np.ones(len(cols)) / len(cols))
                  for s in df.station_id.to_numpy()])
    return (df[cols].to_numpy(float) * W).sum(1)


def ridge_stack(fit_df, apply_df, cols, alpha=10.0, nonneg=False):
    """Global ridge stacker on member predictions (per-basin standardized).

    With nonneg=True the member weights are constrained >= 0 (intercept free).
    Unconstrained ridge gives lstm_daymet a stable -0.19 weight, exploiting its
    0.87 correlation with dhbv_daymet to sharpen the fit by cancellation. That
    is a real in-sample gain but extrapolates badly if the correlation structure
    shifts between periods -- the classic reason forecast-combination practice
    prefers non-negative weights.
    """
    Xf = fit_df[cols].to_numpy(float)
    yf = fit_df["truth"].to_numpy(float)
    # per-basin scale normalization so big basins don't dominate the fit
    sc = fit_df.groupby("station_id")["truth"].transform(
        lambda s: max(s.std(), 1e-6)).to_numpy(float)
    A = np.hstack([Xf / sc[:, None], np.ones((len(Xf), 1))])
    b = yf / sc
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    if nonneg:
        # projected gradient on the ridge objective, members clipped at 0
        G = A.T @ A + reg
        c = A.T @ b
        coef = np.linalg.solve(G, c)
        coef[:-1] = np.maximum(coef[:-1], 0.0)
        step = 1.0 / (np.linalg.eigvalsh(G).max() + 1e-9)
        for _ in range(500):
            coef = coef - step * (G @ coef - c)
            coef[:-1] = np.maximum(coef[:-1], 0.0)
    else:
        coef = np.linalg.solve(G := (A.T @ A + reg), A.T @ b)
    Xa = apply_df[cols].to_numpy(float)
    return np.hstack([Xa, np.ones((len(Xa), 1))]) @ coef, coef


# ------------------------------------------------------------------- train CV

def train_cv(merged_tr, cols):
    """Fit rules on 1980-90, score on 1990-95. Returns leaderboard."""
    fit = merged_tr[merged_tr.date <= CV_FIT_END]
    val = merged_tr[merged_tr.date >= CV_VAL_START]
    print(f"train-CV: fit rows={len(fit)} val rows={len(val)} "
          f"basins={val.station_id.nunique()}")
    stats = per_basin_stats(fit, cols)
    rows = []

    base, _ = median_nse(val.assign(pred=val[cols].to_numpy(float).mean(1)), "pred")
    rows.append(("plain (all streams)", base, {}))

    for k in range(2, len(cols)):
        w = w_topk(stats, cols, k=k)
        m, _ = median_nse(val.assign(pred=apply_weights(val, cols, w)), "pred")
        rows.append((f"topk k={k}", m, {"k": k}))

    # subset selection (trimming) chosen on the FIT slice, scored on the VAL slice
    from itertools import combinations
    best_sub, best_m = None, -9e9
    for k in range(3, len(cols) + 1):
        for sub in combinations(cols, k):
            mm, _ = median_nse(
                fit.assign(pred=fit[list(sub)].to_numpy(float).mean(1)), "pred")
            if mm > best_m:
                best_m, best_sub = mm, sub
    m, _ = median_nse(
        val.assign(pred=val[list(best_sub)].to_numpy(float).mean(1)), "pred")
    rows.append((f"subset({len(best_sub)}) fit-chosen", m, {"subset": best_sub}))
    print(f"  [subset rule] fit-slice best subset = {', '.join(best_sub)}")

    for theta in (0.5, 1.0, 2.0):
        for lam in (0.25, 0.5, 0.75):
            w = w_invmse(stats, cols, theta=theta, lam=lam)
            m, _ = median_nse(val.assign(pred=apply_weights(val, cols, w)), "pred")
            rows.append((f"invmse th={theta} lam={lam}", m,
                         {"theta": theta, "lam": lam}))

    for theta in (0.5, 1.0, 2.0):
        for lam in (0.25, 0.5, 0.75):
            w = w_invmse_z(stats, cols, theta=theta, lam=lam)
            m, _ = median_nse(val.assign(pred=apply_weights(val, cols, w)), "pred")
            rows.append((f"invmse_z th={theta} lam={lam}", m,
                         {"theta": theta, "lam": lam}))

    for k in (4, 6, 8):
        for lam in (0.3, 0.5, 0.7):
            try:
                w = w_gate(stats, cols, k=k, lam=lam)
            except Exception as e:
                print("gate skipped:", e, file=sys.stderr)
                break
            m, _ = median_nse(val.assign(pred=apply_weights(val, cols, w)), "pred")
            rows.append((f"gate k={k} lam={lam}", m, {"k": k, "lam": lam}))

    for alpha in (1.0, 10.0, 100.0):
        try:
            pred, _ = ridge_stack(fit, val, cols, alpha=alpha)
            m, _ = median_nse(val.assign(pred=pred), "pred")
            rows.append((f"ridge a={alpha}", m, {"alpha": alpha}))
        except Exception as e:
            print("ridge skipped:", e, file=sys.stderr)
            break

    rows.sort(key=lambda r: -r[1])
    print("\n=== TRAIN-CV LEADERBOARD (fit 1980-90, scored 1990-95) ===")
    for name, m, _ in rows:
        flag = "  <-- baseline" if name.startswith("plain") else ""
        print(f"  {name:26s} {m:.4f}{flag}")
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-eval", default="",
                    help="rule spec to score on TEST, e.g. 'topk:k=4' or "
                         "'invmse:theta=1.0,lam=0.5' (spends a test query)")
    ap.add_argument("--drop", default="", help="comma-separated streams to drop")
    args = ap.parse_args()
    drop = tuple(x for x in args.drop.split(",") if x)

    merged_tr, cols_tr = build_merged(train=True, drop=drop)
    print(f"TRAIN merged rows={len(merged_tr)} basins={merged_tr.station_id.nunique()}")
    print(f"streams: {cols_tr}")
    rows, _ = train_cv(merged_tr, cols_tr)

    if not args.test_eval:
        print("\n(no --test-eval: no test observations were touched)")
        return

    rule, _, kvs = args.test_eval.partition(":")
    kw = {}
    for kv in filter(None, kvs.split(",")):
        k, v = kv.split("=")
        kw[k] = float(v) if "." in v else int(v)
    merged_te, cols_te = build_merged(train=False, drop=drop)
    assert cols_te == cols_tr, f"stream mismatch {cols_te} vs {cols_tr}"
    stats = per_basin_stats(merged_tr, cols_tr)   # fit on FULL train window
    if rule == "subset":
        # choose the subset on the FULL train window, then score once on test
        from itertools import combinations
        best_sub, best_m = None, -9e9
        for k in range(3, len(cols_tr) + 1):
            for sub in combinations(cols_tr, k):
                mm, _ = median_nse(merged_tr.assign(
                    pred=merged_tr[list(sub)].to_numpy(float).mean(1)), "pred")
                if mm > best_m:
                    best_m, best_sub = mm, sub
        print(f"train-chosen subset ({len(best_sub)}): {', '.join(best_sub)}")
        pred = merged_te[list(best_sub)].to_numpy(float).mean(1)
    elif rule == "ridge":
        pred, _ = ridge_stack(merged_tr, merged_te, cols_tr, **kw)
    else:
        pred = apply_weights(merged_te, cols_te, RULES[rule](stats, cols_tr, **kw))
    m, n = median_nse(merged_te.assign(pred=pred), "pred")
    base, _ = median_nse(
        merged_te.assign(pred=merged_te[cols_te].to_numpy(float).mean(1)), "pred")
    print(f"\n=== TEST EVAL [{args.test_eval}] day-1 medNSE = {m:.4f} "
          f"(basins={n}) | plain mean = {base:.4f} | delta = {m-base:+.4f} ===")


if __name__ == "__main__":
    main()
