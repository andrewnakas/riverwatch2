#!/usr/bin/env python3
"""Train the v16 multi-basin LSTM (app/mblstm.py) on the local corpus.

Data: data/mblstm/corpus/<id>.csv.gz from scripts/build_mblstm_data.py.
Splits (strict temporal, NWM-archive test era untouched):
    train  : windows whose last target date <= --train-end (2024-12-31)
    val    : windows fully inside [--val-start, --val-end] (calendar 2025)
    2026+  : never seen — reserved for the honest backtest.

Each sample = 365-day encoder window (weather + observed discharge with
missing-mask) + 14-day decoder window (forecastable weather only). Targets
are per-station standardized asinh(q); loss is pinball over (0.1, 0.5, 0.9).
Decoder weather at train time is the observed archive ("perfect forcing") —
flagged in the checkpoint so backtests can report the caveat honestly.

Augmentation: with prob --ar-mask-p the trailing 1-14 days of encoder
discharge are masked (simulates stale gauges), plus light random dropout, so
the member degrades gracefully when USGS data lags.

Usage:
  .venv/bin/python scripts/train_mblstm.py --epochs 8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import gages2  # noqa: E402
from app.mblstm import (  # noqa: E402
    CONTEXT_DAYS, DEC_VARS, ENC_VARS, QUANTILES, STATIC_FEATS, build_model,
    cmal_mean, cmal_nll,
)

CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
GFS_DIR = ROOT / "data" / "mblstm" / "gfs_fcst"
HRRR_DIR = ROOT / "data" / "mblstm" / "hrrr_fcst"
GEFS_DIR = ROOT / "data" / "mblstm" / "gefs_fcst"
ECMWF_DIR = ROOT / "data" / "mblstm" / "ecmwf_fcst"
# Archived-forecast sources usable in --forcing-mix ("perfect" = observed).
MIX_SRC_DIRS = {"gfs": GFS_DIR, "gefs": GEFS_DIR, "ecmwf": ECMWF_DIR}
HORIZON = 14

# Variables shared by the Daymet training corpus and the Open-Meteo serve
# path — the safe pilot feature set (no train/serve distribution gap from
# always-missing channels).
COMPAT_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum",
]


# ---------------------------------------------------------------- corpus ----

def load_station(path: Path, enc_vars: list[str]) -> dict | None:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    idx = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D")
    df = df.set_index("date").reindex(idx)
    q = df["q_cfs"].to_numpy(dtype=np.float64)
    wx = df.reindex(columns=enc_vars).to_numpy(dtype=np.float32)
    return {"id": path.name.split(".")[0], "dates": idx, "q": q, "wx": wx}


# GAGES-II attributes present in data/gages2_attrs.json but unused by the v16
# static set — the --static-set full expansion (topographic wetness, basin
# temperature, soil hydrologic groups, wetland fractions).
STATIC_EXTRAS = [
    "TOPWET", "T_AVG_BASIN",
    "HGA_PCT", "HGB_PCT", "HGC_PCT", "HGD_PCT",
    "EMERGWETNLCD06", "WOODYWETNLCD06",
]


def raw_static(attrs: dict, feats: list[str] = STATIC_FEATS) -> np.ndarray:
    out = []
    for name in feats:
        if name == "log_drain_area":
            da = attrs.get("drain_area_sqmi")
            v = math.log1p(float(da)) if da not in (None, 0) and np.isfinite(da) and da > 0 else np.nan
        else:
            v = attrs.get(name)
            v = float(v) if v is not None and np.isfinite(v) else np.nan
        out.append(v)
    return np.asarray(out, dtype=np.float64)


def doy_sincos(idx: pd.DatetimeIndex) -> np.ndarray:
    ang = 2.0 * np.pi * idx.dayofyear.to_numpy(dtype=np.float32) / 366.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


# --------------------------------------------------------------- dataset ----

class Corpus:
    """Per-station arrays pre-normalized once; windows index into them."""

    def __init__(self, stations: list[dict], attrs_by_id: dict, train_end: pd.Timestamp,
                 enc_vars: list[str], dec_vars: list[str], stats: dict | None = None,
                 q_transform: str = "asinh", static_feats: list[str] = STATIC_FEATS):
        self.enc_vars, self.dec_vars = enc_vars, dec_vars
        self.static_feats = static_feats
        self.q_transform = q_transform
        self.gfs = None  # optional (si, t0) -> normalized decoder forcing override
        if stats is not None:
            # Fine-tune: inherit the base checkpoint's normalization verbatim so
            # the weights see identically-scaled inputs.
            self.wx_mean = np.asarray([stats["wx_mean"][c] for c in enc_vars], dtype=np.float64)
            self.wx_std = np.asarray([max(stats["wx_std"][c], 1e-6) for c in enc_vars], dtype=np.float64)
            self.static_median = np.asarray(stats["static_median"], dtype=np.float64)
            self.static_mean = np.asarray(stats["static_mean"], dtype=np.float64)
            self.static_std = np.maximum(np.asarray(stats["static_std"], dtype=np.float64), 1e-9)
            sv = np.stack([raw_static(attrs_by_id.get(st["id"], {}), static_feats) for st in stations])
            filled = np.where(np.isfinite(sv), sv, self.static_median)
        else:
            # Global weather stats from train-period rows only.
            s = np.zeros(len(enc_vars)); ss = np.zeros(len(enc_vars)); n = np.zeros(len(enc_vars))
            for st in stations:
                m = st["dates"] <= train_end
                w = st["wx"][m].astype(np.float64)
                fin = np.isfinite(w)
                s += np.where(fin, w, 0).sum(0)
                ss += np.where(fin, w * w, 0).sum(0)
                n += fin.sum(0)
            n = np.maximum(n, 1)
            self.wx_mean = s / n
            self.wx_std = np.maximum(np.sqrt(np.maximum(ss / n - self.wx_mean ** 2, 0)), 1e-6)

            # Static stats across stations (raw, NaN-aware).
            sv = np.stack([raw_static(attrs_by_id.get(st["id"], {}), static_feats) for st in stations])
            self.static_median = np.nanmedian(sv, axis=0)
            self.static_median = np.where(np.isfinite(self.static_median), self.static_median, 0.0)
            filled = np.where(np.isfinite(sv), sv, self.static_median)
            self.static_mean = filled.mean(0)
            self.static_std = np.maximum(filled.std(0), 1e-9)

        self.stations = []
        for st, svec in zip(stations, filled):
            q_train = st["q"][st["dates"] <= train_end]
            v = np.clip(q_train[np.isfinite(q_train)], 0, None)
            qa = np.clip(st["q"], 0, None)
            if q_transform == "asinh":
                v, qa = np.asinh(v), np.asinh(qa)
            if len(v) < 365 or np.std(v) < 1e-6:
                continue
            mu_q, sd_q = float(np.mean(v)), float(np.std(v))
            q_mask = np.isfinite(qa).astype(np.float32)
            q_n = np.nan_to_num((qa - mu_q) / sd_q, nan=0.0).astype(np.float32)
            wx_n = np.nan_to_num(
                (st["wx"].astype(np.float64) - self.wx_mean) / self.wx_std, nan=0.0
            ).astype(np.float32)
            self.stations.append({
                "id": st["id"], "dates": st["dates"],
                "q_n": q_n, "q_mask": q_mask, "wx_n": wx_n,
                "doy": doy_sincos(st["dates"]),
                "static": ((svec - self.static_mean) / self.static_std).astype(np.float32),
            })
        self.dec_cols = np.asarray([enc_vars.index(c) for c in dec_vars])

    def window_index(self, lo: pd.Timestamp | None, hi: pd.Timestamp) -> np.ndarray:
        """(N, 2) int32 array of (station_idx, t0) where t0 is the last encoder
        day, targets t0+1..t0+HORIZON all <= hi, and (if lo) t0+1 >= lo.
        Requires >=50% q coverage in the encoder window and >=7 valid targets.
        Vectorized with cumulative sums — the naive per-day loop is minutes at
        1,900 stations. Arrays, not tuples: 18.7M window tuples cost ~2.8GB of
        CPython overhead, which became swap-death on 16GB machines."""
        chunks = []
        for si, st in enumerate(self.stations):
            T = len(st["dates"])
            if T < CONTEXT_DAYS + HORIZON + 1:
                continue
            c = np.concatenate([[0.0], np.cumsum(st["q_mask"], dtype=np.float64)])
            t0s = np.arange(CONTEXT_DAYS - 1, T - HORIZON)
            ctx_cov = (c[t0s + 1] - c[t0s + 1 - CONTEXT_DAYS]) / CONTEXT_DAYS
            tgt_cnt = c[t0s + 1 + HORIZON] - c[t0s + 1]
            ok = (ctx_cov >= 0.5) & (tgt_cnt >= 7) & (st["dates"][t0s + HORIZON] <= hi)
            if lo is not None:
                ok &= st["dates"][t0s + 1] >= lo
            sel = t0s[ok]
            if len(sel):
                chunks.append(np.stack(
                    [np.full(len(sel), si, dtype=np.int32), sel.astype(np.int32)], axis=1))
        if not chunks:
            return np.empty((0, 2), dtype=np.int32)
        return np.concatenate(chunks)

    def sample(self, si: int, t0: int, rng: np.random.Generator | None, src: int = 0):
        st = self.stations[si]
        a = t0 - CONTEXT_DAYS + 1
        q_n = st["q_n"][a: t0 + 1].copy()
        q_mask = st["q_mask"][a: t0 + 1].copy()
        if rng is not None:
            if rng.random() < self.ar_mask_p:
                k = int(rng.integers(1, HORIZON + 1))
                q_n[-k:] = 0.0
                q_mask[-k:] = 0.0
            if rng.random() < 0.1:
                drop = rng.random(CONTEXT_DAYS) < 0.1
                q_n[drop] = 0.0
                q_mask[drop] = 0.0
        sv = st["static"]
        x_enc = np.concatenate([
            st["wx_n"][a: t0 + 1],
            q_n[:, None], q_mask[:, None],
            st["doy"][a: t0 + 1],
            np.repeat(sv[None, :], CONTEXT_DAYS, axis=0),
        ], axis=1)
        lead = (np.arange(1, HORIZON + 1, dtype=np.float32) / HORIZON)[:, None]
        if src > 0 and self.mix_lookups is not None:
            # Forcing-mix sample: decoder sees an archived real forecast
            # (already normalized at load). Window membership was filtered
            # against the lookup, so the key must exist.
            dec_wx = self.mix_lookups[src][(int(si), int(t0))]
        elif self.gfs is not None:
            dec_wx = self.gfs[(si, t0)]
        else:
            dec_wx = st["wx_n"][t0 + 1: t0 + 1 + HORIZON][:, self.dec_cols]
            if rng is not None and self.forcing_noise > 0:
                # Lead-scaled synthetic forecast error for perfect-forcing
                # samples (z-space, sigma grows linearly with lead) — cheap
                # robustness for windows with no archived forecast.
                sig = self.forcing_noise * lead
                dec_wx = dec_wx + (rng.normal(0.0, 1.0, dec_wx.shape) * sig).astype(np.float32)
        x_dec = np.concatenate([
            dec_wx,
            st["doy"][t0 + 1: t0 + 1 + HORIZON],
            lead,
            np.repeat(sv[None, :], HORIZON, axis=0),
        ], axis=1)
        y = st["q_n"][t0 + 1: t0 + 1 + HORIZON]
        m = st["q_mask"][t0 + 1: t0 + 1 + HORIZON]
        return x_enc, x_dec, y, m

    ar_mask_p = 0.3
    forcing_noise = 0.0
    mix_lookups: list | None = None


def _parse_forcing_file(p: Path, dec_vars: list[str], n_leads: int):
    """One extracted init csv -> (station ids, (n, n_leads, n_vars) raw array).
    Stations with incomplete lead coverage are dropped."""
    try:
        df = pd.read_csv(p, dtype={"station_id": str})
    except Exception:
        return None, None  # truncated/corrupt extraction — refetch will replace it
    df = df.sort_values(["station_id", "lead_day"])
    counts = df.groupby("station_id", sort=False).size()
    full = counts[counts == n_leads].index
    df = df[df["station_id"].isin(full)]
    if not len(full):
        return None, None
    return full, df[dec_vars].to_numpy(dtype=np.float32).reshape(len(full), n_leads, -1)


def load_gfs_windows(corpus, use_hrrr: bool = False, src_dir: Path = GFS_DIR,
                     label: str = "GFS") -> dict[tuple[int, int], np.ndarray]:
    """(station_idx, t0) -> normalized (HORIZON, n_dec_vars) archived-forecast
    decoder forcing from any fetcher-schema directory (GFS/GEFS/ECMWF). A 00z
    init on day D has lead_day 1 = calendar day D, so the matching issue date
    is t0 = D-1 (observations through yesterday, today's 00z run — the serving
    setup). Only complete 14-lead extractions are kept. With use_hrrr, lead
    days 1-2 are overlaid with the 3 km HRRR forecast where extracted."""
    si_by_id = {st["id"]: si for si, st in enumerate(corpus.stations)}
    mu = corpus.wx_mean[corpus.dec_cols]
    sd = corpus.wx_std[corpus.dec_cols]

    def norm(a):
        return np.nan_to_num((a - mu) / sd, nan=0.0).astype(np.float32)

    def t0_of(si, init):
        st = corpus.stations[si]
        t0 = int((init - pd.Timedelta(days=1) - st["dates"][0]).days)
        return t0 if 0 <= t0 < len(st["dates"]) else None

    lookup: dict[tuple[int, int], np.ndarray] = {}
    files = sorted(p for p in src_dir.glob("*.csv.gz")
                   if not p.name.startswith("._") and ".members." not in p.name)
    for p in files:
        init = pd.Timestamp(p.name.split(".")[0])
        sids, arr = _parse_forcing_file(p, corpus.dec_vars, HORIZON)
        if sids is None:
            continue
        for sid, a in zip(sids, norm(arr)):
            si = si_by_id.get(sid)
            if si is not None and (t0 := t0_of(si, init)) is not None:
                lookup[(si, t0)] = a
    n_hrrr = 0
    if use_hrrr:
        for p in sorted(HRRR_DIR.glob("*.csv.gz")):
            if p.name.startswith("._") or ".members." in p.name:
                continue
            init = pd.Timestamp(p.name.split(".")[0])
            sids, arr = _parse_forcing_file(p, corpus.dec_vars, 2)
            if sids is None:
                continue
            for sid, a in zip(sids, norm(arr)):
                si = si_by_id.get(sid)
                if si is not None and (t0 := t0_of(si, init)) is not None \
                        and (si, t0) in lookup:
                    lookup[(si, t0)][:2] = a
                    n_hrrr += 1
    print(f"{label} decoder forcings: {len(files)} inits -> {len(lookup)} usable "
          f"(station, t0) pairs" + (f"; HRRR d1-2 overlay on {n_hrrr}" if use_hrrr else ""))
    return lookup


def make_batches(corpus, windows, batch, rng, shuffle=True, augment=True):
    order = np.arange(len(windows))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch):
        chunk = windows[order[i: i + batch]]
        # Windows are (si, t0) or (si, t0, src) — src selects the decoder
        # forcing source under --forcing-mix (0 = perfect/observed).
        srcs = chunk[:, 2] if chunk.shape[1] > 2 else np.zeros(len(chunk), np.int32)
        xs, xd, ys, ms = zip(*[
            corpus.sample(int(r[0]), int(r[1]), rng if augment else None, src=int(s))
            for r, s in zip(chunk, srcs)])
        yield (np.stack(xs), np.stack(xd), np.stack(ys), np.stack(ms), chunk[:, 0])


# -------------------------------------------------------------- training ----

def pinball(yq, y, m, quantiles, torch):
    """yq: (B,H,Q) predicted; y,m: (B,H). Masked mean pinball."""
    losses = []
    for qi, tau in enumerate(quantiles):
        e = y - yq[:, :, qi]
        losses.append(torch.maximum(tau * e, (tau - 1) * e))
    L = torch.stack(losses, dim=-1).mean(-1)
    return (L * m).sum() / m.sum().clamp(min=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--val-start", default="2025-01-01")
    ap.add_argument("--val-end", default="2025-12-31")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--windows-per-station", type=int, default=300)
    ap.add_argument("--val-stride", type=int, default=10)
    ap.add_argument("--limit-stations", type=int, default=0)
    ap.add_argument("--compat-vars", action="store_true",
                    help="train on the Daymet/Open-Meteo shared variable set")
    ap.add_argument("--enc-vars", choices=["full", "compat"], default="",
                    help="encoder weather set (full = 13-var Open-Meteo, compat "
                         "= 5-var). Overrides --compat-vars for the encoder.")
    ap.add_argument("--dec-vars", choices=["full", "compat"], default="",
                    help="decoder weather set. Use compat with --enc-vars full "
                         "so the om13 model stays fine-tunable/servable against "
                         "the 5-var GFS/GEFS/ECMWF forecast archives.")
    ap.add_argument("--gfs-finetune", action="store_true",
                    help="fine-tune --init-ckpt with decoder forcings from archived "
                         "GFS forecasts (real forecast error) instead of observed weather")
    ap.add_argument("--forcing-mix", default="",
                    help="per-sample decoder forcing source mix, e.g. "
                         "'perfect:0.25,gfs:0.4,gefs:0.35'. Forecast-source "
                         "samples draw from windows with an archived init; "
                         "perfect samples draw from all windows. Validation "
                         "runs under real GFS forcing. Subsumes --gfs-finetune.")
    ap.add_argument("--forcing-noise", type=float, default=0.0,
                    help="with --forcing-mix: z-space sigma at lead 14 for "
                         "lead-scaled Gaussian noise on perfect-forcing "
                         "samples (0 = off)")
    ap.add_argument("--static-set", choices=["v16", "full"], default="v16",
                    help="static catchment features: v16 = the 14-feature set, "
                         "full = +8 GAGES-II extras (TOPWET, T_AVG_BASIN, soil "
                         "hydro groups, wetlands). Fine-tunes inherit the base "
                         "ckpt's set.")
    ap.add_argument("--q-transform", choices=["asinh", "linear"], default="asinh",
                    help="per-station discharge transform before z-scoring. "
                         "linear = no asinh compression (peak-gradient "
                         "ablation); fine-tunes inherit the base ckpt's.")
    ap.add_argument("--init-ckpt", default="",
                    help="checkpoint to start from: with --gfs-finetune the GFS "
                         "fine-tune base, otherwise a plain warm start (resume "
                         "after a crash without losing trained epochs)")
    ap.add_argument("--hrrr", action="store_true",
                    help="with --gfs-finetune: overlay 3km HRRR on decoder lead "
                         "days 1-2 (hybrid forcing)")
    ap.add_argument("--head", choices=["quantile", "cmal"], default="quantile",
                    help="probabilistic head: 'quantile' (pinball, legacy) or "
                         "'cmal' (mixture of asymmetric Laplacians, NLL — "
                         "sharper right-skewed peaks, analytic quantiles)")
    ap.add_argument("--cmal-k", type=int, default=3,
                    help="number of mixture components for --head cmal")
    ap.add_argument("--corpus-dir", default="",
                    help="override corpus dir (e.g. data/mblstm/corpus_openmeteo for "
                         "the full-13-var Open-Meteo corpus). Default: data/mblstm/corpus")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "data" / "mblstm" / "model.pt"))
    args = ap.parse_args()

    import torch

    dev = args.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    base_payload = None
    if args.gfs_finetune and not args.init_ckpt:
        print("--gfs-finetune requires --init-ckpt")
        return 1
    if args.init_ckpt:
        base_payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        base_cfg = base_payload["cfg"]
        enc_vars, dec_vars = base_cfg["enc_vars"], base_cfg["dec_vars"]
        if args.gfs_finetune and not set(dec_vars) <= set(COMPAT_VARS):
            print(f"GFS archive only carries {COMPAT_VARS}; checkpoint decoder "
                  f"wants {dec_vars} — fine-tune only supports compat-vars checkpoints")
            return 1
        args.hidden = int(base_cfg["hidden"])
    else:
        enc_vars = COMPAT_VARS if args.compat_vars else ENC_VARS
        dec_vars = COMPAT_VARS if args.compat_vars else DEC_VARS
        # Per-side override: enc-13/dec-5 keeps the encoder's soil/snow/wind
        # signal while the decoder stays drivable by the 5-var forecast
        # archives (a full-vars decoder could never be forcing-fine-tuned).
        if args.enc_vars:
            enc_vars = ENC_VARS if args.enc_vars == "full" else COMPAT_VARS
        if args.dec_vars:
            dec_vars = DEC_VARS if args.dec_vars == "full" else COMPAT_VARS
        if not set(dec_vars) <= set(enc_vars):
            print("--dec-vars must be a subset of --enc-vars (decoder columns "
                  "are indexed out of the encoder array)")
            return 1

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else CORPUS_DIR
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    files = sorted(p for p in corpus_dir.glob("*.csv.gz") if not p.name.startswith("._"))
    if not files:
        print(f"no corpus files in {corpus_dir}")
        return 1
    # Guard: the corpus must actually carry every requested forcing variable,
    # else a full-13-var train against a 5-var corpus would silently train on
    # all-NaN→0 channels. Check the first file's columns up front.
    have_cols = set(pd.read_csv(files[0], nrows=1).columns)
    missing = [v for v in enc_vars if v not in have_cols]
    if missing:
        print(f"corpus {corpus_dir.name} is missing requested vars {missing}; "
              f"available={sorted(have_cols - {'date','q_cfs'})}. Use --compat-vars "
              f"or point --corpus-dir at a corpus with these columns.")
        return 1
    if args.limit_stations:
        files = files[: args.limit_stations]
    print(f"loading {len(files)} stations from {corpus_dir} (enc_vars={len(enc_vars)}) ...")
    stations = [s for s in (load_station(p, enc_vars) for p in files) if s is not None]

    registry = {st["id"]: st for st in json.loads(STATIONS_PATH.read_text())["stations"]}
    attrs_by_id = {
        s["id"]: gages2.enrich_station_attrs(dict(registry.get(s["id"], {"id": s["id"]})))
        for s in stations
    }

    train_end = pd.Timestamp(args.train_end)
    q_transform = (base_payload["cfg"].get("q_transform", "asinh")
                   if base_payload else args.q_transform)
    static_feats = (list(base_payload["cfg"]["static_feats"]) if base_payload
                    else (STATIC_FEATS + STATIC_EXTRAS if args.static_set == "full"
                          else STATIC_FEATS))
    corpus = Corpus(stations, attrs_by_id, train_end, enc_vars, dec_vars,
                    stats=base_payload["cfg"] if base_payload else None,
                    q_transform=q_transform, static_feats=static_feats)
    print(f"usable stations: {len(corpus.stations)}")

    train_windows = corpus.window_index(None, train_end)
    val_all = corpus.window_index(pd.Timestamp(args.val_start), pd.Timestamp(args.val_end))

    def _filter_in(wins, lookup):
        keep = np.fromiter(((int(si), int(t0)) in lookup for si, t0 in wins[:, :2]),
                           dtype=bool, count=len(wins))
        return wins[keep]

    mix: list[tuple[str, float]] = []
    mix_pools: dict[int, np.ndarray] = {}
    if args.forcing_mix:
        if args.gfs_finetune:
            print("--forcing-mix subsumes --gfs-finetune; use one or the other")
            return 1
        for part in args.forcing_mix.split(","):
            name, frac = part.split(":")
            if name != "perfect" and name not in MIX_SRC_DIRS:
                print(f"unknown mix source {name!r} (have perfect,{','.join(MIX_SRC_DIRS)})")
                return 1
            mix.append((name, float(frac)))
        if abs(sum(f for _, f in mix) - 1.0) > 1e-6:
            print("--forcing-mix fractions must sum to 1")
            return 1
        # src code 0 = perfect; forecast sources get codes 1.. in mix order.
        src_names = ["perfect"] + [n for n, _ in mix if n != "perfect"]
        corpus.mix_lookups = [None] + [
            load_gfs_windows(corpus, src_dir=MIX_SRC_DIRS[n], label=n.upper())
            for n in src_names[1:]]
        corpus.forcing_noise = args.forcing_noise
        for code, n in enumerate(src_names[1:], start=1):
            pool = _filter_in(train_windows, corpus.mix_lookups[code])
            if not len(pool):
                print(f"mix source {n} has no archived train windows — fetch first")
                return 1
            mix_pools[code] = pool
            print(f"mix source {n}: {len(pool)} archived train windows")
        # Validate under real GFS forcing (the deployed task): prefer gfs if
        # mixed, else the first forecast source.
        vname = "gfs" if "gfs" in src_names else src_names[1]
        vcode = src_names.index(vname)
        val_all = _filter_in(val_all, corpus.mix_lookups[vcode])
        val_all = np.column_stack(
            [val_all, np.full(len(val_all), vcode, dtype=np.int32)])
        print(f"validation forcing: {vname} ({len(val_all)} windows)")
    elif args.gfs_finetune:
        # Keep only windows whose issue date has an archived GFS init.
        corpus.gfs = load_gfs_windows(corpus, use_hrrr=args.hrrr)
        train_windows = _filter_in(train_windows, corpus.gfs)
        val_all = _filter_in(val_all, corpus.gfs)
    val_windows = val_all[:: args.val_stride]
    print(f"windows: train={len(train_windows)} val={len(val_windows)} (of {len(val_all)})")
    if len(train_windows) == 0 or len(val_windows) == 0:
        print("not enough data — fetch more corpus first")
        return 1

    # Group train windows by station for balanced per-epoch subsampling.
    by_station = {int(si): train_windows[train_windows[:, 0] == si]
                  for si in np.unique(train_windows[:, 0])}

    if base_payload is not None:
        cfg = dict(base_payload["cfg"])
        base_head = cfg.get("head", "quantile")
        # The CLI head wins so you can warm-start a quantile checkpoint into a
        # cmal head (encoder+decoder transfer, head re-inits — handled by the
        # strict=False load below). Changing head is only meaningful as a plain
        # warm start, not a forcing-only fine-tune of identical weights.
        cfg["head"] = args.head
        if args.head == "cmal":
            cfg["cmal_k"] = int(args.cmal_k)
        else:
            cfg.pop("cmal_k", None)
        cfg["head_changed_from"] = base_head if base_head != args.head else None
        if args.forcing_mix:
            cfg["decoder_forcing"] = (f"forcing-mix {args.forcing_mix} "
                                      f"(dynamical.org archives + perfect"
                                      + (f", noise {args.forcing_noise}" if args.forcing_noise else "")
                                      + "; real forecast error)")
            cfg["finetuned_from"] = args.init_ckpt
        elif args.gfs_finetune:
            cfg["decoder_forcing"] = (
                "archived HRRR d1-2 + GFS d3-14 hybrid "
                "(dynamical.org, real forecast error)" if args.hrrr else
                "archived GFS forecasts (dynamical.org, real forecast error)")
            cfg["finetuned_from"] = args.init_ckpt
        else:
            cfg["warmstarted_from"] = args.init_ckpt
        cfg["trained_at"] = pd.Timestamp.utcnow().isoformat()
    else:
        cfg = {
            "enc_vars": enc_vars, "dec_vars": dec_vars, "static_feats": static_feats,
            "quantiles": list(QUANTILES), "hidden": args.hidden,
            "head": args.head,
            "horizon": HORIZON, "context": CONTEXT_DAYS,
            "wx_mean": {c: float(v) for c, v in zip(enc_vars, corpus.wx_mean)},
            "wx_std": {c: float(v) for c, v in zip(enc_vars, corpus.wx_std)},
            "static_median": [float(v) for v in corpus.static_median],
            "static_mean": [float(v) for v in corpus.static_mean],
            "static_std": [float(v) for v in corpus.static_std],
            "train_end": args.train_end, "val_range": [args.val_start, args.val_end],
            "n_stations": len(corpus.stations),
            "q_transform": q_transform,
            "decoder_forcing": "observed-archive (perfect-forcing caveat for h>3)",
            "trained_at": pd.Timestamp.utcnow().isoformat(),
        }
        if args.head == "cmal":
            cfg["cmal_k"] = int(args.cmal_k)
        if args.forcing_mix:
            cfg["decoder_forcing"] = (f"forcing-mix {args.forcing_mix} "
                                      f"(dynamical.org archives + perfect"
                                      + (f", noise {args.forcing_noise}" if args.forcing_noise else "")
                                      + "; real forecast error)")
    model = build_model(cfg).to(dev)
    if base_payload is not None:
        # Warm start. If the head changed (e.g. quantile -> cmal) the head
        # Linear shapes differ. strict=False ignores missing/unexpected keys but
        # NOT shape mismatches on shared keys (head.2.* exists in both), so first
        # drop any incoming tensor whose shape doesn't match the current model,
        # then load — encoder+decoder transfer, the new head stays random-init.
        cur = model.state_dict()
        src = base_payload["state_dict"]
        compatible = {k: v for k, v in src.items()
                      if k in cur and cur[k].shape == v.shape}
        dropped = [k for k in src if k not in compatible]
        res = model.load_state_dict(compatible, strict=False)
        print(f"warm start: loaded {len(compatible)}/{len(src)} tensors "
              f"(head re-init); dropped shape-mismatch={dropped}; "
              f"still-random={list(res.missing_keys)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    quantiles = list(QUANTILES)
    med_i = len(quantiles) // 2
    head = cfg.get("head", "quantile")
    # Lower-is-better val objective and point estimate are head-specific:
    #   quantile -> pinball loss, median as the point;
    #   cmal     -> NLL,           mixture mean as the point.
    loss_name = "nll" if head == "cmal" else "pinball"

    def head_loss(out, y, m):
        if head == "cmal":
            return cmal_nll(out, y, m)
        return pinball(out, y, m, quantiles, torch)

    def head_point(out):
        # (B,H) z-space point estimate as numpy, on whatever device `out` is.
        if head == "cmal":
            return cmal_mean(out, lib=torch).cpu().numpy()
        return out[:, :, med_i].cpu().numpy()

    def run_val():
        model.eval()
        tot, num = 0.0, 0.0
        sse: dict[int, float] = {}; sst_y: dict[int, list] = {}
        preds: dict[int, list] = {}
        with torch.no_grad():
            for xs, xd, ys, ms, sis in make_batches(corpus, val_windows, args.batch, rng, shuffle=False, augment=False):
                xs_t = torch.from_numpy(xs).to(dev); xd_t = torch.from_numpy(xd).to(dev)
                ys_t = torch.from_numpy(ys).to(dev); ms_t = torch.from_numpy(ms).to(dev)
                yq = model(xs_t, xd_t)
                tot += float(head_loss(yq, ys_t, ms_t) * ms_t.sum())
                num += float(ms_t.sum())
                yh = head_point(yq)
                for b in range(len(sis)):
                    si = int(sis[b]); m = ms[b] > 0
                    preds.setdefault(si, []).append(yh[b][m])
                    sst_y.setdefault(si, []).append(ys[b][m])
        nses = []
        for si in preds:
            yh = np.concatenate(preds[si]); yt = np.concatenate(sst_y[si])
            if len(yt) < 20 or np.var(yt) < 1e-9:
                continue
            nses.append(1.0 - np.mean((yt - yh) ** 2) / np.var(yt))
        model.train()
        return tot / max(num, 1), (float(np.median(nses)) if nses else float("nan"))

    best = float("inf")
    if base_payload is not None:
        # Don't let a weak first epoch overwrite the loaded checkpoint when
        # --out == --init-ckpt: the bar starts at the loaded model's own val.
        best, nse0 = run_val()
        print(f"init-ckpt val_{loss_name}={best:.4f}  val_medNSE={nse0:.3f}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frac_perfect = next((f for n, f in mix if n == "perfect"), 0.0) if mix else 1.0
    mix_fracs = {("perfect" if n == "perfect" else n): f for n, f in mix} if mix else {}
    for ep in range(1, args.epochs + 1):
        if mix:
            # Perfect-forcing share: the usual balanced per-station draw,
            # scaled by its mix fraction. Forecast shares: uniform draws from
            # each source's archived-window pool (stations have near-equal
            # archived init counts, so balance holds without grouping).
            parts = []
            n_total = sum(min(args.windows_per_station, len(w)) for w in by_station.values())
            if frac_perfect > 0:
                per_st = max(1, round(args.windows_per_station * frac_perfect))
                base = np.concatenate([
                    wlist[rng.choice(len(wlist), size=min(per_st, len(wlist)), replace=False)]
                    for wlist in by_station.values()])
                parts.append(np.column_stack(
                    [base, np.zeros(len(base), dtype=np.int32)]))
            src_names_ep = ["perfect"] + [n for n, _ in mix if n != "perfect"]
            for code, pool in mix_pools.items():
                frac = mix_fracs.get(src_names_ep[code], 0.0)
                n_src = int(round(n_total * frac))
                if n_src == 0:
                    continue
                idx = rng.choice(len(pool), size=n_src, replace=len(pool) < n_src)
                parts.append(np.column_stack(
                    [pool[idx], np.full(n_src, code, dtype=np.int32)]))
            ep_windows = np.concatenate(parts)
        else:
            ep_windows = np.concatenate([
                wlist[rng.choice(len(wlist), size=min(args.windows_per_station, len(wlist)),
                                 replace=False)]
                for wlist in by_station.values()])
        t0 = time.time()
        tot, num, steps, skipped = 0.0, 0.0, 0, 0
        for xs, xd, ys, ms, _ in make_batches(corpus, ep_windows, args.batch, rng):
            xs_t = torch.from_numpy(xs).to(dev); xd_t = torch.from_numpy(xd).to(dev)
            ys_t = torch.from_numpy(ys).to(dev); ms_t = torch.from_numpy(ms).to(dev)
            loss = head_loss(model(xs_t, xd_t), ys_t, ms_t)
            if not torch.isfinite(loss):
                # One bad batch must not poison the run (h256 diverged to NaN
                # at lr 1e-3) — drop it and keep going.
                opt.zero_grad()
                skipped += 1
                continue
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                # MPS LSTM backward sporadically emits NaN grads on a finite
                # loss (torch 2.11) — stepping would poison the weights for
                # the rest of the run. Skip the step, keep the loss stats.
                opt.zero_grad()
                skipped += 1
                continue
            opt.step()
            tot += float(loss.detach()) * float(ms_t.sum()); num += float(ms_t.sum()); steps += 1
        sched.step()
        val_pin, val_nse = run_val()
        marker = ""
        if val_pin < best:
            best = val_pin
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "cfg": cfg}, out_path)
            marker = "  *saved*"
        print(f"epoch {ep}/{args.epochs}  train_{loss_name}={tot / max(num, 1):.4f}  "
              f"val_{loss_name}={val_pin:.4f}  val_medNSE(norm-asinh)={val_nse:.3f}  "
              f"steps={steps}  skipped={skipped}  {time.time() - t0:.0f}s{marker}", flush=True)

    print(f"\nbest val {loss_name} {best:.4f} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
