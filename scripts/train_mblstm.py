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
)

CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
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


def raw_static(attrs: dict) -> np.ndarray:
    out = []
    for name in STATIC_FEATS:
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
                 enc_vars: list[str], dec_vars: list[str]):
        self.enc_vars, self.dec_vars = enc_vars, dec_vars
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
        sv = np.stack([raw_static(attrs_by_id.get(st["id"], {})) for st in stations])
        self.static_median = np.nanmedian(sv, axis=0)
        self.static_median = np.where(np.isfinite(self.static_median), self.static_median, 0.0)
        filled = np.where(np.isfinite(sv), sv, self.static_median)
        self.static_mean = filled.mean(0)
        self.static_std = np.maximum(filled.std(0), 1e-9)

        self.stations = []
        for st, svec in zip(stations, filled):
            q_train = st["q"][st["dates"] <= train_end]
            v = np.asinh(np.clip(q_train[np.isfinite(q_train)], 0, None))
            if len(v) < 365 or np.std(v) < 1e-6:
                continue
            mu_q, sd_q = float(np.mean(v)), float(np.std(v))
            qa = np.asinh(np.clip(st["q"], 0, None))
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

    def window_index(self, lo: pd.Timestamp | None, hi: pd.Timestamp) -> list[tuple[int, int]]:
        """All (station_idx, t0) where t0 is the last encoder day, targets
        t0+1..t0+HORIZON all <= hi, and (if lo) t0+1 >= lo. Requires >=50%
        q coverage in the encoder window and >=7 valid targets. Vectorized
        with cumulative sums — the naive per-day loop is minutes at 1,900
        stations."""
        out: list[tuple[int, int]] = []
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
            out.extend((si, int(t)) for t in t0s[ok])
        return out

    def sample(self, si: int, t0: int, rng: np.random.Generator | None):
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
        x_dec = np.concatenate([
            st["wx_n"][t0 + 1: t0 + 1 + HORIZON][:, self.dec_cols],
            st["doy"][t0 + 1: t0 + 1 + HORIZON],
            lead,
            np.repeat(sv[None, :], HORIZON, axis=0),
        ], axis=1)
        y = st["q_n"][t0 + 1: t0 + 1 + HORIZON]
        m = st["q_mask"][t0 + 1: t0 + 1 + HORIZON]
        return x_enc, x_dec, y, m

    ar_mask_p = 0.3


def make_batches(corpus, windows, batch, rng, shuffle=True, augment=True):
    order = np.arange(len(windows))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch):
        chunk = [windows[j] for j in order[i: i + batch]]
        xs, xd, ys, ms = zip(*[corpus.sample(si, t0, rng if augment else None) for si, t0 in chunk])
        sis = np.asarray([si for si, _ in chunk])
        yield (np.stack(xs), np.stack(xd), np.stack(ys), np.stack(ms), sis)


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

    enc_vars = COMPAT_VARS if args.compat_vars else ENC_VARS
    dec_vars = COMPAT_VARS if args.compat_vars else DEC_VARS

    files = sorted(CORPUS_DIR.glob("*.csv.gz"))
    if args.limit_stations:
        files = files[: args.limit_stations]
    print(f"loading {len(files)} stations from {CORPUS_DIR} ...")
    stations = [s for s in (load_station(p, enc_vars) for p in files) if s is not None]

    registry = {st["id"]: st for st in json.loads(STATIONS_PATH.read_text())["stations"]}
    attrs_by_id = {
        s["id"]: gages2.enrich_station_attrs(dict(registry.get(s["id"], {"id": s["id"]})))
        for s in stations
    }

    train_end = pd.Timestamp(args.train_end)
    corpus = Corpus(stations, attrs_by_id, train_end, enc_vars, dec_vars)
    print(f"usable stations: {len(corpus.stations)}")

    train_windows = corpus.window_index(None, train_end)
    val_all = corpus.window_index(pd.Timestamp(args.val_start), pd.Timestamp(args.val_end))
    val_windows = val_all[:: args.val_stride]
    print(f"windows: train={len(train_windows)} val={len(val_windows)} (of {len(val_all)})")
    if not train_windows or not val_windows:
        print("not enough data — fetch more corpus first")
        return 1

    # Group train windows by station for balanced per-epoch subsampling.
    by_station: dict[int, list] = {}
    for si, t0 in train_windows:
        by_station.setdefault(si, []).append((si, t0))

    cfg = {
        "enc_vars": enc_vars, "dec_vars": dec_vars, "static_feats": STATIC_FEATS,
        "quantiles": list(QUANTILES), "hidden": args.hidden,
        "horizon": HORIZON, "context": CONTEXT_DAYS,
        "wx_mean": {c: float(v) for c, v in zip(enc_vars, corpus.wx_mean)},
        "wx_std": {c: float(v) for c, v in zip(enc_vars, corpus.wx_std)},
        "static_median": [float(v) for v in corpus.static_median],
        "static_mean": [float(v) for v in corpus.static_mean],
        "static_std": [float(v) for v in corpus.static_std],
        "train_end": args.train_end, "val_range": [args.val_start, args.val_end],
        "n_stations": len(corpus.stations),
        "decoder_forcing": "observed-archive (perfect-forcing caveat for h>3)",
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    quantiles = list(QUANTILES)
    med_i = len(quantiles) // 2

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
                tot += float(pinball(yq, ys_t, ms_t, quantiles, torch) * ms_t.sum())
                num += float(ms_t.sum())
                yh = yq[:, :, med_i].cpu().numpy()
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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        ep_windows = []
        for si, wlist in by_station.items():
            k = min(args.windows_per_station, len(wlist))
            idx = rng.choice(len(wlist), size=k, replace=False)
            ep_windows.extend(wlist[j] for j in idx)
        t0 = time.time()
        tot, num, steps = 0.0, 0.0, 0
        for xs, xd, ys, ms, _ in make_batches(corpus, ep_windows, args.batch, rng):
            xs_t = torch.from_numpy(xs).to(dev); xd_t = torch.from_numpy(xd).to(dev)
            ys_t = torch.from_numpy(ys).to(dev); ms_t = torch.from_numpy(ms).to(dev)
            loss = pinball(model(xs_t, xd_t), ys_t, ms_t, quantiles, torch)
            if not torch.isfinite(loss):
                # One bad batch must not poison the run (h256 diverged to NaN
                # at lr 1e-3) — drop it and keep going.
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        print(f"epoch {ep}/{args.epochs}  train_pinball={tot / max(num, 1):.4f}  "
              f"val_pinball={val_pin:.4f}  val_medNSE(norm-asinh)={val_nse:.3f}  "
              f"steps={steps}  {time.time() - t0:.0f}s{marker}", flush=True)

    print(f"\nbest val pinball {best:.4f} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
