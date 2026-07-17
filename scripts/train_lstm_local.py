#!/usr/bin/env python3
"""Standalone seq-to-one rainfall-runoff LSTM — the Kratzert/Li-Shen reference
recipe — trained LOCALLY on Apple MPS (or CPU). Faithful to neuralhydrology's
CudaLSTM: it IS a plain single-layer nn.LSTM(hidden=256) + dropout + linear head,
seq-to-one (predict the last day), NSE loss, 5 forcings + 27 statics, target =
specific discharge (mm/day). We reimplement it standalone because NH's heavy
CUDA-centric dep tree is fragile on an 8 GB M1; the model itself is simple.

Recipe (verified vs Kratzert 2021 + Li/Shen 2025 HESS 29:6829, Table D1):
  split : train 1999-10-01→2008-09-30, test 1989-10-01→1999-09-30
  model : LSTM hidden 256, seq 365, predict last 1, dropout 0.4 on FC input
  loss  : basin-normalized NSE loss (per-sample /(σ_basin+ε)² weighting)
  inputs: [prcp,tmax,tmin,vp,srad] + 27 Addor statics ; target q_mm
  opt   : Adam, LR 1e-3→5e-4@ep20→1e-4@ep25, 30 epochs, batch 128 (8GB-safe)

8 GB notes: batch 128 (not 256), float32, one forcing's corpus at a time, windows
built lazily per-basin then stacked. Reads corpus_<forcing>_v2/<id>.csv.gz directly
(download one forcing locally first). Writes a dump in the combine_dumps stride-14
grid (cfs, zero-padded station ids) so it joins the δHBV members.

Usage (one member):
  python scripts/train_lstm_local.py --forcing nldas --seed 111 \
    --corpus-dir /path/camels_corpus_nldas_v2 \
    --out data/mblstm/gpu_dumps_s14/camels531_nldas_nhlstm_s111.csv.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

FORCING_MAP = {
    "precipitation_sum": "prcp", "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin", "vapor_pressure": "vp",
    "shortwave_radiation_sum": "srad",
}
FORCINGS = list(FORCING_MAP.values())
STATIC_ATTRS = [
    "p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2", "soil_depth_pelletier",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
    "max_water_content", "sand_frac", "silt_frac", "clay_frac",
    "frac_forest", "lai_max", "gvf_max", "gvf_diff", "root_depth_50",
    "carbonate_rocks_frac", "geol_permeability",
]
CFS_TO_MMDAY_PER_KM2 = 0.0283168 * 86400 / 1e6 * 1000   # 2.446576
SEQ = 365
TRAIN = ("1999-10-01", "2008-09-30")
TEST = ("1989-10-01", "1999-09-30")


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_531() -> list[str]:
    d = json.loads((ROOT / "data" / "camels_gauge_ids.json").read_text())
    return [str(x).strip().zfill(8) for x in d["531"]]


class LSTMRegressor(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_dyn + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_d, x_s):
        # x_d: (B,SEQ,n_dyn); x_s: (B,n_static) broadcast over time
        B, T, _ = x_d.shape
        xs = x_s.unsqueeze(1).expand(B, T, x_s.shape[-1])
        out, _ = self.lstm(torch.cat([x_d, xs], dim=-1))
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)   # last step → scalar


def build_basin(csv_path: Path, area: float):
    """Return (dates, dyn[T,5], q_mm[T], static_raw[27]) for one basin, or None."""
    df = pd.read_csv(csv_path)
    if not {*FORCING_MAP, "q_cfs", "date"} <= set(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    idx = pd.date_range(df.index[0], df.index[-1], freq="D")
    df = df.reindex(idx)
    dyn = np.stack([df[src].to_numpy("float32") for src in FORCING_MAP], axis=1)
    q_mm = df["q_cfs"].to_numpy("float64") * CFS_TO_MMDAY_PER_KM2 / area
    return idx, dyn, q_mm.astype("float32")


def window_ends(idx, dyn, q_mm, lo, hi):
    """Index-only seq-to-one windows: return list of (end_j, target_date) whose
    TARGET day falls in [lo,hi] and whose 365-day window is complete + finite.
    LAZY — we do NOT materialize the (SEQ,5) arrays here (that OOMs an 8 GB box);
    the training loop slices dyn[j-SEQ+1:j+1] on the fly per batch."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    pos = {d: i for i, d in enumerate(idx)}
    out = []
    days = pd.date_range(max(lo, idx[0] + pd.Timedelta(days=SEQ - 1)), min(hi, idx[-1]),
                         freq="D")
    for d in days:
        j = pos.get(d)
        if j is None or j < SEQ - 1:
            continue
        if not np.isfinite(q_mm[j]):
            continue
        if not np.isfinite(dyn[j - SEQ + 1: j + 1]).all():
            continue
        out.append((j, d))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forcing", required=True, choices=["daymet", "nldas", "maurer"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="cap basins (smoke)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device()
    print(f"device={dev} forcing={args.forcing} seed={args.seed}", flush=True)

    attrs = json.loads((ROOT / "data" / "camels_attrs.json").read_text())
    ids = load_531()
    if args.limit:
        ids = ids[: args.limit]
    corpus = Path(args.corpus_dir)

    # ---- load all basins, build train windows + keep test series ----
    basins = {}           # bid -> dict(idx,dyn,q_mm,static_raw,area)
    train_w = []          # (dyn[SEQ,5], target, basin_code)
    stat_raw = []
    for bid in ids:
        p = corpus / f"{bid}.csv.gz"
        if not p.exists():
            p = corpus / f"{bid}.csv"
        if not p.exists():
            continue
        area = attrs.get(bid, {}).get("area_gages2")
        if not (area and np.isfinite(area) and area > 0):
            continue
        b = build_basin(p, area)
        if b is None:
            continue
        idx, dyn, q_mm = b
        s = np.array([attrs.get(bid, {}).get(k, np.nan) for k in STATIC_ATTRS], "float32")
        basins[bid] = dict(idx=idx, dyn=dyn, q_mm=q_mm, static=s, area=area)
        stat_raw.append(s)
    codes = {bid: i for i, bid in enumerate(basins)}
    print(f"loaded {len(basins)} basins", flush=True)

    # ---- static normalization ----
    S = np.stack(stat_raw)
    s_mean = np.nanmean(S, 0); s_std = np.nanstd(S, 0) + 1e-6
    for bid in basins:
        s = (basins[bid]["static"] - s_mean) / s_std
        basins[bid]["static_n"] = np.nan_to_num(s, nan=0.0).astype("float32")

    # ---- LAZY train index: (basin_code, end_j) pointers + normalization stats ----
    # Dynamic/target stats computed from the TARGET days only (streaming, no window
    # materialization). Per-basin σ for the NSE-loss weighting.
    d_sum = np.zeros(5); d_sq = np.zeros(5); d_n = 0
    train_index = []                 # (code, end_j)
    basin_std = {}
    q_sum = q_sq = q_n = 0.0
    for bid, b in basins.items():
        we = window_ends(b["idx"], b["dyn"], b["q_mm"], *TRAIN)
        c = codes[bid]
        ts = []
        for j, _ in we:
            train_index.append((c, j))
            ts.append(b["q_mm"][j])
            # dynamic stats over the target day's features (cheap proxy for full-window
            # stats; standardization only needs stable per-feature mean/std)
            v = b["dyn"][j]
            d_sum += v; d_sq += v ** 2; d_n += 1
        ts = np.array(ts, "float32")
        basin_std[bid] = (ts.std() + 0.1) if len(ts) else 1.0
        q_sum += ts.sum(); q_sq += (ts ** 2).sum(); q_n += len(ts)
    d_mean = (d_sum / d_n).astype("float32")
    d_std = (np.sqrt(np.maximum(d_sq / d_n - (d_sum / d_n) ** 2, 1e-12)) + 1e-6).astype("float32")
    q_mean = q_sum / q_n
    q_std = np.sqrt(max(q_sq / q_n - q_mean ** 2, 1e-12)) + 1e-6
    # pre-normalize each basin's dynamic array ONCE (small: 531×~12k×5 ≈ 320MB)
    for bid in basins:
        basins[bid]["dyn_n"] = ((basins[bid]["dyn"] - d_mean) / d_std).astype("float32")
    bcode = {c: bid for bid, c in codes.items()}
    N = len(train_index)
    print(f"train windows (lazy): {N}", flush=True)

    idx_arr = np.array(train_index, dtype=np.int64)      # (N,2): code, end_j
    bstd_arr = np.array([basin_std[bcode[c]] / q_std for c, _ in train_index], "float32")
    dyn_by_code = [basins[bcode[c]]["dyn_n"] for c in range(len(basins))]
    stat_by_code = np.stack([basins[bcode[c]]["static_n"] for c in range(len(basins))])
    q_by_code = [basins[bcode[c]]["q_mm"] for c in range(len(basins))]

    def slice_batch(rows):
        """Materialize ONLY this batch's windows (rows = indices into train_index)."""
        codes_b = idx_arr[rows, 0]; ends = idx_arr[rows, 1]
        xd = np.empty((len(rows), SEQ, 5), "float32")
        yt = np.empty(len(rows), "float32")
        for k, (c, j) in enumerate(zip(codes_b, ends)):
            xd[k] = dyn_by_code[c][j - SEQ + 1: j + 1]
            yt[k] = (q_by_code[c][j] - q_mean) / q_std
        xs = stat_by_code[codes_b]
        return (torch.from_numpy(xd), torch.from_numpy(xs),
                torch.from_numpy(yt), torch.from_numpy(bstd_arr[rows]))

    model = LSTMRegressor(5, len(STATIC_ATTRS), args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def lr_for(ep):
        return 1e-3 if ep < 20 else (5e-4 if ep < 25 else 1e-4)

    # ---- train (lazy batches — flat memory) ----
    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = lr_for(ep)
        model.train()
        perm = np.random.permutation(N)
        tot = 0.0
        for i in range(0, N, args.batch):
            rows = perm[i: i + args.batch]
            xd, xs, yt, w = slice_batch(rows)
            xd = xd.to(dev); xs = xs.to(dev); yt = yt.to(dev); w = w.to(dev)
            opt.zero_grad()
            pred = model(xd, xs)
            loss = (((pred - yt) ** 2) / (w ** 2 + 1e-6)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(rows)
        print(f"  epoch {ep+1}/{args.epochs} lr={lr_for(ep):.0e} loss={tot/N:.4f}",
              flush=True)

    # ---- test: predict continuous daily over the test decade, then window to grid ----
    model.eval()
    rows = []
    with torch.no_grad():
        for bid, b in basins.items():
            we = window_ends(b["idx"], b["dyn"], b["q_mm"], *TEST)
            if not we:
                continue
            dyn_n = b["dyn_n"]
            xs = torch.from_numpy(b["static_n"][None])
            preds = {}
            for k in range(0, len(we), args.batch):
                chunk = we[k: k + args.batch]
                xd = torch.from_numpy(
                    np.stack([dyn_n[j - SEQ + 1: j + 1] for j, _ in chunk])).to(dev)
                out = model(xd, xs.expand(len(chunk), -1).to(dev)).cpu().numpy()
                for (_, d), p in zip(chunk, out):
                    preds[d] = float(p) * q_std + q_mean       # de-norm mm/day
            # → stride-14 windows, cfs, matching the δHBV dump grid
            to_cfs = b["area"] / CFS_TO_MMDAY_PER_KM2
            obs_cfs = {d: b["q_mm"][i] * to_cfs
                       for i, d in enumerate(b["idx"]) if np.isfinite(b["q_mm"][i])}
            t0s = pd.date_range(TEST[0], TEST[1], freq="D")[::14]
            for t0 in t0s:
                persist = obs_cfs.get(t0 - pd.Timedelta(days=1), np.nan)
                for h in range(1, 15):
                    d = t0 + pd.Timedelta(days=h - 1)
                    if d not in preds:
                        continue
                    pc = preds[d] * to_cfs
                    tc = obs_cfs.get(d, np.nan)
                    rows.append((bid, t0.strftime("%Y-%m-%d"), h,
                                 float(tc) if np.isfinite(tc) else np.nan,
                                 pc, pc, pc, pc, float(persist)))

    out = pd.DataFrame(rows, columns=["station_id", "t0", "h", "truth",
                                      "ylo", "ymed", "yhi", "ymean", "persist"])
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, compression="gzip")

    # quick standalone median NSE
    def nse(g):
        t, p = g["truth"].to_numpy(), g["ymed"].to_numpy()
        m = np.isfinite(t) & np.isfinite(p)
        if m.sum() < 50:
            return np.nan
        t, p = t[m], p[m]; dn = ((t - t.mean()) ** 2).sum()
        return 1 - ((t - p) ** 2).sum() / dn if dn > 0 else np.nan
    med = out.groupby("station_id").apply(nse, include_groups=False).dropna().median()
    print(f"DONE {args.forcing} s{args.seed}: median NSE {med:.4f}, "
          f"{out['station_id'].nunique()} basins → {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
