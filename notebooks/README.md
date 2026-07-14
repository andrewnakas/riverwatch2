# Kaggle no-q campaign — how to run

Goal: push the no-observed-discharge CAMELS-531 median NSE from ~0.80 toward the
Li/Shen 2025 record of **0.83**, and produce the **exact-Kratzert day-1 number**
directly comparable to it.

The key lever (verified this session): the shipped record-recipe δHBV members were
trained with `cfg.dhbv_loss=None` (plain MSE) — the **combined decorrelation loss
was never actually trained in**. These notebooks retrain δHBV with the two fidelity
fixes: `--dhbv-loss combined` + `--epochs 100`.

## One-time setup

1. **Kaggle credentials** (never committed — `kaggle.json` is gitignored):
   ```
   pip install kaggle
   mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   ⚠️ **Rotate the API key** afterward if it was ever pasted in chat
   (kaggle.com → Account → *Expire API Token*).

2. **Package the data as Kaggle Datasets** (needs the SD-mounted corpora):
   ```
   bash notebooks/package_kaggle_data.sh <your-kaggle-username>
   ```
   Creates `rw2-camels-corpus-{daymet,maurer,nldas}` (~265MB each) and
   `rw2-camels-static`.

## Run order (on kaggle.com/code, GPU accelerator ON)

| # | Notebook | Inputs to add | What it does |
|---|----------|---------------|--------------|
| 1 | `kaggle_train_dhbv.ipynb` | code (git clone, automatic) + `rw2-camels-corpus-<F>` + `rw2-camels-static` | Trains combined-loss δHBV for one forcing (set `FORCING`). ~3 seeds/12-hr session. **Save Version** → makes `rw2-noq-ckpts`. Repeat per forcing. |
| 2 | `kaggle_dump.ipynb` | + `rw2-noq-ckpts` | Seed-ensembles the new members, writes stride-14/ss3 dump csv.gz (asserts CAMELS static overlay). Download the csv.gz to `data/mblstm/gpu_dumps_s14/` locally. |
| 3 | `kaggle_eval_day1.ipynb` | + `rw2-noq-ckpts` | Exact-Kratzert day-1 NSE: LSTM validation (stride-1) then δHBV headline (stride-3). |

## Combine locally (no GPU)

After downloading the new dumps, swap them into the grand ensemble:
```
python scripts/combine_dumps.py \
  --dumps <new combined100 dumps> <existing v2r + fused dumps> \
  --fit-weights --val-end 1998-09-30 --label combined100_grand
# optional per-basin static gate (ties baseline on current members; retry once
# the combined-loss members decorrelate):
python scripts/combine_dumps.py --dumps ... --static-gate 3 --val-end 1998-09-30
```
Baseline to beat: **pooled 0.8025 / day-1 0.8283** (7-member fit-weights).
Pre-registered target: **~0.815–0.82 pooled** (a clean 0.83 without observed-q is
at the field ceiling; day-1 already matches 0.83).

## Kill/keep discipline

A lever counts only if it raises the **combined** number vs the baseline (per-member
gains compress at the ensemble — measured: fcorr +0.015/member → +0.004 grand).
Record each run in `benchmarks/EXPERIMENTS.md`.
