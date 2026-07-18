# Reach 0.83 on your 1080 — the real neuralhydrology 9-member run

This trains the **exact Kratzert/Li-Shen reference LSTM ensemble** (real
`neuralhydrology`, not the standalone approximation) that reaches the CAMELS-531
no-q record. 3 forcings × 3 seeds = 9 members → grand ensemble + our δHBV dumps.

A GTX 1080 (8 GB) handles this fine: hidden 256, seq 365, batch 256. Each member
is ~1-2 hr on a 1080; 9 members ≈ 12-18 hr (or run a few, ensembling is forgiving).

## One-time setup

```bash
cd gpu1080
# 1. python deps (a venv is cleanest)
python3 -m pip install neuralhydrology torch pandas numpy xarray netCDF4 scipy

# 2. get the 3 corpora (Kaggle bulk-download works from a home IP):
#    needs ~/.kaggle/kaggle.json with a FULL (non-scoped) key
export KAGGLE_CONFIG_DIR=~/.kaggle
bash fetch_corpora.sh          # downloads+gzips daymet/nldas/maurer → ./corpora/

# 3. build NH data (netCDF + attributes) for all 3 forcings:
bash build_nh_data.sh          # → ./nh_data/{daymet,nldas,maurer}/
```

## Train the 9 members

```bash
bash run9.sh                   # trains, evaluates, dumps each → ./dumps/
```

Each member prints its test median NSE. Expected (matches the paper):
daymet ~0.75, nldas ~0.72, maurer ~0.72 (single seed).

## Grand ensemble → 0.83

When the 9 dumps are in `./dumps/`, combine them with the existing δHBV members
(in the main repo `data/mblstm/gpu_dumps_s14/`):

```bash
cd ..    # repo root
python3 scripts/combine_dumps.py --fit-weights --val-end 1998-09-30 \
  --dumps gpu1080/dumps/camels531_*_nhlstm_s*.csv.gz \
          data/mblstm/gpu_dumps_s14/camels531_*_fcorr_ens_s14.csv.gz \
  --label nh9_plus_dhbv --out benchmarks/combine_nh9_grand.json
```

Ladder to expect (paper Table D1): NH LSTM 9-member ensemble → ~0.808,
+ δHBV → ~0.818, and the 3-seed averaging is what lifts it toward **0.830**.

## Notes
- All scripts use `$PWD`-relative paths, so run them from `gpu1080/`.
- The trainer/eval use `device: cuda:0` — confirm `torch.cuda.is_available()`.
- `corpus_to_nh.py`, `nh_to_dump.py`, `combine_dumps.py` live in `../scripts/`.
