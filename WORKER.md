# MB-LSTM training worker — instructions for Claude

You are a training worker for RiverWatch2's v16 multi-basin LSTM streamflow
model. Your job on this machine (Windows + WSL2 + GTX 1080 8GB): fetch the
training corpus, train **seeds 103 and 104** at hidden size 256, and push the
checkpoints back to this branch (`mblstm-worker`). The main machine (a Mac)
is training seeds 101 and 102; together they form a 4-seed ensemble.

## 0. Environment (WSL2 Ubuntu)

```bash
# Verify the GPU is visible from WSL first — if this fails, the NVIDIA
# Windows driver needs updating (no driver install inside WSL!):
nvidia-smi

# Python 3.10–3.12 is fine (3.13+ also ok if torch wheels exist)
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install pandas numpy
# Sanity: GTX 1080 is compute capability 6.1 (sm_61). If the cu126 wheel
# warns it dropped sm_61, fall back to: pip install torch --index-url
# https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Fetch the corpus (~30–45 min, resumable)

The corpus is NOT in this branch (too big for git). Build it from public
APIs — Daymet (ORNL) weather + USGS discharge, both free and fast:

```bash
python scripts/build_mblstm_data_daymet.py --sleep 0.3
```

- Writes one csv.gz per station to `data/mblstm/corpus/` (~1,790 stations, ~430MB).
- **Fully resumable** — if WiFi drops or the run dies, just rerun the same
  command; it skips finished stations. Loop it until the final line reports
  ~1,780+ stations.
- A small number of failures (~100 stations with short records) is expected.

## 2. Train seeds 103 and 104 (sequential, ~1.5–2h each on the 1080)

```bash
python scripts/train_mblstm.py --compat-vars --epochs 12 \
  --windows-per-station 300 --hidden 256 --batch 256 --val-stride 20 \
  --lr 2e-4 --seed 103 --device cuda --out data/mblstm/model_h256_s103.pt \
  2>&1 | tee logs_s103.txt

python scripts/train_mblstm.py --compat-vars --epochs 12 \
  --windows-per-station 300 --hidden 256 --batch 256 --val-stride 20 \
  --lr 2e-4 --seed 104 --device cuda --out data/mblstm/model_h256_s104.pt \
  2>&1 | tee logs_s104.txt
```

Expectations (from the Mac's runs — flag anything far off):
- Each epoch prints `val_medNSE(norm-asinh)`: ~0.81 after epoch 1, climbing
  to ~0.86–0.87 by epoch 12. **If you see `nan`, stop and lower --lr to 1e-4.** (The Mac hit NaN at 4e-4 with hidden 256 — 2e-4 is already the lowered default here.)
- 8GB VRAM is plenty; if you somehow OOM, drop --batch to 128.
- If an epoch is slower than ~25 min, check `nvidia-smi` — the run should
  show a python process using the GPU.

## 3. Push the checkpoints back

```bash
git add data/mblstm/model_h256_s103.pt data/mblstm/model_h256_s104.pt logs_s10*.txt
git commit -m "Add h256 seed 103+104 checkpoints from 1080 worker"
git push origin mblstm-worker
```

Checkpoints are ~15MB each — fine for git. Do NOT commit the corpus
(`data/mblstm/corpus/` is gitignored here).

## Notes

- Data attribution: Daymet (ORNL DAAC, Thornton et al. 2022); USGS NWIS.
- The model/dataset code lives in `app/mblstm.py` and `scripts/train_mblstm.py`;
  don't change architecture or splits — checkpoints must stay compatible with
  the Mac's (same cfg → same `build_model`).
- Temporal splits are load-bearing: train ≤2024-12-31, val 2025. Don't extend
  training into 2025+ — 2025/2026 are the honest evaluation eras.
