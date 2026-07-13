#!/bin/bash
# Build a self-contained bundle to run the CAMELS recipe-v2 ensemble on a cloud
# GPU box. Produces /tmp/rw2_gpu_bundle.tar.gz (~1.1 GB) with: the training/eval
# code, the four recipe-v2 corpora, the static attrs, and a requirements list.
# On the GPU box: tar xzf rw2_gpu_bundle.tar.gz && cd rw2_gpu && \
#   pip install -r requirements_gpu.txt && bash scripts/run_gpu_ensemble.sh
set -eu
cd "$(dirname "$0")/.."
SD=/Volumes/STORAGE_SD/riverwatch2_data
STAGE=$(mktemp -d)/rw2_gpu
mkdir -p "$STAGE"/{app,scripts,data/gpu_corpora,data/mblstm/dumps,benchmarks,logs}

# Code: app package (model, metrics, hbv, deps) + the scripts we need.
cp app/*.py "$STAGE/app/" 2>/dev/null || true
cp scripts/train_mblstm.py scripts/backtest_mblstm.py scripts/combine_dumps.py \
   scripts/run_gpu_ensemble.sh scripts/run_gpu_dhbv.sh "$STAGE/scripts/"
# Static attrs + gauge lists + registry the trainer/eval read.
cp data/camels_attrs.json data/camels_gauge_ids.json \
   data/camels_station_meta.json data/gages2_attrs.json "$STAGE/data/"
# stations registry: train/backtest read data/stations_40_enriched.json.
cp data/stations_40_enriched.json "$STAGE/data/"

# Corpora (dereference symlinks) — strip exFAT AppleDouble ._* junk.
for C in camels_corpus_daymet_v2 camels_corpus_maurer_v2 \
         camels_corpus_nldas_v2 camels_corpus_3f_v2; do
  mkdir -p "$STAGE/data/gpu_corpora/$C"
  find "$SD/$C/" -name '*.csv.gz' ! -name '._*' -exec cp {} "$STAGE/data/gpu_corpora/$C/" \;
  echo "  packed $C: $(ls "$STAGE/data/gpu_corpora/$C" | wc -l | tr -d ' ') basins"
done

cat > "$STAGE/requirements_gpu.txt" <<'EOF'
torch
numpy
pandas
EOF
cat > "$STAGE/README_GPU.txt" <<'EOF'
CAMELS recipe-v2 per-forcing ensemble — GPU run.
1) pip install -r requirements_gpu.txt   (a CUDA torch build)
2) bash scripts/run_gpu_ensemble.sh [N_SEEDS=8] [JOBS=3]
   - trains daymet/maurer/nldas single-forcing (camels1f) + fused (camels3fv2),
     recipe-v2 (linear NSE loss + vapor pressure + 27 CAMELS statics), 30 epochs,
     JOBS in parallel; then stride-1 dumps + the grand-ensemble combine.
3) Pull back: data/mblstm/*.pt, data/mblstm/dumps/*.csv.gz, benchmarks/*v2r*.json
Headline lands in benchmarks/combine_camels531_grand_v2r.json (pooled + day-1).
EOF

OUT=/tmp/rw2_gpu_bundle.tar.gz
tar czf "$OUT" -C "$(dirname "$STAGE")" rw2_gpu
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1))"
rm -rf "$(dirname "$STAGE")"
