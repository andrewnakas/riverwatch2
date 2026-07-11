#!/bin/bash
# Full CAMELS recipe-v2 per-forcing ensemble, for a CUDA GPU box (RTX 4090-class).
# Unlike the local 8 GB Mac supervisors, a GPU box has no swap constraint, so
# this trains seeds in PARALLEL (JOBS at a time) with --device cuda, no swap
# guard, no one-job serialization. Idempotent: skips any checkpoint already
# present (.partial + promote), so a re-run resumes.
#
# Primary lever (Li/Shen 2025): a per-forcing ENSEMBLE of single-forcing models
# beats one fused model. Trains N_SEEDS each on Daymet / Maurer / NLDAS
# single-forcing (camels1f, recipe-v2 = linear NSE loss + vapor pressure + 27
# CAMELS statics), plus the fused camels3fv2 as an extra member, then dumps
# stride-1 windows per forcing for the offline grand-ensemble combiner.
#
# Usage on the GPU box (from the repo root, after unpacking the bundle):
#   bash scripts/run_gpu_ensemble.sh [N_SEEDS] [JOBS]
# defaults: N_SEEDS=8, JOBS=3 (fits a 24GB 4090; raise if more VRAM).
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1   # stream training progress to per-seed logs in real time
PY=${PY:-python}
N_SEEDS=${1:-8}
JOBS=${2:-3}
DATA=${DATA:-data/gpu_corpora}     # bundle unpacks corpora here (see pack script)
EPOCHS=${EPOCHS:-30}               # Kratzert used 30; GPU makes this cheap
mkdir -p data/mblstm/dumps benchmarks logs
echo "=== GPU ensemble: N_SEEDS=$N_SEEDS JOBS=$JOBS EPOCHS=$EPOCHS $(date) ==="

train_one() {  # forcing, encvars, seed, corpus_subdir
  local F=$1 EV=$2 SEED=$3 CORPUS=$4
  local OUT=data/mblstm/camels531_${F}_v2r_s${SEED}.pt
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; return; }
  echo "train $F seed $SEED $(date +%T)"
  $PY scripts/train_mblstm.py \
    --corpus-dir "$DATA/$CORPUS" --enc-vars "$EV" --no-q-input \
    --point-loss mse --q-transform linear --static-set camels \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    --hidden 256 --epochs $EPOCHS --windows-per-station 1000 --batch 256 \
    --val-stride 10 --lr 1e-3 --seed ${SEED} --device cuda \
    --out "$OUT.partial" > logs/tr_${F}_s${SEED}.log 2>&1 \
    && mv "$OUT.partial" "$OUT" && echo "done $OUT $(date +%T)" \
    || { echo "FAIL $F seed $SEED"; rm -f "$OUT.partial"; }
}
export -f train_one
export PY DATA EPOCHS

# Build the job list: 3 single-forcing sets + fused, N_SEEDS each.
JOBLIST=$(mktemp)
for SEED in $(seq 941 $((940 + N_SEEDS))); do
  echo "daymet camels1f $SEED camels_corpus_daymet_v2" >> "$JOBLIST"
  echo "maurer camels1f $SEED camels_corpus_maurer_v2" >> "$JOBLIST"
  echo "nldas  camels1f $SEED camels_corpus_nldas_v2"  >> "$JOBLIST"
  echo "fused  camels3fv2 $SEED camels_corpus_3f_v2"   >> "$JOBLIST"
done
# Run JOBS at a time.
xargs -P "$JOBS" -L1 bash -c 'train_one "$@"' _ < "$JOBLIST"
rm -f "$JOBLIST"
echo "=== all training done $(date) ==="

# Per-forcing stride-1 dumps for the grand-ensemble combiner.
for F in daymet maurer nldas fused; do
  case $F in
    daymet) C=camels_corpus_daymet_v2 ;; maurer) C=camels_corpus_maurer_v2 ;;
    nldas)  C=camels_corpus_nldas_v2  ;; fused)  C=camels_corpus_3f_v2 ;;
  esac
  CK=$(ls data/mblstm/camels531_${F}_v2r_s*.pt 2>/dev/null | grep -v partial | paste -sd: -)
  [ -z "$CK" ] && continue
  echo "dump+eval $F ens $(date +%T)"
  RW2_ENABLE_MBLSTM=1 $PY scripts/backtest_mblstm.py \
    --ckpt "$CK" --corpus-dir "$DATA/$C" \
    --start 1989-10-01 --end 1999-09-30 --stride 1 \
    --camels-subset 531 --label camels531_${F}_v2r_ens \
    --dump-windows data/mblstm/dumps/camels531_${F}_v2r_ens.csv.gz \
    > logs/bt_${F}_v2r_ens.log 2>&1 && echo "eval $F done" || echo "eval $F FAIL"
done

# Grand ensemble across all four forcings (full-531 stride-1 headline).
DUMPS=$(ls data/mblstm/dumps/camels531_*_v2r_ens.csv.gz 2>/dev/null)
if [ -n "$DUMPS" ]; then
  $PY scripts/combine_dumps.py --dumps $DUMPS \
    --label camels531_grand_v2r --out benchmarks/combine_camels531_grand_v2r.json \
    > logs/grand_v2r.log 2>&1 && cat logs/grand_v2r.log
fi
echo "=== GPU ensemble COMPLETE $(date) — pull data/mblstm/*.pt, data/mblstm/dumps/, benchmarks/*v2r*.json ==="
