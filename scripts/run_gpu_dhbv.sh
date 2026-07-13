#!/bin/bash
# δHBV per-forcing ensemble on a CUDA GPU box (the beat-0.83 members).
# Trains --head dhbv on each single-forcing corpus, N_SEEDS each, then dumps.
# Idempotent (.partial + promote, skips existing). ONE driver only.
#
# S1 pilot: bash scripts/run_gpu_dhbv.sh 1 1 daymet   (1 seed, daymet only)
# S2 full:  bash scripts/run_gpu_dhbv.sh 3 3          (3 seeds, all 3 forcings)
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
# Vast pytorch images put python in a venv, not on PATH — default to it.
PY=${PY:-/venv/main/bin/python}
command -v "$PY" >/dev/null || PY=python
N_SEEDS=${1:-3}
JOBS=${2:-3}
ONLY=${3:-}                       # optional: restrict to one forcing (pilot)
DATA=${DATA:-data/gpu_corpora}
EPOCHS=${EPOCHS:-50}              # Li/Shen use 50 for δHBV
mkdir -p data/mblstm/dumps logs benchmarks
echo "=== δHBV ensemble: N_SEEDS=$N_SEEDS JOBS=$JOBS ONLY=${ONLY:-all} EPOCHS=$EPOCHS $(date) ==="

FORCINGS="daymet maurer nldas"
[ -n "$ONLY" ] && FORCINGS="$ONLY"

run_seed() {  # forcing seed
  local F=$1 SEED=$2
  local OUT=data/mblstm/camels531_${F}_dhbv_s${SEED}.pt
  [ -f "$OUT" ] && { echo "skip $OUT"; return; }
  echo "train δHBV $F seed $SEED $(date +%T)"
  "$PY" scripts/train_mblstm.py \
    --corpus-dir "$DATA/camels_corpus_${F}_v2" --enc-vars camels1f --no-q-input \
    --q-transform linear --static-set camels --head dhbv \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    --hidden 256 --epochs $EPOCHS --windows-per-station 1000 --batch 256 \
    --val-stride 10 --lr 1e-3 --seed ${SEED} --device cuda \
    --out "$OUT.partial" > "logs/tr_${F}_dhbv_s${SEED}.log" 2>&1 \
    && mv "$OUT.partial" "$OUT" && echo "done $OUT $(date +%T)" \
    || { echo "FAIL δHBV $F seed $SEED"; rm -f "$OUT.partial"; }
}

# job list, JOBS in parallel
for F in $FORCINGS; do
  for SEED in $(seq 941 $((940 + N_SEEDS))); do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 8; done
    run_seed "$F" "$SEED" &
    sleep 3
  done
done
wait
echo "=== δHBV training done $(date) ==="

# stride-1 full-531 dump per forcing (seed-averaged), fixed backtest w/ CAMELS overlay
for F in $FORCINGS; do
  CK=$(ls data/mblstm/camels531_${F}_dhbv_s*.pt 2>/dev/null | grep -v partial | paste -sd: -)
  [ -z "$CK" ] && continue
  echo "dump δHBV $F $(date +%T)"
  RW2_ENABLE_MBLSTM=1 "$PY" scripts/backtest_mblstm.py --ckpt "$CK" \
    --corpus-dir "$DATA/camels_corpus_${F}_v2" \
    --start 1989-10-01 --end 1999-09-30 --stride 1 --camels-subset 531 \
    --label camels531_${F}_dhbv_ens \
    --dump-windows data/mblstm/dumps/camels531_${F}_dhbv_ens.csv.gz \
    > "logs/bt_${F}_dhbv_ens.log" 2>&1 \
    && echo "δHBV $F dump: $(grep -oE 'CAMELS median NSE=[0-9.-]+' logs/bt_${F}_dhbv_ens.log | tail -1)" \
    || echo "δHBV $F dump FAIL"
done
echo "=== δHBV COMPLETE $(date) — pull data/mblstm/camels531_*_dhbv_s*.pt + dumps ==="
echo "δHBV ENSEMBLE DONE $(date)" > /workspace/dhbv_done.marker 2>/dev/null || true
