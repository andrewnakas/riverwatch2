#!/bin/zsh
# Phase 3 training queue (sequential on MPS, resumable via ckpt existence):
#   1. forcing-mixture fine-tune of seeds 101-104 (perfect/gfs/gefs mix)
#   2. CMAL v2: warm-start heads from the mixture seeds, same mix, 14 epochs
#   3. asinh-free (linear q-transform) 1-seed ablation from scratch
# Gated on: GEFS train-period archive fetched (>=180 inits 2021-2024) and no
# other trainer running (s105 done/crashed).
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/p3_queue.log
MIX="perfect:0.25,gfs:0.4,gefs:0.35"

gefs_train_count() {
  find data/mblstm/gefs_fcst/ -name '202[1-4]-*.csv.gz' ! -name '*members*' 2>/dev/null | wc -l | tr -d ' '
}

echo "=== P3 queue armed $(date)" >> $LOG
while true; do
  N=$(gefs_train_count)
  BUSY=0
  pgrep -f "train_mblstm.py" > /dev/null && BUSY=1
  echo "gate: gefs_train=$N busy=$BUSY $(date)" >> $LOG
  if [ "$N" -ge 180 ] && [ "$BUSY" -eq 0 ]; then break; fi
  sleep 900
done

echo "=== P3 stage 1: mixture fine-tunes $(date)" >> $LOG
for SEED in 101 102 103 104; do
  OUT=data/mblstm/model_h256_s${SEED}_mixft.pt
  [ -f "$OUT" ] && continue
  caffeinate -i $PY scripts/train_mblstm.py \
    --init-ckpt data/mblstm/model_h256_s${SEED}.pt \
    --forcing-mix "$MIX" --forcing-noise 0.15 \
    --epochs 6 --windows-per-station 300 --batch 256 --val-stride 5 \
    --lr 1e-4 --seed ${SEED} --device mps \
    --out "$OUT" > logs/mblstm_mixft_s${SEED}.log 2>&1
  [ -f "$OUT" ] || { echo "mixft s${SEED} FAILED" >> $LOG; }
done

echo "=== P3 stage 2: CMAL v2 warm-starts $(date)" >> $LOG
for SEED in 101 102 103 104; do
  OUT=data/mblstm/model_h256_s${SEED}_cmalv2.pt
  BASE=data/mblstm/model_h256_s${SEED}_mixft.pt
  [ -f "$OUT" ] && continue
  [ -f "$BASE" ] || { echo "cmalv2 s${SEED}: no mixft base, skip" >> $LOG; continue; }
  caffeinate -i $PY scripts/train_mblstm.py \
    --init-ckpt "$BASE" --head cmal --cmal-k 3 \
    --forcing-mix "$MIX" --forcing-noise 0.15 \
    --epochs 14 --windows-per-station 300 --batch 256 --val-stride 5 \
    --lr 2e-4 --seed ${SEED} --device mps \
    --out "$OUT" > logs/mblstm_cmalv2_s${SEED}.log 2>&1
  [ -f "$OUT" ] || { echo "cmalv2 s${SEED} FAILED" >> $LOG; }
done

echo "=== P3 stage 3: linear q-transform ablation $(date)" >> $LOG
OUT=data/mblstm/model_h256_linear_s301.pt
if [ ! -f "$OUT" ]; then
  caffeinate -i $PY scripts/train_mblstm.py --compat-vars --q-transform linear \
    --epochs 12 --windows-per-station 300 --hidden 256 --batch 256 \
    --val-stride 20 --lr 2e-4 --seed 301 --device mps \
    --out "$OUT" > logs/mblstm_linear_s301.log 2>&1
fi

echo "=== P3 queue complete $(date)" >> $LOG
