#!/bin/zsh
# Phase 5.2: static-attrs 14->22 A/B — 1 scratch seed with --static-set full.
# Gated on the P3 queue finishing (GPU free). Compare vs s105/s106 base seeds
# at stride-3 perfect forcing first (screen), then real forcing if it passes.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/p5_static.log
echo "=== P5 static A/B armed $(date)" >> $LOG
until grep -q "P3 queue complete" logs/p3_queue.log 2>/dev/null; do sleep 900; done
OUT=data/mblstm/model_h256_static22_s401.pt
if [ ! -f "$OUT" ]; then
  echo "starting static22 s401 $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py --compat-vars --static-set full \
    --epochs 12 --windows-per-station 300 --hidden 256 --batch 256 \
    --val-stride 20 --lr 2e-4 --seed 401 --device mps \
    --out "$OUT" > logs/mblstm_static22_s401.log 2>&1
fi
echo "=== P5 static A/B complete $(date)" >> $LOG
