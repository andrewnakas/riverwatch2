#!/bin/zsh
# Night-1 MPS chain (GEFS still fetching, so no full mixture ft yet):
#   1. CMAL v2 PILOT: seed 101 warm-started from the GFS-ft ckpt with a
#      perfect/gfs forcing mix — early de-risk read on the FHV/CRPS gates.
#   2. s106 base seed (toward the 8-seed ensemble) if the GPU is still free.
# The P3 queue takes over once the GEFS train archive lands (its gate also
# waits for these trainers to exit).
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/night1_gpu.log
echo "=== night1 gpu start $(date)" >> $LOG

OUT=data/mblstm/model_h256_s101_cmalv2p.pt
if [ ! -f "$OUT" ]; then
  caffeinate -i $PY scripts/train_mblstm.py \
    --init-ckpt data/mblstm/model_h256_s101_gfsft.pt --head cmal --cmal-k 3 \
    --forcing-mix "perfect:0.4,gfs:0.6" --forcing-noise 0.15 \
    --epochs 14 --windows-per-station 300 --batch 256 --val-stride 5 \
    --lr 2e-4 --seed 101 --device mps \
    --out "$OUT" > logs/mblstm_cmalv2p_s101.log 2>&1
  echo "cmal v2 pilot done $(date)" >> $LOG
fi

OUT=data/mblstm/model_h256_s106.pt
if [ ! -f "$OUT" ]; then
  caffeinate -i $PY scripts/train_mblstm.py --compat-vars --epochs 12 \
    --windows-per-station 300 --hidden 256 --batch 256 --val-stride 20 \
    --lr 2e-4 --seed 106 --device mps \
    --out "$OUT" > logs/mblstm_train_h256_s106.log 2>&1
  echo "s106 done $(date)" >> $LOG
fi
echo "=== night1 gpu complete $(date)" >> $LOG
