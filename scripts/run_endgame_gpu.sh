#!/bin/zsh
# Corrected post-P3 GPU sequence (mixft failed its gate -> CMAL-from-mixft
# s103/s104 cancelled; lineage A/B keeps s101-102):
#   1. linear q-transform ablation (exp 7)
#   2. static-22 seed (P5.2)
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/endgame_gpu.log
echo "=== endgame gpu start $(date)" >> $LOG
while pgrep -f "train_mblstm.py" > /dev/null; do sleep 300; done
if [ ! -f data/mblstm/model_h256_linear_s301.pt ]; then
  caffeinate -i $PY scripts/train_mblstm.py --compat-vars --q-transform linear \
    --epochs 12 --windows-per-station 300 --hidden 256 --batch 256 \
    --val-stride 20 --lr 2e-4 --seed 301 --device mps \
    --out data/mblstm/model_h256_linear_s301.pt > logs/mblstm_linear_s301.log 2>&1
  echo "linear s301 done $(date)" >> $LOG
fi
if [ ! -f data/mblstm/model_h256_static22_s401.pt ]; then
  caffeinate -i $PY scripts/train_mblstm.py --compat-vars --static-set full \
    --epochs 12 --windows-per-station 300 --hidden 256 --batch 256 \
    --val-stride 20 --lr 2e-4 --seed 401 --device mps \
    --out data/mblstm/model_h256_static22_s401.pt > logs/mblstm_static22_s401.log 2>&1
  echo "static22 s401 done $(date)" >> $LOG
fi
echo "=== endgame gpu complete $(date)" >> $LOG
