#!/bin/zsh
# Night-0 MPS chain: wait for the stride-3 hybrid baseline to free RAM, then
# train base seed s105 toward the 8-seed ensemble.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/night0_gpu.log
echo "=== night0 gpu chain armed $(date)" >> $LOG
until [ -f benchmarks/mblstm_backtest_ens4ft_hybrid_str3.json ]; do sleep 300; done
if [ ! -f data/mblstm/model_h256_s105.pt ]; then
  echo "starting s105 $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py --compat-vars --epochs 12 \
    --windows-per-station 300 --hidden 256 --batch 256 --val-stride 20 \
    --lr 2e-4 --seed 105 --device mps \
    --out data/mblstm/model_h256_s105.pt > logs/mblstm_train_h256_s105.log 2>&1
fi
echo "=== night0 gpu chain complete $(date)" >> $LOG
