#!/bin/zsh
# Phase 0 baseline queue (sequential to respect 8GB RAM):
#   1. stride-3 GFS reference        (matched subset for all future A/B screens)
#   2. stride-3 GFS+HRRR reference
#   3. perfect-forcing full re-baseline (sizes the current forcing gap)
# Each step skips itself if its benchmark JSON already exists (resumable).
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
CKPT="data/mblstm/model_h256_s101_gfsft.pt:data/mblstm/model_h256_s102_gfsft.pt:data/mblstm/model_h256_s103_gfsft.pt:data/mblstm/model_h256_s104_gfsft.pt"

if [ ! -f benchmarks/mblstm_backtest_ens4ft_gfs_str3.json ]; then
  caffeinate -i $PY scripts/backtest_mblstm.py --ckpt "$CKPT" --gfs \
    --stride-stations 3 --label ens4ft_gfs_str3 \
    > logs/bt_ens4ft_gfs_str3.log 2>&1
fi

if [ ! -f benchmarks/mblstm_backtest_ens4ft_hybrid_str3.json ]; then
  caffeinate -i $PY scripts/backtest_mblstm.py --ckpt "$CKPT" --gfs --hrrr \
    --stride-stations 3 --label ens4ft_hybrid_str3 \
    > logs/bt_ens4ft_hybrid_str3.log 2>&1
fi

if [ ! -f benchmarks/mblstm_backtest_ens4ft_perfect.json ]; then
  caffeinate -i $PY scripts/backtest_mblstm.py --ckpt "$CKPT" \
    --label ens4ft_perfect \
    > logs/bt_ens4ft_perfect.log 2>&1
fi

echo "p0 baseline queue complete $(date)"
