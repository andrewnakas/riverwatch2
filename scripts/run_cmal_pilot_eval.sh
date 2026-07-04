#!/bin/zsh
# CMAL v2 pilot evaluation (exp 6 pre-read): when the pilot ckpt lands, score
# it against the single-seed quantile GFS-ft reference on the same stride-3
# subset under GFS forcing. Gate (pre-registered, EXPERIMENTS.md row 6):
# CRPS -5% AND FHV +8 pts, NSE within -0.01 of the quantile reference.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/cmal_pilot_eval.log
echo "=== cmal pilot eval armed $(date)" >> $LOG
until [ -f data/mblstm/model_h256_s101_cmalv2p.pt ]; do sleep 600; done

if [ ! -f benchmarks/mblstm_backtest_s101ft_gfs_str3.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt data/mblstm/model_h256_s101_gfsft.pt --gfs --stride-stations 3 \
    --point median --label s101ft_gfs_str3 > logs/bt_s101ft_gfs_str3.log 2>&1
fi

if [ ! -f benchmarks/mblstm_backtest_cmalv2p_gfs_str3.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt data/mblstm/model_h256_s101_cmalv2p.pt --gfs --stride-stations 3 \
    --label cmalv2p_gfs_str3 > logs/bt_cmalv2p_gfs_str3.log 2>&1
fi

$PY scripts/compare_backtests.py \
  benchmarks/mblstm_backtest_s101ft_gfs_str3.json \
  benchmarks/mblstm_backtest_cmalv2p_gfs_str3.json > logs/cmal_pilot_compare.log 2>&1
echo "=== cmal pilot eval complete $(date)" >> $LOG
