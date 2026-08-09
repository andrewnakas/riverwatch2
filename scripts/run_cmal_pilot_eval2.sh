#!/bin/zsh
# Corrected CMAL pilot eval: wait for the TRAINER PROCESS to exit (the ckpt
# file appears after epoch 1 — gating on it scores an undertrained model),
# then backtest the final pilot ckpt and compare vs the quantile reference.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/cmal_pilot_eval.log
echo "=== corrected eval armed $(date)" >> $LOG
while pgrep -f "cmalv2p" | grep -qv $$; do sleep 300; done
[ -f data/mblstm/model_h256_s101_cmalv2p.pt ] || { echo "no pilot ckpt — trainer failed $(date)" >> $LOG; exit 1; }
RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
  --ckpt data/mblstm/model_h256_s101_cmalv2p.pt --gfs --stride-stations 3 \
  --label cmalv2p_gfs_str3 > logs/bt_cmalv2p_gfs_str3.log 2>&1
$PY scripts/compare_backtests.py \
  benchmarks/mblstm_backtest_s101ft_gfs_str3.json \
  benchmarks/mblstm_backtest_cmalv2p_gfs_str3.json > logs/cmal_pilot_compare.log 2>&1
echo "=== corrected eval complete $(date)" >> $LOG
