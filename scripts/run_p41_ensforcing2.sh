#!/bin/zsh
# P4.1 re-arm (v2): the members backtest thrashed against the CMAL training
# queue all morning (parked swap strangled the box). Now gated on the GPU
# queue being idle 30 min — runs in a genuine gap or after P3 drains.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
CKPT="data/mblstm/model_h256_s101_gfsft.pt:data/mblstm/model_h256_s102_gfsft.pt:data/mblstm/model_h256_s103_gfsft.pt:data/mblstm/model_h256_s104_gfsft.pt"
LOG=logs/p41_ensforcing.log
echo "=== P4.1 v2 armed (waits for 30min trainer-free) $(date)" >> $LOG
while true; do
  if ! pgrep -f "train_mblstm.py" > /dev/null; then
    sleep 1800
    pgrep -f "train_mblstm.py" > /dev/null || break
  else
    sleep 600
  fi
done
echo "=== P4.1 v2 window open $(date)" >> $LOG
if [ ! -f benchmarks/mblstm_backtest_ens_ecmwf_members_str3.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CKPT" --members-source ecmwf --stride-stations 3 \
    --label ens_ecmwf_members_str3 > logs/bt_ens_ecmwf_members_str3.log 2>&1
fi
if [ ! -f benchmarks/mblstm_backtest_ens_ecmwf_single_str3.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CKPT" --forcing-plan "ecmwf:1-14" --stride-stations 3 \
    --point median --label ens_ecmwf_single_str3 > logs/bt_ens_ecmwf_single_str3.log 2>&1
fi
$PY scripts/compare_backtests.py \
  benchmarks/mblstm_backtest_ens_ecmwf_single_str3.json \
  benchmarks/mblstm_backtest_ens_ecmwf_members_str3.json > logs/p41_compare.log 2>&1
echo "=== P4.1 v2 complete $(date)" >> $LOG
