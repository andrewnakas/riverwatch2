#!/bin/zsh
# Phase 4.1: ensemble-forcing A/B at stride-3 on the frozen ensemble.
# Gated on the ECMWF per-member archives for 2025 being fetched (>=48 inits).
# Gate (pre-registered): pooled PICP90 in [0.86, 0.93] AND approx-CRPS -5%
# vs the matched single-forcing (ens-mean) arm.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
CKPT="data/mblstm/model_h256_s101_gfsft.pt:data/mblstm/model_h256_s102_gfsft.pt:data/mblstm/model_h256_s103_gfsft.pt:data/mblstm/model_h256_s104_gfsft.pt"
LOG=logs/p41_ensforcing.log

echo "=== P4.1 queue armed $(date)" >> $LOG
while true; do
  N=$(find data/mblstm/ecmwf_fcst/ -name '2025-*.members.csv.gz' ! -name '._*' 2>/dev/null | wc -l | tr -d ' ')
  echo "gate: ecmwf 2025 member files=$N $(date)" >> $LOG
  [ "$N" -ge 48 ] && break
  sleep 900
done

if [ ! -f benchmarks/mblstm_backtest_ens_ecmwf_members_str3.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CKPT" --members-source ecmwf --stride-stations 3 \
    --label ens_ecmwf_members_str3 > logs/bt_ens_ecmwf_members_str3.log 2>&1
  echo "members arm done $(date)" >> $LOG
fi

if [ ! -f benchmarks/mblstm_backtest_ens_ecmwf_single_str3.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CKPT" --forcing-plan "ecmwf:1-14" --stride-stations 3 \
    --point median --label ens_ecmwf_single_str3 > logs/bt_ens_ecmwf_single_str3.log 2>&1
  echo "single arm done $(date)" >> $LOG
fi

$PY scripts/compare_backtests.py \
  benchmarks/mblstm_backtest_ens_ecmwf_single_str3.json \
  benchmarks/mblstm_backtest_ens_ecmwf_members_str3.json \
  > logs/p41_compare.log 2>&1
echo "=== P4.1 complete $(date)" >> $LOG
