#!/bin/zsh
# P2 full-scale confirm: ECMWF forcing passed stride-3 (+0.040 vs gate +0.015).
# Gate: full-set NSE >= gfs_full + 0.01 AND no tercile regression -> promote
# ECMWF to the serving forcing plan (with the ens-mean caveat: the archive is
# IFS ENS mean; serve-side reproduction needs the ensemble API, not HRES).
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
CKPT="data/mblstm/model_h256_s101_gfsft.pt:data/mblstm/model_h256_s102_gfsft.pt:data/mblstm/model_h256_s103_gfsft.pt:data/mblstm/model_h256_s104_gfsft.pt"
LOG=logs/p2_confirm.log
echo "=== P2 ECMWF full confirm start $(date)" >> $LOG
if [ ! -f benchmarks/mblstm_backtest_ens4ft_ecmwf_full.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CKPT" --forcing-plan "ecmwf:1-14" --point median \
    --label ens4ft_ecmwf_full > logs/bt_ens4ft_ecmwf_full.log 2>&1
fi
$PY scripts/compare_backtests.py \
  benchmarks/mblstm_backtest_h256ens4ft_gfs.json \
  benchmarks/mblstm_backtest_ens4ft_ecmwf_full.json > logs/p2_confirm_compare.log 2>&1
echo "=== P2 ECMWF full confirm complete $(date)" >> $LOG
