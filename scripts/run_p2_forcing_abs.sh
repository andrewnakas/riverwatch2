#!/bin/zsh
# Phase 2: eval-time forcing A/Bs at stride-3 on the frozen ens4ft checkpoints.
# Gated: waits for the P1 dump run to finish (CPU free) and for the ECMWF +
# GEFS-2025 archives to be fetched. Each run skips itself if its JSON exists.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
CKPT="data/mblstm/model_h256_s101_gfsft.pt:data/mblstm/model_h256_s102_gfsft.pt:data/mblstm/model_h256_s103_gfsft.pt:data/mblstm/model_h256_s104_gfsft.pt"
LOG=logs/p2_forcing_abs.log

# Trailing slash: these dirs are SD-card symlinks and find won't follow a bare
# symlink path.
count() { find "$1/" -name '2025-*.csv.gz' ! -name '*members*' 2>/dev/null | wc -l | tr -d ' '; }

echo "=== P2 queue armed $(date)" >> $LOG
while true; do
  EC=$(count data/mblstm/ecmwf_fcst)
  GE=$(count data/mblstm/gefs_fcst)
  DUMP_DONE=0
  [ -f benchmarks/mblstm_backtest_ens4ft_hybrid_str3_dump.json ] && DUMP_DONE=1
  echo "gate: ecmwf2025=$EC gefs2025=$GE dump=$DUMP_DONE $(date)" >> $LOG
  if [ "$EC" -ge 48 ] && [ "$GE" -ge 48 ] && [ "$DUMP_DONE" -eq 1 ]; then break; fi
  sleep 900
done

echo "=== P2 A/Bs starting $(date)" >> $LOG
run_ab() {  # label plan
  [ -f "benchmarks/mblstm_backtest_$1.json" ] && return
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CKPT" --forcing-plan "$2" --stride-stations 3 \
    --point median --label "$1" > "logs/bt_$1.log" 2>&1
  echo "done $1 $(date)" >> $LOG
}

run_ab ens4ft_ecmwf_str3        "ecmwf:1-14"
run_ab ens4ft_ecmwf_hrrr_str3   "ecmwf:1-14,hrrr?:1-2"
run_ab ens4ft_gefs_str3         "gefs:1-14"
run_ab ens4ft_gefs_hrrr_str3    "gefs:1-14,hrrr?:1-2"

echo "=== P2 A/B queue complete $(date)" >> $LOG
