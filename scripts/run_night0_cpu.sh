#!/bin/zsh
# Night-0 CPU chain: stride-3 hybrid baseline -> perfect-forcing re-baseline
# -> stride-3 dump run -> offline anchor/point sweep. Resumable at every step.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
CKPT="data/mblstm/model_h256_s101_gfsft.pt:data/mblstm/model_h256_s102_gfsft.pt:data/mblstm/model_h256_s103_gfsft.pt:data/mblstm/model_h256_s104_gfsft.pt"
LOG=logs/night0_cpu.log
echo "=== night0 cpu chain start $(date)" >> $LOG

if [ ! -f benchmarks/mblstm_backtest_ens4ft_hybrid_str3.json ]; then
  caffeinate -i $PY scripts/backtest_mblstm.py --ckpt "$CKPT" --gfs --hrrr \
    --stride-stations 3 --label ens4ft_hybrid_str3 \
    > logs/bt_ens4ft_hybrid_str3.log 2>&1
  echo "hybrid_str3 done $(date)" >> $LOG
fi

if [ ! -f benchmarks/mblstm_backtest_ens4ft_perfect.json ]; then
  caffeinate -i $PY scripts/backtest_mblstm.py --ckpt "$CKPT" \
    --label ens4ft_perfect > logs/bt_ens4ft_perfect.log 2>&1
  echo "perfect re-baseline done $(date)" >> $LOG
fi

if [ ! -f data/mblstm/dumps/ens4ft_hybrid_str3.csv.gz ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CKPT" --gfs --hrrr --stride-stations 3 \
    --label ens4ft_hybrid_str3_dump \
    --dump-windows data/mblstm/dumps/ens4ft_hybrid_str3.csv.gz \
    --point median > logs/bt_hybrid_str3_dump.log 2>&1
  echo "dump run done $(date)" >> $LOG
fi

$PY scripts/sweep_anchor_point.py \
  --dump data/mblstm/dumps/ens4ft_hybrid_str3.csv.gz \
  --label hybrid_str3 > logs/sweep_anchor_point_str3.log 2>&1
echo "=== night0 cpu chain complete $(date)" >> $LOG
