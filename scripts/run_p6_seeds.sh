#!/bin/zsh
# P6: extend the shipped CMAL ensemble 4 -> 6 seeds. Warm-start CMAL heads on
# the fresh base seeds s105/s106 with the pilot-proven recipe, then evaluate
# the 6-seed Vincentized ensemble vs the shipped 4-seed under ECMWF forcing.
# Gate (pre-registered, EXPERIMENTS row 14): adopt if NSE >= +0.005 or
# CRPS <= -2% at NSE >= -0.002.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/p6_seeds.log
echo "=== P6 seed extension armed $(date)" >> $LOG
while pgrep -f "train_mblstm.py" > /dev/null; do sleep 600; done

for SEED in 105 106; do
  OUT=data/mblstm/model_h256_s${SEED}_cmalv2p.pt
  [ -f "$OUT" ] && continue
  echo "starting cmalv2p s${SEED} $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py \
    --init-ckpt data/mblstm/model_h256_s${SEED}.pt --head cmal --cmal-k 3 \
    --forcing-mix "perfect:0.4,gfs:0.6" --forcing-noise 0.15 \
    --epochs 14 --windows-per-station 300 --batch 256 --val-stride 5 \
    --lr 2e-4 --seed ${SEED} --device mps \
    --out "$OUT" > logs/mblstm_cmalv2p_s${SEED}.log 2>&1
  [ -f "$OUT" ] || echo "cmalv2p s${SEED} FAILED" >> $LOG
done

CK6="data/mblstm/model_h256_s101_cmalv2p.pt:data/mblstm/model_h256_s102_cmalv2p.pt:data/mblstm/model_h256_s103_cmalv2p.pt:data/mblstm/model_h256_s104_cmalv2p.pt:data/mblstm/model_h256_s105_cmalv2p.pt:data/mblstm/model_h256_s106_cmalv2p.pt"
if [ -f data/mblstm/model_h256_s105_cmalv2p.pt ] && [ -f data/mblstm/model_h256_s106_cmalv2p.pt ] \
   && [ ! -f benchmarks/mblstm_backtest_cmalv2p_ens6v_ecmwf_full.json ]; then
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK6" --forcing-plan "ecmwf:1-14" --label cmalv2p_ens6v_ecmwf_full \
    > logs/bt_cmalv2p_ens6v_ecmwf.log 2>&1
  $PY scripts/compare_backtests.py \
    benchmarks/mblstm_backtest_cmalv2p_ens4v_ecmwf_full.json \
    benchmarks/mblstm_backtest_cmalv2p_ens6v_ecmwf_full.json \
    > logs/p6_ens6_compare.log 2>&1
fi
echo "=== P6 seed extension complete $(date)" >> $LOG
