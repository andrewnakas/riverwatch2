#!/bin/zsh
# B1 round 2: SIM-tuned strict CAMELS retrain. Fixes vs round 1 (0.587 day-1):
#   - windows-per-station 300 -> 1000 and epochs 12 -> 16 (round 1 saw ~27x
#     less data than the published protocol)
#   - --point-loss mse (basin-normalized squared error = the leaderboard loss)
# 2 screening seeds; extend to 8 if the ens2 day-1 eval clears 0.65.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/b1_round2.log
echo "=== B1 round 2 armed $(date)" >> $LOG
while pgrep -f "train_mblstm.py" > /dev/null; do sleep 600; done

for SEED in 921 922; do
  OUT=data/mblstm/camels531_noq_mse_s${SEED}.pt
  [ -f "$OUT" ] && continue
  echo "training mse seed ${SEED} $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py \
    --corpus-dir data/mblstm/camels_corpus --compat-vars --no-q-input \
    --point-loss mse \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    --hidden 256 --epochs 16 --windows-per-station 1000 --batch 256 \
    --val-stride 20 --lr 2e-4 --seed ${SEED} --device mps \
    --out "$OUT" > logs/camels531_mse_s${SEED}.log 2>&1
  [ -f "$OUT" ] || echo "mse seed ${SEED} FAILED" >> $LOG
done

CK=$(ls data/mblstm/camels531_noq_mse_s92*.pt 2>/dev/null | paste -sd: -)
if [ -n "$CK" ] && [ ! -f benchmarks/mblstm_backtest_camels531_mse_ens2.json ]; then
  # Wait for any running eval (with-q ens4) to finish first — one CPU job.
  while pgrep -f "backtest_mblstm.py" > /dev/null; do sleep 600; done
  echo "eval mse ens2 (test decade) $(date)" >> $LOG
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK" --corpus-dir data/mblstm/camels_corpus \
    --start 1989-10-01 --end 1999-09-30 --stride 14 \
    --camels-subset 531 --label camels531_mse_ens2 \
    --dump-windows data/mblstm/dumps/camels531_mse_ens2.csv.gz \
    > logs/bt_camels531_mse_ens2.log 2>&1
fi
echo "=== B1 round 2 complete $(date)" >> $LOG
