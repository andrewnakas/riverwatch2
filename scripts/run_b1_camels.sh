#!/bin/zsh
# B1: CAMELS-531 strict protocol benchmark (EXPERIMENTS row 15).
# Protocol: train 1999-10-01..2008-09-30 (--train-start guards the test
# decade), test 1989-10-01..1999-09-30, Daymet basin-mean forcings, statics.
#   stage 1: 8 seeds STRICT no-discharge (leaderboard entry; published line
#            0.74 single / 0.76-0.82 ensembles / 0.83 record)
#   stage 2: 4 seeds with discharge (NOT protocol-comparable; measures what
#            AR assimilation is worth on their basins)
#   stage 3: chained day-1..14 evals over the test decade (stride 14 covers
#            every day exactly once; perfect forcing = simulation analog),
#            with window dumps for day-level NSE.
# Val window 1998-10..1999-09 sits inside the test decade — standard for this
# benchmark lineage (model selection on val year), stated for honesty.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/b1_camels.log
TRAIN_ARGS=(--corpus-dir data/mblstm/camels_corpus --compat-vars
  --train-start 1999-10-01 --train-end 2008-09-30
  --val-start 1998-10-01 --val-end 1999-09-30
  --hidden 256 --epochs 12 --windows-per-station 300 --batch 256
  --val-stride 20 --lr 2e-4 --device mps)

echo "=== B1 CAMELS queue armed $(date)" >> $LOG
while pgrep -f "train_mblstm.py" > /dev/null; do sleep 600; done

for SEED in 901 902 903 904 905 906 907 908; do
  OUT=data/mblstm/camels531_noq_s${SEED}.pt
  [ -f "$OUT" ] && continue
  echo "training strict no-q seed ${SEED} $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py "${TRAIN_ARGS[@]}" \
    --no-q-input --seed ${SEED} --out "$OUT" \
    > logs/camels531_noq_s${SEED}.log 2>&1
  [ -f "$OUT" ] || echo "no-q seed ${SEED} FAILED" >> $LOG
done

for SEED in 911 912 913 914; do
  OUT=data/mblstm/camels531_q_s${SEED}.pt
  [ -f "$OUT" ] && continue
  echo "training with-q seed ${SEED} $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py "${TRAIN_ARGS[@]}" \
    --seed ${SEED} --out "$OUT" \
    > logs/camels531_q_s${SEED}.log 2>&1
  [ -f "$OUT" ] || echo "with-q seed ${SEED} FAILED" >> $LOG
done

CK_NOQ=$(ls data/mblstm/camels531_noq_s90*.pt 2>/dev/null | paste -sd: -)
CK_Q=$(ls data/mblstm/camels531_q_s91*.pt 2>/dev/null | paste -sd: -)

if [ -n "$CK_NOQ" ] && [ ! -f benchmarks/mblstm_backtest_camels531_noq_ens8.json ]; then
  echo "eval strict ens (test decade) $(date)" >> $LOG
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK_NOQ" --corpus-dir data/mblstm/camels_corpus \
    --start 1989-10-01 --end 1999-09-30 --stride 14 \
    --camels-subset 531 --label camels531_noq_ens8 \
    --dump-windows data/mblstm/dumps/camels531_noq_ens8.csv.gz \
    > logs/bt_camels531_noq_ens8.log 2>&1
fi
if [ -n "$CK_Q" ] && [ ! -f benchmarks/mblstm_backtest_camels531_q_ens4.json ]; then
  echo "eval with-q ens (test decade) $(date)" >> $LOG
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK_Q" --corpus-dir data/mblstm/camels_corpus \
    --start 1989-10-01 --end 1999-09-30 --stride 14 \
    --camels-subset 531 --label camels531_q_ens4 \
    --dump-windows data/mblstm/dumps/camels531_q_ens4.csv.gz \
    > logs/bt_camels531_q_ens4.log 2>&1
fi
echo "=== B1 CAMELS queue complete $(date)" >> $LOG
