#!/bin/zsh
# A1 Google-overlap re-score, queued behind B1 (2026-07-06). First attempt ran
# concurrently with the CAMELS trainer and was fully paged out within minutes
# (0% CPU, 0 RSS, +3.8 GB swap) — this box handles ONE heavy torch job at a
# time. Waits for the B1 round-2 supervisor's completion marker (all seeds +
# ens8 eval), then runs the 527-gauge shipped-config backtest (EXPERIMENTS
# row 20).
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
LOG=logs/aifl527_queue.log
echo "=== aifl527 queued $(date)" >> $LOG

# Wait on the round's true completion artifact — the ens8 eval JSON. (A log
# grep is unreliable here: b1_round2.log is append-only across rounds, so the
# July-5 run's "complete" line already matches.)
until [ -f benchmarks/mblstm_backtest_camels531_mse_ens8.json ]; do
  sleep 900
done
# Paranoia: no other heavy job (trainer restarted manually, stray eval).
while pgrep -f "train_mblstm.p[y]" > /dev/null \
   || pgrep -f "backtest_mblstm.p[y]" > /dev/null \
   || pgrep -f "backtest_blend_2026.p[y]" > /dev/null; do
  sleep 600
done

echo "starting aifl527 run $(date)" >> $LOG
CK="data/mblstm/model_h256_s101_cmalv2p.pt:data/mblstm/model_h256_s102_cmalv2p.pt:data/mblstm/model_h256_s103_cmalv2p.pt:data/mblstm/model_h256_s104_cmalv2p.pt"
RW2_ENABLE_MBLSTM=1 caffeinate -i .venv/bin/python scripts/backtest_mblstm.py \
  --ckpt "$CK" --stations-file data/grdc_usgs_crosswalk.json \
  --forcing-plan ecmwf:1-14 --point mean \
  --label aifl_google_527 \
  --dump-windows data/mblstm/dumps/aifl527_ecmwf.csv.gz \
  > logs/bt_aifl527.log 2>&1 \
  && echo "aifl527 done $(date)" >> $LOG \
  || echo "aifl527 FAILED $(date)" >> $LOG
