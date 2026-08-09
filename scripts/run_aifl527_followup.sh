#!/bin/zsh
# Fires after the queued aifl527 backtest writes its dump: scores flood F1 on
# it (matched-quantile thresholds — 1 year of simulation can't support
# own-record) and drops a marker for the comparison step. Light CPU.
# NOTE: the ecmwf:1-14 plan has weekly 2025 inits (52), so per-lead mode would
# leave 6-day NaN gaps that split events — chained mode only here. A true
# per-lead F1 vs Google's leads 0-7 needs a gfs:1-14 run (~244 inits) later.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
LOG=logs/aifl527_followup.log
echo "=== followup armed $(date)" >> $LOG
until [ -f data/mblstm/dumps/aifl527_ecmwf.csv.gz ] \
   && [ -f benchmarks/mblstm_backtest_aifl_google_527.json ]; do
  sleep 900
done
sleep 60  # let the writer finish flushing
echo "scoring flood F1 on aifl527 dump $(date)" >> $LOG
.venv/bin/python scripts/score_flood_f1.py \
  --dump data/mblstm/dumps/aifl527_ecmwf.csv.gz \
  --corpus-dir data/mblstm/corpus \
  --mode chained --sim-thresholds matched-quantile \
  --point ymean --label aifl527_chained \
  > logs/flood_f1_aifl527.log 2>&1 \
  && echo "flood F1 done $(date)" >> $LOG \
  || echo "flood F1 FAILED $(date)" >> $LOG
