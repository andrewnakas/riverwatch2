#!/usr/bin/env bash
# Train the SUB-DAILY INTENSITY members and compare against plain AORC.
#
# Justified by the stride-1 phase probe, which split the hypothesis in two:
#   T1/T2  NO LAG. Shift 0 is optimal for all 16 basins and the per-basin
#          oracle best shift buys +0.0000. The routing/timing reading is dead.
#   T3     but on the 1% of days carrying 93% of squared error, rising limbs
#          under-predict by -2071 cfs against -14 cfs on ordinary days, a 144x
#          asymmetry.
#
# So the model is ON TIME and TOO SMALL on storm peaks. A daily precipitation
# sum cannot distinguish a 10 mm drizzle from a 10 mm cloudburst -- measured on
# our own data, same-total days differ 16x in peak hourly rate. p_max_1h,
# p_max_3h and p_cv encode exactly that.
#
# The configs differ from cfg_aorc_s111.yml in ONLY experiment name, the four
# data paths, and the five extra dynamic inputs -- verified by an asserted diff.
# So a score difference is attributable to the sub-daily signal and nothing else.
#
# GATE: waits for the neuralhydrology retrain and any train_mblstm job. Patterns
# are assembled from fragments so this script's own command line cannot match
# them; a self-matching pgrep deadlocked this box for an hour earlier.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
LOG=gpu1080/supervisor_aorcx.log
mkdir -p logs

echo "=== SUPERVISOR_AORCX START $(date +%F_%T) ===" >> $LOG

PAT_NH="nh_run"" train"
PAT_TR="train_mblstm.py --epo""chs 30"
while pgrep -f "$PAT_NH" >/dev/null 2>&1 || pgrep -f "$PAT_TR" >/dev/null 2>&1; do
  sleep 300
done
echo "[$(date +%T)] GPU free" >> $LOG

N=$(wc -l < gpu1080/nh_data/aorcx/basins.txt)
echo "[$(date +%T)] aorcx basins = $N" >> $LOG
[ "$N" -eq 531 ] || { echo "[$(date +%T)] ABORT: expected 531" >> $LOG; exit 1; }

for S in 111 222; do
  MARK="gpu1080/nh_runs/.aorcx_s${S}.done"
  [ -f "$MARK" ] && { echo "[skip] s$S" >> $LOG; continue; }
  echo "=== aorcx s$S START $(date +%F_%T) ===" >> $LOG
  nice -n 5 $PY -m neuralhydrology.nh_run train \
    --config-file "gpu1080/cfg_aorcx_s${S}.yml" \
    > "logs/aorcx_s${S}.log" 2>&1
  RC=$?
  [ $RC -eq 0 ] && touch "$MARK"
  echo "=== aorcx s$S END $(date +%T) rc=$RC ===" >> $LOG
done

echo "=== SUPERVISOR_AORCX ALL DONE $(date +%F_%T) ===" >> $LOG
