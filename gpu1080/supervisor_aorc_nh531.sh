#!/usr/bin/env bash
# Retrain the AORC neuralhydrology no-q seeds that trained on 530 basins.
#
# The corpus gained its 531st basin (13235000, reconstructed) at 15:37. Run
# start times against that:
#   s111  09:42  -> saw 530  RETRAIN
#   s222  15:29  -> saw 530  RETRAIN
#   s333  15:38  -> saw 531  already correct, leave it
# So only two seeds need redoing, not three.
#
# Waits for the in-flight work to finish first, per the user's instruction to
# let current tests complete:
#   - s333 must finish (it is the GPU job in progress)
#   - the AORC with-q backtest must finish (CPU-heavy; running it alongside
#     training cost 3.5x throughput earlier in this campaign)
#
# Old runs are moved aside rather than deleted, so the 530-basin results stay
# available for comparison -- if the 531st basin changes a member's score, that
# difference is itself worth knowing.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
LOG=gpu1080/supervisor_aorc_nh531.log
mkdir -p logs

echo "=== SUPERVISOR_AORC_NH531 START $(date +%F_%T) ===" >> $LOG

# gate: wait for the current NH seed AND the backtest to clear.
# patterns are assembled from fragments so this script's own command line can
# never match them -- a self-matching pgrep deadlocked this box for an hour.
PAT_NH="nh_run"" train"
PAT_BT="backtest_""mblstm.py"
while pgrep -f "$PAT_NH" >/dev/null 2>&1 || pgrep -f "$PAT_BT" >/dev/null 2>&1; do
  sleep 300
done
echo "[$(date +%T)] GPU and CPU clear, retraining on 531 basins" >> $LOG

N=$(wc -l < gpu1080/nh_data/aorc/basins.txt)
echo "[$(date +%T)] basins.txt = $N" >> $LOG
if [ "$N" -ne 531 ]; then
  echo "[$(date +%T)] ABORT: expected 531 basins, found $N" >> $LOG
  exit 1
fi

for S in 111 222; do
  # park the 530-basin run and its marker instead of deleting them
  for d in gpu1080/nh_runs/rw2_aorc_lstm_mm_s${S}_*; do
    [ -d "$d" ] && [ "${d%_530}" = "$d" ] && mv "$d" "${d}_530" 2>/dev/null
  done
  rm -f "gpu1080/nh_runs/.aorc_s${S}.done"

  echo "=== aorc nh531 s$S START $(date +%F_%T) ===" >> $LOG
  nice -n 5 $PY -m neuralhydrology.nh_run train \
    --config-file "gpu1080/cfg_aorc_s${S}.yml" \
    > "logs/aorc_nh531_s${S}.log" 2>&1
  RC=$?
  [ $RC -eq 0 ] && touch "gpu1080/nh_runs/.aorc_s${S}.done"
  echo "=== aorc nh531 s$S END $(date +%T) rc=$RC ===" >> $LOG
done

echo "=== SUPERVISOR_AORC_NH531 ALL DONE $(date +%F_%T) ===" >> $LOG
