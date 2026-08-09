#!/usr/bin/env bash
# Dump the AORC neuralhydrology members (plain + intensity) to the no-q stream format.
#
# Four members, all trained on the full 531-basin corpus:
#   aorc  s111 s222 s333   the plain 4th-forcing member
#   aorcx s111 (s222 later) the sub-daily intensity variant
#
# Recipe copied from supervisor_084.sh, which produced the existing ls dumps:
#   nh-run evaluate --period test  ->  test_results.p
#   scripts/nh_to_dump.py          ->  camels531ls_<forcing>_nhlstm_s<seed>.csv.gz
# Any deviation would put a differently-built member into gate_eval's streams.
#
# Also emits TRAIN-period dumps. gate_eval fits every combination rule on the
# train window, so a member without a _TRAIN dump cannot be gated and would have
# to be included on faith.
#
# The _530 run directories are the pre-reconstruction versions kept for
# comparison; the globs below deliberately exclude them by taking the newest
# matching directory without that suffix.
set -u
cd ~/riverwatch2
export PATH="$PWD/gpu1080/.venv/bin:$PATH"
LOG=logs/dump_aorc_nh.log
mkdir -p logs gpu1080/dumps

echo "=== AORC NH DUMPS START $(date +%F_%T) ===" >> $LOG

# wait for training to clear; patterns split so this script cannot self-match
PAT_NH="nh_run"" train"
while pgrep -f "$PAT_NH" >/dev/null 2>&1; do sleep 300; done
echo "[$(date +%T)] GPU free" >> $LOG

dump_one() {
  local FORCING=$1 SEED=$2 PERIOD=$3
  local SUFFIX=""; [ "$PERIOD" = "train" ] && SUFFIX="_TRAIN"
  local OUT="gpu1080/dumps/camels531ls_${FORCING}_nhlstm${SUFFIX}_s${SEED}.csv.gz"
  [ -f "$OUT" ] && { echo "[skip] $OUT" >> $LOG; return; }

  # newest run dir for this member that is NOT a parked _530 version
  local RUN
  RUN=$(ls -dt gpu1080/nh_runs/rw2_${FORCING}_lstm_mm_s${SEED}_* 2>/dev/null \
        | grep -v '_530$' | head -1)
  [ -z "$RUN" ] && { echo "[MISS] no run dir for $FORCING s$SEED" >> $LOG; return; }

  echo "=== $FORCING s$SEED $PERIOD ($(basename $RUN)) $(date +%T) ===" >> $LOG
  nh-run evaluate --run-dir "$RUN" --period "$PERIOD" \
    >> "logs/eval_${FORCING}_s${SEED}_${PERIOD}.log" 2>&1
  local RES
  RES=$(ls "$RUN"/${PERIOD}/model_epoch030/${PERIOD}_results.p 2>/dev/null | head -1)
  [ -z "$RES" ] && { echo "[FAIL] no results.p for $FORCING s$SEED $PERIOD" >> $LOG; return; }
  gpu1080/.venv/bin/python scripts/nh_to_dump.py --results "$RES" \
    --forcing "$FORCING" --out "$OUT" \
    >> "logs/eval_${FORCING}_s${SEED}_${PERIOD}.log" 2>&1
  echo "=== $FORCING s$SEED $PERIOD done rows=$(zcat "$OUT" 2>/dev/null | wc -l) ===" >> $LOG
}

for S in 111 222 333; do
  dump_one aorc "$S" test
  dump_one aorc "$S" train
done
for S in 111 222; do
  dump_one aorcx "$S" test
  dump_one aorcx "$S" train
done

echo "=== AORC NH DUMPS DONE $(date +%F_%T) ===" >> $LOG
ls -la gpu1080/dumps/camels531ls_aorc*_nhlstm*.csv.gz >> $LOG 2>&1
