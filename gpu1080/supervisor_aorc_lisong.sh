#!/usr/bin/env bash
# Retrain the AORC no-q members on the CORRECT Li/Song split.
#
# The previous members used the Kratzert split (train 1999-2008, test 1989-99)
# because their config was cloned from a with-q era file. Every existing no-q
# stream uses Li/Song (train 1980-95, test 1995-2010), so the grids did not
# overlap at all -- an inner join against an existing stream returned zero rows,
# and the members could never have joined the ensemble.
#
# Configs regenerated from nh_config_lisong.yml.tmpl and validated against
# cfgls_multi_s111.yml, a known-good no-q config, rather than against whatever
# file happened to be nearest. That validation is the check that would have
# caught the original mistake.
#
# Gate patterns are assembled from fragments so this script's own command line
# cannot match them; a self-matching pgrep deadlocked this box for an hour.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
LOG=gpu1080/supervisor_aorc_lisong.log
mkdir -p logs

echo "=== SUPERVISOR_AORC_LISONG START $(date +%F_%T) ===" >> $LOG

PAT_NH="nh_run"" train"
PAT_TR="train_mblstm.py --epo""chs 30"
while pgrep -f "$PAT_NH" >/dev/null 2>&1 || pgrep -f "$PAT_TR" >/dev/null 2>&1; do
  sleep 300
done
echo "[$(date +%T)] GPU free" >> $LOG

N=$(wc -l < gpu1080/nh_data/aorc/basins.txt)
[ "$N" -eq 531 ] || { echo "[$(date +%T)] ABORT: aorc basins=$N, expected 531" >> $LOG; exit 1; }
echo "[$(date +%T)] aorc basins = $N" >> $LOG

for S in 111 222 333; do
  MARK="gpu1080/nh_runs/.aorcls_s${S}.done"
  [ -f "$MARK" ] && { echo "[skip] s$S" >> $LOG; continue; }
  CFG="gpu1080/cfgls_aorc_s${S}.yml"
  [ -f "$CFG" ] || { echo "[MISS] $CFG" >> $LOG; continue; }

  echo "=== aorc-lisong s$S START $(date +%F_%T) ===" >> $LOG
  nice -n 5 $PY -m neuralhydrology.nh_run train --config-file "$CFG" \
    > "logs/aorcls_s${S}.log" 2>&1
  RC=$?
  [ $RC -eq 0 ] && touch "$MARK"
  echo "=== aorc-lisong s$S END $(date +%T) rc=$RC ===" >> $LOG

  # dump immediately while the next seed trains: evaluation is inference and
  # does not need exclusive GPU, so the queue never blocks on dumping.
  RUN=$(ls -dt gpu1080/nh_runs/rw2ls_aorc_lstm_mm_s${S}_* 2>/dev/null | head -1)
  [ -z "$RUN" ] && { echo "[MISS] no run dir s$S" >> $LOG; continue; }
  for PERIOD in test train; do
    SUF=""; [ "$PERIOD" = "train" ] && SUF="_TRAIN"
    OUT="gpu1080/dumps/camels531ls_aorc_nhlstm${SUF}_s${S}.csv.gz"
    [ -f "$OUT" ] && continue
    gpu1080/.venv/bin/nh-run evaluate --run-dir "$RUN" --period "$PERIOD" \
      >> "logs/evalls_aorc_s${S}_${PERIOD}.log" 2>&1
    RES="$RUN/$PERIOD/model_epoch030/${PERIOD}_results.p"
    [ -f "$RES" ] || { echo "[FAIL] no $PERIOD results s$S" >> $LOG; continue; }
    $PY scripts/nh_to_dump.py --results "$RES" --forcing aorc --out "$OUT" \
      >> "logs/evalls_aorc_s${S}_${PERIOD}.log" 2>&1
    echo "[dump] s$S $PERIOD rows=$(zcat "$OUT" 2>/dev/null | wc -l)" >> $LOG
  done
done

echo "=== SUPERVISOR_AORC_LISONG ALL DONE $(date +%F_%T) ===" >> $LOG
