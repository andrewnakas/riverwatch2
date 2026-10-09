#!/usr/bin/env bash
# GATE 1 — fill the Open-Meteo point corpus across Open-Meteo's HOURLY quota.
#
# One hourly window buys ~43 of the 150 registered basins (each request is
# ~1,826 days x 7 variables, and the free tier weights by volume), so the build
# needs ~4 windows. The builder is resume-safe: it skips any basin whose file
# already exists and writes via a .tmp rename, so a quota cut-off never leaves a
# partial file.
#
# The deadline keys on PROGRESS, not elapsed time: it stops when a whole window
# adds no basins (quota permanently changed, creds broken, network gone), not
# when some clock runs out. A wall-clock deadline once killed a healthy build
# here and left the card idle.
set -u
cd "$HOME/riverwatch2" || exit 1

TARGET=150
OUT=data/modern_corpus/v3/camels_corpus_openmeteo_m3_cohort
LOG=logs/gate1_om_corpus.log
PY=gpu1080/.venv/bin/python
MAX_WINDOWS=10
STALL_LIMIT=2          # consecutive windows with zero new basins -> give up

count() { ls "$OUT" 2>/dev/null | wc -l | tr -d ' '; }

stalls=0
for w in $(seq 1 "$MAX_WINDOWS"); do
  before=$(count)
  if [ "$before" -ge "$TARGET" ]; then
    echo "[chain] complete: $before/$TARGET" >> "$LOG"
    break
  fi
  echo "[chain] window $w starting at $before/$TARGET ($(date -u +%FT%TZ))" >> "$LOG"
  $PY scripts_modern/build_openmeteo_point_corpus.py \
      --subsample 150 --seed 0 \
      --start 2013-09-01 --end 2018-09-30 --sleep 2.5 >> "$LOG" 2>&1
  after=$(count)
  echo "[chain] window $w done: $before -> $after" >> "$LOG"
  if [ "$after" -ge "$TARGET" ]; then
    echo "[chain] complete: $after/$TARGET" >> "$LOG"
    break
  fi
  if [ "$after" -le "$before" ]; then
    stalls=$((stalls + 1))
    echo "[chain] WARNING no progress this window (stall $stalls/$STALL_LIMIT)" >> "$LOG"
    if [ "$stalls" -ge "$STALL_LIMIT" ]; then
      echo "[chain] ABORT: two consecutive windows added nothing" >> "$LOG"
      exit 1
    fi
  else
    stalls=0
  fi
  echo "[chain] sleeping 3900s for the next hourly quota window" >> "$LOG"
  sleep 3900
done
echo "[chain] finished at $(count)/$TARGET ($(date -u +%FT%TZ))" >> "$LOG"
