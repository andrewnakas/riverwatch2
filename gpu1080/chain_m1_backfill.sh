#!/usr/bin/env bash
# MODERN-1: wait for the discharge store to FINISH, verify the ARTIFACT (not the
# exit code), then launch the two 1980-2013 back-extensions unattended.
#
# The builders each carry assert_q_store_covers(), so a premature launch aborts
# rather than writing 34 years of untargeted forcings -- this chain exists so the
# GPU-free hours are not wasted waiting for a human.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
QS=data/modern_corpus/q_store
LOG=logs/m1_chain.log
exec >>"$LOG" 2>&1
echo "=== CHAIN START $(date +%F_%H:%M:%S) ==="

# ---- gate: the q_store must be FINISHED and complete ----------------------
for i in $(seq 1 240); do                   # up to 4 h, polls the RESOURCE
  n=$(ls "$QS"/*.csv.gz 2>/dev/null | wc -l)
  fin=$(grep -c '^FINISHED' logs/m1_q_store.log 2>/dev/null || echo 0)
  if [ "$n" -ge 671 ] && [ "$fin" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] q_store COMPLETE: $n files, log FINISHED"; break
  fi
  [ $((i % 10)) -eq 0 ] && echo "[$(date +%H:%M:%S)] waiting: $n/671 files, finished=$fin"
  sleep 60
done
n=$(ls "$QS"/*.csv.gz 2>/dev/null | wc -l)
if [ "$n" -lt 671 ]; then
  echo "GATE FAIL: q_store stalled at $n/671 - NOT launching the back-extensions"; exit 1
fi
# spot-check three real artifacts rather than trusting the count
for f in $(ls "$QS"/*.csv.gz | head -3); do
  gzip -t "$f" || { echo "GATE FAIL: $f is corrupt"; exit 1; }
done
echo "GATE OK: 671 files, 3 spot-checked clean"

# ---- Daymet 1980-2013 (the long pole, ~32 h) ------------------------------
echo "=== DAYMET 1980-2013 START $(date +%F_%H:%M:%S) ==="
$PY scripts_modern/build_daymet_areal_modern.py \
    --start-year 1980 --end-year 2013 --workers 8 \
    --out data/modern_corpus/_daymet_1980_2013 > logs/m1_daymet_back.log 2>&1
echo "=== DAYMET rc=$? $(date +%F_%H:%M:%S) files=$(ls data/modern_corpus/_daymet_1980_2013/*.csv.gz 2>/dev/null | wc -l) ==="

# ---- NLDAS-2 1980-2013 (~30 h) --------------------------------------------
echo "=== NLDAS 1980-2013 START $(date +%F_%H:%M:%S) ==="
$PY scripts_modern/build_nldas2_areal.py \
    --start-year 1980 --end-year 2013 --workers 12 \
    > logs/m1_nldas_back.log 2>&1
echo "=== NLDAS rc=$? $(date +%F_%H:%M:%S) ==="
echo "=== CHAIN DONE $(date +%F_%H:%M:%S) ==="
