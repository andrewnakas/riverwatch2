#!/bin/zsh
# Resumable 13-var Open-Meteo corpus trickle. build_mblstm_data.py stops
# itself on sustained rate-limit blocks; this loop re-launches it hourly
# until the corpus is essentially complete. Safe to kill/restart anytime.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
TARGET=1750
while true; do
  # [0-9]* pattern: USGS ids only — exFAT AppleDouble junk (._*) must not count
  n=$(find data/mblstm/corpus_openmeteo/ -name '[0-9]*.csv.gz' 2>/dev/null | wc -l | tr -d ' ')
  echo "=== trickle pass start: $n/$TARGET stations $(date)" >> logs/om_corpus_trickle.log
  if [ "$n" -ge "$TARGET" ]; then
    echo "=== corpus complete ($n) $(date)" >> logs/om_corpus_trickle.log
    break
  fi
  .venv/bin/python scripts/build_mblstm_data.py \
    --out-dir data/mblstm/corpus_openmeteo --sleep 0.7 \
    >> logs/om_corpus_trickle.log 2>&1
  sleep 3600
done
