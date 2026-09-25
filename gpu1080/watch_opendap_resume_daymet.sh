#!/usr/bin/env bash
# MODERN-1: opendap.earthdata.nasa.gov went 503 service-wide on 2026-09-23
# (every collection, even bare .dmr metadata, while plain HTTPS data
# distribution kept returning 200). NLDAS has a direct-download fallback and
# keeps running; Daymet does NOT -- it needs OPeNDAP's SPATIAL subsetting,
# because the alternative is pulling whole 1 km CONUS annual grids.
#
# So: poll the RESOURCE, not a file, on a deadline, and resume the build the
# moment the service returns. Daymet's builder is idempotent per basin (it skips
# any basin whose output already has the right row count), so resuming is free.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_opendap_watch.log
URL="https://opendap.earthdata.nasa.gov/collections/C2532426483-ORNL_CLOUD/granules/Daymet_Daily_V4R1.daymet_v4_daily_na_prcp_1982.nc.dmr"
DEADLINE=$(( $(date +%s) + 172800 ))      # 48 h, then give up loudly
echo "=== OPeNDAP WATCH START $(date +%F_%T) ===" >> "$LOG"
while :; do
  code=$(curl -s -n -L -c /tmp/owck -b /tmp/owck --max-time 60 -o /dev/null -w "%{http_code}" "$URL" || echo 000)
  now=$(date +%s)
  if [ "$code" = "200" ]; then
    echo "[$(date +%F_%T)] OPeNDAP is BACK (http=$code) - resuming Daymet" >> "$LOG"
    break
  fi
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "[$(date +%F_%T)] DEADLINE HIT: OPeNDAP still http=$code after 48 h. Daymet NOT resumed." >> "$LOG"
    exit 2
  fi
  echo "[$(date +%F_%T)] still down (http=$code)" >> "$LOG"
  sleep 600
done
# confirm twice, so a single lucky probe does not restart into a flapping service
sleep 120
code2=$(curl -s -n -L -c /tmp/owck -b /tmp/owck --max-time 60 -o /dev/null -w "%{http_code}" "$URL" || echo 000)
if [ "$code2" != "200" ]; then
  echo "[$(date +%F_%T)] second probe http=$code2 - flapping, staying down" >> "$LOG"
  exec bash gpu1080/watch_opendap_resume_daymet.sh
fi
echo "[$(date +%F_%T)] confirmed twice; relaunching Daymet 1980-2013" >> "$LOG"
nohup setsid gpu1080/.venv/bin/python scripts_modern/build_daymet_areal_modern.py \
  --start-year 1980 --end-year 2013 --workers 6 \
  --out data/modern_corpus/_daymet_1980_2013 >> logs/m1_daymet_back.log 2>&1 < /dev/null &
echo "[$(date +%F_%T)] Daymet relaunched (6 workers, gentler on a just-recovered service)" >> "$LOG"
