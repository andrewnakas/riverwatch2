#!/usr/bin/env bash
# MODERN-1 — resume NLDAS-2 as the FOURTH forcing product, 1980-2013.
#
# Why this is worth the transfer: "+1 forcing product" is the largest lever
# measured on this corpus (+0.0165 val1), an order of magnitude above any recipe
# flag. NLDAS was PAUSED, not abandoned: its remaining transfer was ~490 GB
# against Daymet's 4.5 GB, so Daymet went first and its throughput rose 2/h -> 10/h.
#
# Resuming is far cheaper than it looks: the builder reduces each month to a ~0.8 MB
# .npz and unlinks the download (build_nldas2_areal.py:179), so the cache is ~335 MB
# for all 408 months, and 155 are ALREADY DONE. Disk is a non-issue; only bandwidth.
#
# ⚠️ Route: M1_NLDAS_PREFER_DIRECT is deliberately NOT set, so the cloud subset
# route is used (~219 GB) rather than DIRECT (~490 GB).
# ⚠️ Starts only after Daymet reaches 671 -- its last basins are on the critical
# path of chain_fused3m2.sh, whose stall detector allows 3 h, and stealing
# bandwidth from them could trip it.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
LOG=logs/m1_nldas4_chain.log
exec >>"$LOG" 2>&1
echo "=== NLDAS4 CHAIN START $(date +%F_%T) ==="

SRC=data/modern_corpus/_daymet_1980_2013
LAST=-1; STALL=0
for i in $(seq 1 2880); do
  n=$(ls "$SRC"/*.csv.gz 2>/dev/null | wc -l)
  [ "$n" -ge 671 ] && { echo "[$(date +%T)] daymet complete ($n/671) - starting NLDAS"; break; }
  if [ "$n" -gt "$LAST" ]; then LAST=$n; STALL=0; else STALL=$((STALL + 1)); fi
  [ "$STALL" -ge 90 ] && { echo "daymet stalled at $n/671 - starting NLDAS anyway"; break; }
  sleep 120
done

# workers 6, not the default 12: the GPU batch controller holds a job whenever
# MemAvailable drops under 3500 MiB, so a greedy netCDF decoder here would stall
# Gate A and the fused family -- the actual critical path.
echo "[$(date +%T)] NLDAS 1980-2013, subset route, workers 6"
$PY scripts_modern/build_nldas2_areal.py \
    --start-year 1980 --end-year 2013 --workers 6 \
    --out data/modern_corpus/_nldas_1980_2013
echo "[$(date +%T)] builder rc=$?"
nn=$(ls data/modern_corpus/_nldas_1980_2013/*.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] nldas 1980-2013: $nn/671 basins"
if [ "$nn" -lt 671 ]; then
  echo "INCOMPLETE: $nn/671 - NOT assembling. A partial product must never reach a corpus;"
  echo "  the uploader/corpus preflight refuses anything below 671/671 for exactly this reason."
  exit 3
fi
echo "=== NLDAS4 CHAIN DONE $(date +%F_%T) ==="
