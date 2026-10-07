#!/usr/bin/env bash
# MODERN-1 LEAKAGE AUDIT — the decisive gridMET registration test.
#
# The question this settles: does the gridMET one-day offset INFLATE the modern record?
#   corrected scores HIGHER ⇒ the misregistration was a COST; both records stand, understated.
#   corrected scores LOWER  ⇒ the original was exploiting future rain ⇒ RETRACT and rebuild.
#
# Two earlier attempts could not answer it, both because the column set was wrong:
#   fix  (take 1) shifted precipitation ONLY            -> -0.011137, but de-synchronised precip
#                                                          from tmin INSIDE gridMET
#   fix2 (take 2) shifted precip + tmin + vapor_pressure -> vp is ALIGNED (shifted wrongly) and
#                                                          tmean was missed
# fix3 follows benchmarks/m1_leakaudit_threeway.json: lag-scanned against camels_corpus_nldas_m3,
# the only HOURLY-NATIVE product (our builder makes its calendar days, so its labelling is derived
# rather than inherited). Shift precip + tmin, then RECOMPUTE tmean=(tmax+tmin)/2; leave tmax, srad
# and vapor_pressure. Verified by analysis/m1_tmean_repair_check.py: the recompute beats the
# column-shift in 100% of 60 basins (RMSE 1.63 vs 2.27), a difference the lag scan CANNOT see.
#
# Every offset is explained by gridMET's two upstream sources: PRISM precip is 12Z-12Z and PRISM
# tmin is attributed to the following morning (both +1), PRISM tmax is calendar-day (0), and srad
# and vapor pressure are carried from NLDAS-2 (0) -- measured here as gridmet-srad = 0.99941 x
# nldas-srad, r 0.9996, where daymet/nldas on the same rows reads 0.8437.
#
# ⚠️ fix2 is NOT cancelled, deliberately: fix2-vs-fix3 is the only clean read on vapor_pressure.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_gridfix3_chain.log
exec >>"$LOG" 2>&1
echo "=== GRIDFIX3 CHAIN START $(date +%F_%T) ==="

# Gate on the PRODUCER being gone, not on a quiet card: a job is invisible to nvidia-smi for
# minutes while it loads, and a 20-minute CPU phase makes a busy box look idle.
QUIET=0
for i in $(seq 1 2880); do
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  T=$(ps -eo cmd --no-headers | awk 'index($0,"train_mblstm.py")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$B" -eq 0 ] && [ "$T" -eq 0 ] && [ "$N" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 consecutive polls"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: batch=$B train=$T gpu=$N quiet=$QUIET/3"
  sleep 120
done
B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
[ "$B" -ne 0 ] && { echo "ABORT: a batch is still running after the wait window"; exit 5; }

echo "[$(date +%T)] training gridmetm9fix3, seeds 501-503"
M1_QUEUE=queue_m1e_withq.sh M1_LOG=m1_batch_gridfix3 M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh gridmetm9fix3:501 gridmetm9fix3:502 gridmetm9fix3:503
echo "[$(date +%T)] training returned rc=$?"

PY=gpu1080/.venv/bin/python
for f in 3 2 1; do
  case $f in 3) M=gridmetm9fix3 ;; 2) M=gridmetm9fix2 ;; 1) M=gridmetm9fix ;; esac
  n=$(ls data/mblstm/l51_dumps/m1_${M}_s50?_val1.csv.gz 2>/dev/null | wc -l)
  echo "[$(date +%T)] $M: $n val1 dumps present"
done

echo "[$(date +%T)] ===== THE DECISIVE READ: corrected vs original gridMET ====="
# val1, not test1: this selects whether a record is retracted, and test is read once at the end.
$PY analysis/m1_composition.py --frame val1 --members gridmetm9fix3 --against gridmetm9 \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_leakaudit_gridfix3_vs_orig.json 2>&1 | tail -20

echo "[$(date +%T)] ===== isolating vapor_pressure: fix3 (vp left alone) vs fix2 (vp shifted) ====="
$PY analysis/m1_composition.py --frame val1 --members gridmetm9fix3 --against gridmetm9fix2 \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_leakaudit_gridfix3_vs_fix2.json 2>&1 | tail -14

echo "[$(date +%T)] ===== isolating the de-sync: fix3 (precip+tmin) vs fix (precip only) ====="
$PY analysis/m1_composition.py --frame val1 --members gridmetm9fix3 --against gridmetm9fix \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_leakaudit_gridfix3_vs_fix1.json 2>&1 | tail -14

echo "=== GRIDFIX3 CHAIN DONE $(date +%F_%T) ==="
