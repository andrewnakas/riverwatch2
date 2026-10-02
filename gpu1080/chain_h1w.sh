#!/usr/bin/env bash
# MODERN-1 — the h1-WEIGHT LADDER probe (PREREG_v2.md MODERN-1).
#
# Gated on chain_shipbar being gone plus batch=0 and gpu=0. ⚠️ chain_p4 waits on the same
# batch=0 condition, so in principle both could see it free at once -- but each requires
# THREE consecutive quiet polls 120 s apart, and run_m1_batch.sh sets batch>0 within
# seconds of starting, so whichever launches first resets the other's counter before it
# can reach 3. No lock needed, and no chain had to be edited while running.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
LOG=logs/m1_h1w_chain.log
exec >>"$LOG" 2>&1
echo "=== H1W CHAIN START $(date +%F_%T) ==="

QUIET=0
for i in $(seq 1 2880); do
  SB=$(ps -eo cmd --no-headers | awk 'index($0,"chain_shipbar.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$SB" -eq 0 ] && [ "$B" -eq 0 ] && [ "$N" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 polls"; break; }
  [ $((i % 60)) -eq 0 ] && echo "[$(date +%T)] waiting: shipbar=$SB batch=$B gpu=$N quiet=$QUIET/3"
  sleep 120
done
[ "$(ps -eo cmd --no-headers | awk 'index($0,"chain_shipbar.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')" -ne 0 ] && {
  echo "shipbar never finished - not starting"; exit 5; }

echo "[$(date +%T)] h1-weight ladder: 0.8 and 1.0 against the frozen 0.5, 3 seeds each"
M1_QUEUE=queue_m1c_withq.sh M1_LOG=m1_batch_h1w M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh \
  fused3md3h8:501 fused3md3h8:502 fused3md3h8:503 \
  fused3md3h10:501 fused3md3h10:502 fused3md3h10:503

echo "[$(date +%T)] === is the alignment lever still rising past 0.5? ==="
echo "    registered: +0.001..+0.006 for w=0.8 over w=0.5, and w=1.0 <= w=0.8"
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3md3h8 --against fused3md3 --seeds 501,502,503 \
    --out benchmarks/m1_h1w_08_vs_05.json
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3md3h10 --against fused3md3h8 --seeds 501,502,503 \
    --out benchmarks/m1_h1w_10_vs_08.json
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3md3h10 --against fused3md3 --seeds 501,502,503 \
    --out benchmarks/m1_h1w_10_vs_05.json
echo "=== H1W CHAIN DONE $(date +%F_%T) ==="
