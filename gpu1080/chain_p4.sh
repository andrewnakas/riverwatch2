#!/usr/bin/env bash
# MODERN-1 — the 4-PRODUCT members. The last big lever.
#
# Why these three and not four:
#   fused4md4  4-product encoder AND decoder -- the direct analogue of fused3md3 (the
#              champion at 0.893339) and the expected new best, since the decoder finding
#              is that COUNT of target-day products is what pays (+0.015528 with-q,
#              +0.020291 no-q), and this raises that count from 3 to 4.
#   fused4m    4-product encoder, Daymet-only decoder -- the operationally servable shape
#              (the retrospective products have months of latency, so an app cannot serve
#              an all-product decoder).
#   nldasm9    NLDAS single-product, window-matched, for the product-strength axis.
# ⛔ fused4md4q is SKIPPED: --dec-q measured a NULL on this corpus (+0.000021, CI
#    [-0.000416,+0.000319]), so three more runs would buy nothing. That GPU goes to seed
#    depth instead.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
LOG=logs/m1_p4_chain.log
exec >>"$LOG" 2>&1
echo "=== P4 CHAIN START $(date +%F_%T) ==="

# ---- wait for BOTH the corpus and an exclusive slot ------------------------
QUIET=0
for i in $(seq 1 4320); do
  F4=$(ls "$V3/camels_corpus_fused4m_m3_cohort"/*.csv.gz 2>/dev/null | wc -l)
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$F4" -ge 400 ] && [ "$B" -eq 0 ] && [ "$N" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] corpus ready ($F4) and card clear on 3 polls"; break; }
  [ $((i % 60)) -eq 0 ] && echo "[$(date +%T)] waiting: fused4m=$F4 batch=$B gpu=$N quiet=$QUIET/3"
  sleep 120
done
F4=$(ls "$V3/camels_corpus_fused4m_m3_cohort"/*.csv.gz 2>/dev/null | wc -l)
[ "$F4" -lt 400 ] && { echo "4-product corpus never appeared ($F4) - stopping"; exit 2; }

echo "[$(date +%T)] 4-product members, 3-seed screen"
M1_LOG=m1_batch_p4 M1_ALLLEADS=0 bash gpu1080/run_m1_batch.sh \
  fused4md4:501 fused4md4:502 fused4md4:503 \
  fused4m:501 fused4m:502 fused4m:503 \
  nldasm9:501 nldasm9:502 nldasm9:503

# ---- the registered questions ----------------------------------------------
echo "[$(date +%T)] === Q1: does the 4th product lift the champion? fused4md4 vs fused3md3 ==="
echo "    +1 forcing product has measured ~+0.0165 on this corpus; the decoder finding says"
echo "    COUNT at the scored step is what pays, so this is the cleanest test of both."
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused4md4 --against fused3md3 --seeds 501,502,503 \
    --out benchmarks/m1_p4_champion.json

echo "[$(date +%T)] === Q2: the full roster, 4-product members added ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3m,fused3md3,fused3md3q,nhar0m,gridmetm9,fused4md4,fused4m,nldasm9 \
    --seeds 501,502,503 --corr --dup-control \
    --out benchmarks/m1_p4_roster.json

echo "[$(date +%T)] === Q3: 4-product roster vs the settled 3-product one ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3m,fused3md3,fused3md3q,nhar0m,gridmetm9,fused4md4,fused4m,nldasm9 \
    --against fused3m,fused3md3,fused3md3q,nhar0m,gridmetm9 \
    --seeds 501,502,503 --out benchmarks/m1_p4_vs_p3.json
echo "=== P4 CHAIN DONE $(date +%F_%T) ==="
