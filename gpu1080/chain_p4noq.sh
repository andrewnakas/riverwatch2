#!/usr/bin/env bash
# MODERN-1 — the 4-PRODUCT NO-Q members, sequenced AFTER chain_p4.
#
# ⚠️ Gated on chain_p4 being GONE, not merely on an idle card: two independent waiters on
# the same corpus+slot condition would both fire on the same drain. Reachable in every
# branch -- chain_p4 exits whether its batch succeeds or the corpus never appears.
#
# Uses queue_m1b_noq.sh (a copy of the no-q launcher plus the 4-product members), written
# as a separate file because queue_m1_noq.sh was running the no-q Gate B at the time.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
LOG=logs/m1_p4noq_chain.log
exec >>"$LOG" 2>&1
echo "=== P4-NOQ CHAIN START $(date +%F_%T) ==="

QUIET=0
for i in $(seq 1 4320); do
  P4=$(ps -eo cmd --no-headers | awk 'index($0,"chain_p4.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  F4=$(ls "$V3/camels_corpus_fused4m_m3_cohort"/*.csv.gz 2>/dev/null | wc -l)
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$P4" -eq 0 ] && [ "$F4" -ge 400 ] && [ "$B" -eq 0 ] && [ "$N" -eq 0 ]; then
    QUIET=$((QUIET+1)); else QUIET=0
  fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear: p4=0 corpus=$F4 batch=0 gpu=0 on 3 polls"; break; }
  [ $((i % 60)) -eq 0 ] && echo "[$(date +%T)] waiting: p4=$P4 corpus=$F4 batch=$B gpu=$N quiet=$QUIET/3"
  sleep 120
done
F4=$(ls "$V3/camels_corpus_fused4m_m3_cohort"/*.csv.gz 2>/dev/null | wc -l)
[ "$F4" -lt 400 ] && { echo "4-product corpus never appeared ($F4) - stopping"; exit 2; }

echo "[$(date +%T)] 4-product NO-Q members, 3-seed screen"
M1_QUEUE=queue_m1b_noq.sh M1_LOG=m1_batch_p4noq M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh \
  lstm_fused4md4:501 lstm_fused4md4:502 lstm_fused4md4:503 \
  lstm_fused4m:501 lstm_fused4m:502 lstm_fused4m:503 \
  lstm_nldasm:501 lstm_nldasm:502 lstm_nldasm:503

# arm audit before any score -- both arms share a protocol and a frame
echo "[$(date +%T)] arm audit"
$PY - <<'PYA'
import glob, json
bad = [f for f in sorted(glob.glob("data/mblstm/l51_dumps/m1_lstm_*_val1.json"))
       if json.load(open(f)).get("arm") != "noq"]
print(f"  violations: {len(bad)}")
for b in bad:
    print(f"    {b}")
assert not bad, "a no-q dump does not declare arm=noq"
PYA
[ $? -eq 0 ] || { echo "ABORT: arm audit failed"; exit 7; }

echo "[$(date +%T)] === Q1: does the 4th product lift the no-q champion? ==="
echo "    decoder count effect is LARGER without discharge (+0.020291 vs +0.015528)"
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members lstm_fused4md4 --against lstm_fused3md3 --seeds 501,502,503 \
    --out benchmarks/m1_p4noq_champion.json

echo "[$(date +%T)] === Q2: the no-q roster with 4-product members ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members lstm_fused3md3,lstm_fused3m,lstm_gridmetm,lstm_fused4md4,lstm_fused4m,lstm_nldasm \
    --seeds 501,502,503 --corr --dup-control \
    --out benchmarks/m1_p4noq_roster.json

echo "[$(date +%T)] === Q3: vs the settled 3-member no-q roster ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members lstm_fused3md3,lstm_fused3m,lstm_gridmetm,lstm_fused4md4,lstm_fused4m,lstm_nldasm \
    --against lstm_fused3md3,lstm_fused3m,lstm_gridmetm \
    --seeds 501,502,503 --out benchmarks/m1_p4noq_vs_p3.json
echo "=== P4-NOQ CHAIN DONE $(date +%F_%T) ==="
