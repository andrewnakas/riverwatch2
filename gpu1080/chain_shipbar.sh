#!/usr/bin/env bash
# MODERN-1 — ship-bar seeds (504, 505) for the with-q MB members.
#
# Why these four and why now: all four carry CI-positive LOO inside the settled 5-member
# roster, so they are very likely in the final composition whatever NLDAS adds. 3 seeds is
# a SCREEN; the ship bar is 5, and no record can be frozen below it without --screen-ok.
# NLDAS is ~63 h out at its current (18x degraded) rate, so this is the right use of the
# card in the gap rather than leaving it idle.
#
# ⛔ NOT the NH member: nhar0m's LOO is the smallest of the five (+0.000187 for its
# 3-product twin) and each NH seed is ~12 h, so 2 more seeds would cost 24 h for the
# roster's weakest contributor. Deferred until the composition is final.
# ⛔ NOT the no-q members: wait to see which survive their own composition before spending
# seed depth on them.
# ⛔ NOT all-leads dumps: those belong to the FINAL roster, once NLDAS has settled it.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_shipbar_chain.log
exec >>"$LOG" 2>&1
echo "=== SHIPBAR CHAIN START $(date +%F_%T) ==="

# one waiter, gated on the no-q chain being GONE (reachable: it exits either way)
QUIET=0
for i in $(seq 1 2880); do
  CQ=$(ps -eo cmd --no-headers | awk 'index($0,"chain_noq.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$CQ" -eq 0 ] && [ "$B" -eq 0 ] && [ "$N" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 polls"; break; }
  [ $((i % 60)) -eq 0 ] && echo "[$(date +%T)] waiting: noq=$CQ batch=$B gpu=$N quiet=$QUIET/3"
  sleep 120
done
CQ=$(ps -eo cmd --no-headers | awk 'index($0,"chain_noq.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
[ "$CQ" -ne 0 ] && { echo "no-q chain never finished - not starting"; exit 5; }

echo "[$(date +%T)] with-q ship-bar seeds 504/505 for the four MB members"
M1_LOG=m1_batch_shipbar M1_ALLLEADS=0 bash gpu1080/run_m1_batch.sh \
  fused3md3:504 fused3md3:505 \
  fused3md3q:504 fused3md3q:505 \
  fused3m:504 fused3m:505 \
  gridmetm9:504 gridmetm9:505

PY=gpu1080/.venv/bin/python
echo "[$(date +%T)] === the roster at 5 seeds vs the same roster at 3 (does depth hold?) ==="
# ⚠️ Seed depth helps the MEMBER and can HURT the ensemble -- measured in this campaign
# (2->3 seeds: solo +0.0024 while the ensemble median FELL). So this is a real question,
# not a formality, and the SHIP-BAR number is the one that gets quoted either way.
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3m,fused3md3,fused3md3q,nhar0m,gridmetm9 \
    --seeds 501,502,503,504,505 --corr \
    --out benchmarks/m1_withq_roster_5seed.json
echo "[$(date +%T)] (nhar0m contributes only its 3 seeds; the scorer uses what exists per member)"
echo "=== SHIPBAR CHAIN DONE $(date +%F_%T) ==="
