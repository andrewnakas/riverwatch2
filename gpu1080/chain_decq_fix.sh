#!/usr/bin/env bash
# Wait for the GPU to actually empty, then swap in the fixed launcher and recover
# the dec-q members. Keyed on the RESOURCE (live CUDA processes), never on a log
# marker: on 2026-09-26 a gate keyed on a shared log's "BATCH DONE" matched a
# STALE line from an earlier batch and fired instantly. A GPU slot cannot be stale.
#
# Nothing is retrained: GUARD 1 aborted the dec-q members AFTER a successful
# 30-epoch train, so the checkpoints exist and queue_m1_withq.sh skips training
# when they do. This re-runs the guard and the dumps only.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_decq_fix.log
exec >>"$LOG" 2>&1
echo "=== DECQ FIX CHAIN START $(date +%F_%T) ==="
QUIET=0
for i in $(seq 1 240); do                    # up to 8 h
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$N" -eq 0 ]; then QUIET=$((QUIET + 1)); else QUIET=0; fi
  [ "$QUIET" -ge 2 ] && { echo "[$(date +%T)] GPU empty on two consecutive polls"; break; }
  sleep 120
done
N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$N" -ne 0 ] && { echo "[$(date +%T)] GPU never drained - NOT swapping the launcher"; exit 2; }

# swap in the fixed launcher only once nothing is executing it
cp /tmp/queue_m1_withq.FIXED.sh gpu1080/queue_m1_withq.sh
chmod +x gpu1080/queue_m1_withq.sh
echo "[$(date +%T)] launcher swapped (dec_q width + arm-applied assertion)"
grep -c 'dec_q=' gpu1080/queue_m1_withq.sh | sed 's/^/  dec_q references: /'

# recover the 6 dec-q members (guard + dumps only), then the missing sibling arm
echo "[$(date +%T)] recovering dec-q members, then nclimgridm9"
M1_LOG=m1_batch_decq bash gpu1080/run_m1_batch.sh \
  gridmetmq:501 gridmetmq:502 gridmetmq:503 \
  nclimgridmq:501 nclimgridmq:502 nclimgridmq:503
echo "[$(date +%T)] dec-q recovery done; launching nclimgridm9"
M1_LOG=m1_batch_ncg9 bash gpu1080/run_m1_batch.sh nclimgridm9:501 nclimgridm9:502 nclimgridm9:503
echo "=== DECQ FIX CHAIN DONE $(date +%F_%T) ==="
