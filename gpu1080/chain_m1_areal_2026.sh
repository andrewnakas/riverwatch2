#!/usr/bin/env bash
# NOTE: the month gate it calls now lives in the repo as
# scripts_modern/gate_nldas_month_npz.py. It used to be /tmp/gate_npz.py, which
# a reboot cleared -- this chain would then have reported a gate failure where
# there was only a missing file.
# Backfill the NLDAS-2 areal corpus to today-5, once q_store can support it.
#
# GATE ON THE RESOURCE, NOT A LOG MARKER. build_q_store.py exits 0 even at
# 490/671 ("empty" is not a failure there), and chain_m1_backfill.sh's
# grep -c "^FINISHED" habit has misreported a dead run before. So this waits
# until the ARTIFACT is sufficient: 671 files whose last row reaches the date
# the areal guard will demand.
#
# The deadline outlasts the real interval by a wide margin. A guard shorter
# than the slowest legitimate wait has killed healthy work on this box three
# times, including one that aborted a good 90-minute wait.
set -u
cd /home/nakas/riverwatch2
NEED="2026-10-06"
exec >>logs/m1_areal_chain.log 2>&1
echo "=== chain start $(date -u +%FT%TZ), waiting for q_store to reach $NEED"

DEADLINE=$(( $(date +%s) + 4*3600 ))
while :; do
  N=$(ls data/modern_corpus/q_store/*.csv.gz 2>/dev/null | wc -l)
  # The WORST of a sample, not one file: a single good file proves nothing
  # about the other 670.
  WORST=$(for f in $(ls data/modern_corpus/q_store/*.csv.gz 2>/dev/null | head -20); do
            zcat "$f" 2>/dev/null | tail -1 | cut -d, -f1
          done | sort | head -1)
  if [ "$N" -ge 671 ] && [ -n "$WORST" ] && [ ! "$WORST" '<' "$NEED" ]; then
    echo "[$(date -u +%T)] q_store ready: $N files, worst-of-20 last date $WORST"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[$(date -u +%T)] GATE TIMEOUT: $N files, worst last date '$WORST', needed $NEED"
    exit 1
  fi
  # Abort on a DEAD UPSTREAM rather than on elapsed time: the thing that makes
  # waiting pointless is the producer being gone, not the clock.
  if ! pgrep -f "build_q_store[.]py" >/dev/null; then
    echo "[$(date -u +%T)] upstream q_store builder GONE at $N files (worst '$WORST') -- aborting"
    exit 1
  fi
  sleep 60
done

echo "[$(date -u +%T)] launching areal backfill through $NEED"
nice -n 10 gpu1080/.venv/bin/python scripts_modern/build_nldas2_areal.py \
  --start-year 2026 --end-year 2026 --through-date "$NEED" --workers 6 \
  >> logs/m1_nldas_backfill_2026.log 2>&1
echo "[$(date -u +%T)] backfill exited rc=$?"

echo "[$(date -u +%T)] gating every produced month on bounds + shape"
FAIL=0
for f in data/modern_corpus/_nldas_monthly/nldas_2026*.npz; do
  gpu1080/.venv/bin/python scripts_modern/gate_nldas_month_npz.py "$f" || FAIL=1
done
if [ "$FAIL" = 0 ]; then echo "[$(date -u +%T)] month gate: PASS"; else echo "[$(date -u +%T)] month gate: FAIL"; fi
echo "=== chain done $(date -u +%FT%TZ)."
echo "    Assemble is deliberately NOT automatic: it writes the verified corpus."
