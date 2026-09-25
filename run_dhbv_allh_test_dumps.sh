#!/usr/bin/env bash
# LEDGER 42 B3 — re-dump the 9 δHBV members over the TEST window with ALL 14
# LEADS, from EXISTING checkpoints. Inference only, no training, no GPU.
#
# ⛔ EMBARGO: outputs are WRITE-ONLY until Stage Q (ledger-42 registration).
# Nothing may read truth or predictions from these files before the query;
# integrity checks are gzip -t + lead count + row count ONLY.
#
# Pattern: run_dhbv_allh_dumps.sh (TRAIN ALLH, verified h=1 bit-identical to
# the old dump) x run_dhbv_dumps.sh (test window + test allowlist).
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
OUT=dhbv_dumps
mkdir -p "$OUT" logs
ALLOW="$OUT/_dhbv_t0_allowlist.json"   # LSTM-aligned stride-14 grid, test side
PAR="${PAR:-2}"
[ -f "$ALLOW" ] || { echo "MISSING $ALLOW"; exit 2; }
dump_one() {
  local F=$1 S=$2
  local CORP="gpu1080/corpora671/camels_corpus_${F}_v2"
  local CKPT="gpu1080/dhbv_ckpts/camels531ls_${F}_dhbvR_s${S}.pt"
  local DUMP="$OUT/camels531ls_${F}_dhbv_ALLH_s${S}.csv.gz"
  local LOG="logs/dhbv_allh_test_${F}_s${S}.log"
  # verify the ARTIFACT: gzip -t AND all 14 leads present (metadata columns only)
  if [ -f "$DUMP" ] && gzip -t "$DUMP" 2>/dev/null; then
    local NH=$(zcat "$DUMP" | tail -n +2 | cut -d, -f3 | sort -un | wc -l)
    if [ "$NH" -eq 14 ]; then echo "[skip] $DUMP (14 leads, valid)"; return 0; fi
    echo "[redo] $DUMP has $NH leads, not 14"
  fi
  [ -f "$CKPT" ] || { echo "[MISS ckpt] $CKPT"; return 1; }
  echo "=== $(date +%H:%M:%S) ALLH-TEST-dumping $F s$S ==="
  nice -n 15 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
    --val-start 1995-10-01 --val-end 2010-09-30 --val-stride 1 \
    --dump-allowlist "$ALLOW" \
    --head dhbv --enc-vars camels1f --no-q-input --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --dump-all-leads --dump-day1 "$DUMP" > "$LOG" 2>&1
  if gzip -t "$DUMP" 2>/dev/null; then
    echo "=== $(date +%H:%M:%S) OK $F s$S rows=$(zcat "$DUMP" | wc -l) ==="
  else
    echo "=== $(date +%H:%M:%S) FAIL $F s$S — bad/absent dump ==="
  fi
}
n=0
for F in daymet nldas maurer; do
  for S in 111 222 333; do
    dump_one "$F" "$S" &
    n=$((n+1))
    if [ $((n % PAR)) -eq 0 ]; then wait; fi
  done
done
wait
echo "=== ALL DHBV ALLH TEST DUMPS DONE $(date +%F_%T) ==="
ls -la "$OUT"/camels531ls_*_dhbv_ALLH_s*.csv.gz
