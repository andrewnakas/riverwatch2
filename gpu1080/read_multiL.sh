#!/usr/bin/env bash
# LEDGER 50 — the multiL read, encoding the PRE-REGISTERED decision rule so it cannot be
# fudged after the fact. Written 2026-09-01, BEFORE any multiL dump existed.
#
#   stage 1  verify the artifacts (gzip, 531 basins, 14 leads, TRAIN+TEST present)
#   stage 2  SOLO screen: multiL k-seed stream vs a MATCHED k-seed lstm_multi subset, h==1 held-out
#            k=2 (interim, early-stop only): stop the arm iff delta <= -0.005
#            k=3 (the registered read):      continue iff delta >= +0.0057, else STOP
#   stage 3  ENSEMBLE swap + add under the ship bar — RUN ONLY IF STAGE 2 PASSED.
#
# Usage: bash gpu1080/read_multiL.sh <k>       (k = number of multiL seeds available)
set -u
cd "$HOME/riverwatch2" || exit 1
K=${1:?number of seeds}
PY=gpu1080/.venv/bin/python
D=gpu1080/dumps
case "$K" in
  2) SEEDS="s111,s222";        BAR=""      ; STOPBAR="-0.005" ;;
  3) SEEDS="s111,s222,s333";   BAR="0.0057"; STOPBAR="" ;;
  *) echo "k must be 2 or 3"; exit 2 ;;
esac
IFS=',' read -ra SL <<< "$SEEDS"

echo "=== STAGE 1 — artifacts ==="
for s in "${SL[@]}"; do
  for suf in "" "_TRAIN"; do
    f="$D/camels531ls_multiL_nhlstm${suf}_${s}.csv.gz"
    [ -f "$f" ] || { echo "MISSING $f"; exit 3; }
    gzip -t "$f" || { echo "CORRUPT $f"; exit 3; }
    nb=$(zcat "$f" | tail -n +2 | cut -d, -f1 | sort -u | wc -l)
    nh=$(zcat "$f" | tail -n +2 | cut -d, -f3 | sort -un | wc -l)
    echo "  ok $(basename $f)  basins=$nb leads=$nh"
    [ "$nb" -eq 531 ] || { echo "  ⚠️ expected 531 basins"; exit 3; }
  done
done

echo "=== STAGE 2 — SOLO screen at k=$K, matched depth ==="
$PY ledger50_gate.py --solo $(for s in "${SL[@]}"; do echo -n "$D/camels531ls_multiL_nhlstm_${s}.csv.gz "; done) \
  --out benchmarks/ledger50_multiL_solo_k${K}.json | tee /tmp/_ml_solo.txt
ML=$(grep "seed-average" /tmp/_ml_solo.txt | grep -oE "[0-9]\.[0-9]+" | head -1)
# matched-depth incumbent: the first K of lstm_multi's canonical seed order
MSEEDS=$(echo "s111,s222,s3334" | cut -d, -f1-$K)
IFS=',' read -ra MS <<< "$MSEEDS"
$PY ledger50_gate.py --solo $(for s in "${MS[@]}"; do echo -n "$D/camels531ls_multi_nhlstm_${s}.csv.gz "; done) \
  --out benchmarks/ledger50_multi_solo_k${K}.json | tee /tmp/_mm_solo.txt
MM=$(grep "seed-average" /tmp/_mm_solo.txt | grep -oE "[0-9]\.[0-9]+" | head -1)
DELTA=$($PY -c "print(f'{$ML-$MM:+.6f}')")
echo "  multiL(k=$K) solo $ML   lstm_multi(k=$K) solo $MM   DELTA $DELTA"

if [ "$K" = "2" ]; then
  BAD=$($PY -c "print(1 if $ML-$MM <= $STOPBAR else 0)")
  if [ "$BAD" = "1" ]; then
    echo "  ⛔ INTERIM STOP: delta $DELTA <= $STOPBAR — the arm is hopeless; do not train s333."
    exit 10
  fi
  echo "  interim: not hopeless (delta $DELTA > $STOPBAR). No ensemble read at k=2 by pre-registration."
  exit 0
fi

PASS=$($PY -c "print(1 if $ML-$MM >= $BAR else 0)")
if [ "$PASS" != "1" ]; then
  echo "  ⛔ SOLO SCREEN FAILED: $DELTA < +$BAR. Pre-registration: the arm STOPS, no ensemble read is taken."
  exit 11
fi
echo "  ✅ solo screen passed ($DELTA >= +$BAR) — proceeding to the ensemble read."

echo "=== STAGE 3 — ENSEMBLE swap + add (matched depth), ship bar ==="
$PY ledger50_gate.py \
  --rebuild "lstm_multi:0:$D/camels531ls_multi_nhlstm_{t}{s}.csv.gz:$MSEEDS" \
  --cand multiL --replaces lstm_multi \
  --test $(for s in "${SL[@]}"; do echo -n "$D/camels531ls_multiL_nhlstm_${s}.csv.gz "; done) \
  --train $(for s in "${SL[@]}"; do echo -n "$D/camels531ls_multiL_nhlstm_TRAIN_${s}.csv.gz "; done) \
  --out benchmarks/ledger50_multiL_gate_k${K}.json
echo "=== done. If and only if it SHIPS, produce the headline with:"
echo "    $PY score_noq_test50.py --selftest && $PY score_noq_test50.py --add 'multiL:0:$D/camels531ls_multiL_nhlstm_{t}{s}.csv.gz:$SEEDS' --swap lstm_multi=multiL"
