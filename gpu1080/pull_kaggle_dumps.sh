#!/bin/bash
# Pull whatever NH member dumps Kaggle already produced into ./dumps/ so run9.sh
# trains ONLY the missing (forcing,seed) combos overnight on the 1080.
# Needs ~/.kaggle/kaggle.json. Harmless to re-run.
set -u
cd "$(dirname "$0")"
export KAGGLE_CONFIG_DIR=~/.kaggle
USER=andrewnakas
mkdir -p dumps salvage
# Pull each kernel's FULL output (dumps + checkpoints) into a per-kernel dir, move any
# finished dumps into ./dumps/, and keep checkpoints so a wall-killed-mid-eval seed can
# be finished locally (below) instead of retrained from scratch.
pull() {  # slug
  local d="salvage/$1"; mkdir -p "$d"
  kaggle kernels output $USER/$1 -p "$d" 2>/dev/null || true
  # move any completed member dumps into ./dumps/
  for f in "$d"/dumps/camels531_*_nhlstm_s*.csv.gz "$d"/camels531_*_nhlstm_s*.csv.gz; do
    [ -f "$f" ] && cp -n "$f" dumps/ 2>/dev/null
  done
}
for F in daymet nldas maurer; do
  pull rw2-nh-train-$F
  for S in 111 222 333; do pull rw2-nhs-$F-$S; done
done

# SALVAGE: for any (forcing,seed) with a trained epoch-030 checkpoint but NO dump
# (wall-killed during eval), finish eval+dump locally — cheap, no retrain.
PATH="$PWD/.venv/bin:$PATH"
for run in salvage/*/nh_runs/rw2_*_lstm_mm_s*_* salvage/*/riverwatch2/nh_runs/rw2_*_lstm_mm_s*_*; do
  [ -d "$run" ] || continue
  ck="$run/model_epoch030.pt"; [ -f "$ck" ] || continue
  base=$(basename "$run"); F=$(echo "$base" | sed -E 's/rw2_([a-z]+)_lstm.*/\1/')
  S=$(echo "$base" | sed -E 's/.*_s([0-9]+)_.*/\1/')
  dump="dumps/camels531_${F}_nhlstm_s${S}.csv.gz"
  [ -f "$dump" ] && continue
  echo "salvaging $F s$S from checkpoint (eval+dump, no retrain)"
  nh-run evaluate --run-dir "$run" --period test 2>/dev/null || true
  res=$(ls "$run"/test/model_epoch*/test_results.p 2>/dev/null | sort | tail -1)
  [ -n "$res" ] && python3 ../scripts/nh_to_dump.py --results "$res" --forcing "$F" --out "$dump" 2>/dev/null
done

echo "NH dumps present in ./dumps/:"
ls dumps/camels531_*_nhlstm_s*.csv.gz 2>/dev/null | sed 's#.*/##' || echo "  (none yet)"
echo ""
echo "Now run:  bash run9.sh   # trains only the still-missing seeds"
