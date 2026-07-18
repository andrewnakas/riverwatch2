#!/bin/bash
# Train 9 real-neuralhydrology CudaLSTM members (3 forcings × 3 seeds), evaluate,
# and dump each to the combine_dumps grid. Portable — run from gpu1080/.
set -u
cd "$(dirname "$0")"
BASE="$PWD"
export PATH="$PATH:$HOME/.local/bin"
DUMP2NH=../scripts/nh_to_dump.py
mkdir -p dumps nh_runs

for F in nldas daymet maurer; do
  for S in 111 222 333; do
    DUMP="dumps/camels531_${F}_nhlstm_s${S}.csv.gz"
    if [ -f "$DUMP" ]; then echo "[$F s$S] dump exists, skip"; continue; fi
    echo "=== NH START $F s$S $(date +%H:%M:%S) ==="
    # write config for this member (fill placeholders)
    sed -e "s#__BASE__#$BASE#g" -e "s/FORCING/$F/g" -e "s/SEED/$S/g" \
        nh_config.yml.tmpl > "cfg_${F}_s${S}.yml"
    nh-run train --config-file "cfg_${F}_s${S}.yml" 2>&1 \
      | grep -iE "Epoch [0-9]+ .*loss|NSE:|Error|Traceback" | tail -40
    RUN=$(ls -dt nh_runs/rw2_${F}_lstm_mm_s${S}_* 2>/dev/null | head -1)
    echo "run dir: $RUN"
    nh-run evaluate --run-dir "$RUN" --period test 2>&1 | tail -3
    RESULTS=$(ls "$RUN"/test/model_epoch030/test_results.p 2>/dev/null | head -1)
    if [ -n "$RESULTS" ]; then
      python3 "$DUMP2NH" --results "$RESULTS" --forcing "$F" --out "$DUMP" 2>&1 | tail -2
    else
      echo "NO RESULTS for $F s$S — eval failed"
    fi
    echo "=== NH END $F s$S $(date +%H:%M:%S) ==="
  done
done
echo "NH_ALL_9_DONE $(date +%H:%M:%S)"
