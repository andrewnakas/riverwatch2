#!/usr/bin/env bash
# LEDGER 42 — Track A GPU arm: lstm_multi with the DAYMET PRECIP SLOT replaced
# by the screened station-anchored construction.
#
# ⛔ GATE (benchmarks/ledger42_branch_table.md, unchanged):
#     PRIMARY  3v3 complete seed separation, treatment vs FRESH same-session
#              random-init controls. <3 seeds/arm => INCONCLUSIVE, never PASS.
#     SHIP     recombined train-frame ensemble >= 0.954458 AND
#              near-median delta >= +0.0008.
# ⛔ Controls are trained HERE, in this session. The existing lstm_multi dumps
#    are the NULL SOURCE (S0.3), not the control arm — torch build, data order
#    and machine state are all confounds otherwise.
# ⛔ Nothing in this script touches 1995-2010.
#
# Idempotence keys on the EARLIEST durable artifact (the checkpoint), never the
# dump: the dump appears ~20 min after training ends, and a supervisor that
# checks only for the dump will retrain a finished seed while it evaluates.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
RUNS=gpu1080/nh_runs
DUMPS=gpu1080/dumps
mkdir -p logs "$DUMPS"

CORPUS="${CORPUS:-camels_corpus_evsub_v1}"
TAG="${TAG:-evsub}"
DATA="gpu1080/nh_data_${TAG}"
SEEDS="${SEEDS:-111 222 333}"

# ---------------------------------------------------------------- build data
if [ ! -f "$DATA/basins.txt" ]; then
  echo "=== $(date +%H:%M:%S) building NH data for $CORPUS ==="
  $PY scripts/corpus_to_nh_multi.py \
      --daymet "gpu1080/corpora671/$CORPUS" \
      --nldas  gpu1080/corpora671/camels_corpus_nldas_v2 \
      --maurer gpu1080/corpora671/camels_corpus_maurer_v2 \
      --out "$DATA" || exit 2
fi
NB=$(wc -l < "$DATA/basins.txt")
[ "$NB" -eq 531 ] || { echo "FATAL: $DATA has $NB basins, expected 531"; exit 2; }
echo "NH data ready: $DATA ($NB basins)"

# ------------------------------------------------------------------- configs
# Treatment reuses the FROZEN lstm_multi recipe byte-for-byte except data_dir,
# experiment_name and seed. Any other difference would confound the arm.
mk_cfg () {                       # $1=arm(treat|ctrl) $2=seed
  local ARM=$1 S=$2
  local SRC=gpu1080/cfgls_multi_s111.yml
  local OUT=gpu1080/cfgl42_${TAG}_${ARM}_s${S}.yml
  local DD DATADIR
  if [ "$ARM" = "treat" ]; then DATADIR="$DATA"; else DATADIR="gpu1080/nh_data_multi"; fi
  DD="/home/nakas/riverwatch2/$DATADIR"
  sed -e "s#^experiment_name:.*#experiment_name: rw2l42_${TAG}_${ARM}_s${S}#" \
      -e "s#^data_dir:.*#data_dir: $DD#" \
      -e "s#^train_basin_file:.*#train_basin_file: $DD/basins.txt#" \
      -e "s#^validation_basin_file:.*#validation_basin_file: $DD/basins.txt#" \
      -e "s#^test_basin_file:.*#test_basin_file: $DD/basins.txt#" \
      -e "s#^seed:.*#seed: $S#" "$SRC" > "$OUT"
  # verify EVERY path in the generated config exists (a cloned config that
  # still points at another corpus dies 2 s in and the supervisor logs
  # "exit=1, moving on" three times, then declares itself DONE)
  for p in $(grep -oE '/home/nakas/riverwatch2/[^ ]+' "$OUT" | sort -u); do
    case "$p" in
      */nh_runs) [ -d "$p" ] || mkdir -p "$p" ;;
      *) [ -e "$p" ] || { echo "FATAL: $OUT references missing $p"; exit 2; } ;;
    esac
  done
  echo "$OUT"
}

train_one () {                    # $1=arm  $2=seed
  local ARM=$1 S=$2
  local NAME=rw2l42_${TAG}_${ARM}_s${S}
  local CFG; CFG=$(mk_cfg "$ARM" "$S") || exit 2
  local EX; EX=$(ls -d $RUNS/${NAME}_* 2>/dev/null | head -1)
  if [ -n "$EX" ] && [ -f "$EX/model_epoch030.pt" ]; then
    echo "[skip] $NAME already has model_epoch030.pt"
  else
    echo "=== $(date +%H:%M:%S) TRAIN $NAME ==="
    $PY -m neuralhydrology.nh_run train --config-file "$CFG" \
        > "logs/l42_${TAG}_${ARM}_s${S}.log" 2>&1
    EX=$(ls -dt $RUNS/${NAME}_* 2>/dev/null | head -1)
    [ -n "$EX" ] && [ -f "$EX/model_epoch030.pt" ] || {
      echo "=== $(date +%H:%M:%S) FAIL $NAME (no epoch-30 checkpoint) ==="; return 1; }
  fi
  # dump the TRAIN window only (the gate frame). Test window stays untouched.
  local DUMP="$DUMPS/camels531ls_l42${TAG}${ARM}_nhlstm_TRAIN_s${S}.csv.gz"
  if [ -f "$DUMP" ] && gzip -t "$DUMP" 2>/dev/null; then
    echo "[skip dump] $DUMP valid"; return 0
  fi
  # Dump the BEST epoch, not blindly the last: `nh-run evaluate` defaults to the
  # final epoch, and a run that destabilises late would otherwise ship diverged
  # weights with no warning (this has happened here before).
  local LOGF="logs/l42_${TAG}_${ARM}_s${S}.log"
  local BEST; BEST=$(grep -oE "Epoch [0-9]+ average loss: avg_loss: [0-9.]+" "$LOGF" \
                     | sort -k6 -g | head -1 | awk '{print $2}')
  [ -z "$BEST" ] && BEST=30
  local EP; EP=$(printf "%03d" "$BEST")
  echo "=== $(date +%H:%M:%S) DUMP $NAME (best epoch $BEST) ==="
  gpu1080/.venv/bin/nh-run evaluate --run-dir "$EX" --period train \
      --epoch "$BEST" >> "$LOGF" 2>&1
  local RES="$EX/train/model_epoch${EP}/train_results.p"
  [ -f "$RES" ] || { echo "=== FAIL $NAME: no results at epoch $BEST ==="; return 1; }
  $PY scripts/nh_to_dump.py --results "$RES" --forcing multi \
      --out "$DUMP.tmp" >> "$LOGF" 2>&1
  if gzip -t "$DUMP.tmp" 2>/dev/null; then
    mv "$DUMP.tmp" "$DUMP"
    local NBAS; NBAS=$(zcat "$DUMP" | tail -n +2 | cut -d, -f1 | sort -u | wc -l)
    echo "=== $(date +%H:%M:%S) OK $NAME rows=$(zcat "$DUMP" | wc -l) basins=$NBAS ==="
    [ "$NBAS" -eq 531 ] || echo "  ⚠️ $NBAS basins, expected 531"
  else
    echo "=== $(date +%H:%M:%S) BAD DUMP $NAME — left as .tmp ==="; return 1
  fi
}

echo "=== LEDGER 42 Track-A arm START $(date +%F_%T) corpus=$CORPUS ==="
for S in $SEEDS; do train_one treat "$S"; done
for S in $SEEDS; do train_one ctrl  "$S"; done
echo "=== LEDGER 42 Track-A arm DONE $(date +%F_%T) ==="
ls -la "$DUMPS"/camels531ls_l42${TAG}*_TRAIN_s*.csv.gz 2>/dev/null
