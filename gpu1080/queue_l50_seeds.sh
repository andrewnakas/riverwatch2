#!/usr/bin/env bash
# LEDGER 50 Stage 2 — one extra seed of an existing no-q stream (seed depth, a law; no probe).
# Usage: bash gpu1080/queue_l50_seeds.sh <NAME: multi6|multi5|multi|nldas> <SEED e.g. 444>
# Copies cfgls_<NAME>_s111.yml changing ONLY experiment_name and seed; trains; picks the
# best epoch by min avg_loss; evaluates TEST and TRAIN at that SAME epoch; dumps both.
# Idempotent on the run dir + dumps. Kill by PID only.
set -u
cd "$HOME/riverwatch2" || exit 1
NAME=${1:?name}; S=${2:?seed}; SEED=s${S}
PY=gpu1080/.venv/bin/python; NH=gpu1080/.venv/bin/nh-run
LOG=logs/l50_seeds.log
CFG=gpu1080/cfgls_${NAME}_${SEED}.yml
DT=gpu1080/dumps/camels531ls_${NAME}_nhlstm_${SEED}.csv.gz
DR=gpu1080/dumps/camels531ls_${NAME}_nhlstm_TRAIN_${SEED}.csv.gz
if [ -f "$DT" ] && [ -f "$DR" ] && gzip -t "$DT" && gzip -t "$DR"; then echo "[skip] $NAME $SEED dumps exist" | tee -a "$LOG"; exit 0; fi
if [ ! -f "$CFG" ]; then
  sed -e "s/^experiment_name: .*/experiment_name: rw2ls_${NAME}_lstm_mm_${SEED}/" -e "s/^seed: .*/seed: ${S}/" gpu1080/cfgls_${NAME}_s111.yml > "$CFG"
  grep -q "seed: ${S}$" "$CFG" && grep -q "_${SEED}$" "$CFG" || { echo "cfg substitution failed" | tee -a "$LOG"; exit 2; }
fi
EPOCHS=$(grep -oE "^epochs: [0-9]+" "$CFG" | awk '{print $2}'); [ -z "$EPOCHS" ] && EPOCHS=30
LAST=$(printf "%03d" "$EPOCHS")
R=$(ls -dt gpu1080/nh_runs/rw2ls_${NAME}_lstm_mm_${SEED}_* 2>/dev/null | head -1)
if [ -z "$R" ] || [ ! -f "$R/model_epoch${LAST}.pt" ]; then
  echo "=== L50 seed $NAME $SEED TRAIN start $(date +%F_%T) pid=$$ ===" | tee -a "$LOG"
  nice -n 5 $PY -m neuralhydrology.nh_run train --config-file "$CFG" > "logs/${NAME}_${SEED}.log" 2>&1
  echo "[$(date +%T)] training exit=$?" | tee -a "$LOG"
  R=$(ls -dt gpu1080/nh_runs/rw2ls_${NAME}_lstm_mm_${SEED}_* 2>/dev/null | head -1)
fi
BEST=$(grep -oE "Epoch [0-9]+ average loss: avg_loss: [0-9.]+" "logs/${NAME}_${SEED}.log" | sort -k6 -g | head -1 | awk '{print $2}')
[ -z "$BEST" ] && BEST=$EPOCHS
EP=$(printf "%03d" "$BEST")
echo "[$(date +%T)] $NAME $SEED run=$R best epoch $BEST" | tee -a "$LOG"
[ -f "$R/model_epoch${EP}.pt" ] || { echo "FAIL: no checkpoint $R/model_epoch${EP}.pt" | tee -a "$LOG"; exit 3; }
for P in test train; do
  $NH evaluate --run-dir "$R" --period "$P" --epoch "$BEST" >> "logs/eval_${NAME}_${SEED}_${P}.log" 2>&1
  SUF=""; [ "$P" = "train" ] && SUF="_TRAIN"
  RES="$R/$P/model_epoch${EP}/${P}_results.p"
  [ -f "$RES" ] || { echo "FAIL no $P results at epoch $BEST" | tee -a "$LOG"; exit 4; }
  $PY scripts/nh_to_dump.py --results "$RES" --forcing "$NAME" \
    --out "gpu1080/dumps/camels531ls_${NAME}_nhlstm${SUF}_${SEED}.csv.gz" >> "logs/eval_${NAME}_${SEED}_${P}.log" 2>&1
  echo "[$(date +%T)] $P dumped (epoch $EP)" | tee -a "$LOG"
done
for D in "$DT" "$DR"; do gzip -t "$D" && echo "OK $D rows=$(zcat "$D" | wc -l) basins=$(zcat "$D" | tail -n +2 | cut -d, -f1 | sort -u | wc -l)" | tee -a "$LOG"; done
echo "=== L50 seed $NAME $SEED DONE $(date +%F_%T) ===" | tee -a "$LOG"
