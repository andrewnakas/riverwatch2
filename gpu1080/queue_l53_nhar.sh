#!/usr/bin/env bash
# LEDGER 53 — NeuralHydrology AR-LSTM member (Nearing 2022's own architecture)
# on the Nearing 2022 protocol, scored next to the LEDGER-51 MB-LSTM members.
#
# Same discipline as queue_l51_withq.sh: the protocol is hard-coded here AND
# read back out of the run's config.yml / output.log by a guard before any dump
# is written; the dump sidecar records the guarded window; the scorer refuses
# dumps without it. Never edit this file while a lane is executing it.
#
# usage: queue_l53_nhar.sh <variant> <seed>
#   nhar      hidden 128, 50 % AR holdout (Nearing 2022 exact)
#   nhar256   hidden 256, 50 % AR holdout
#   nhar0     hidden 128, no AR holdout
#   nharsmoke 8 basins, 1 epoch — pipeline test only, never scored
set -u
cd ~/riverwatch2
V="${1:?variant}"; S="${2:?seed}"

exec 9>"/tmp/rw2_l53_nhar_${V}_${S}.lock"
flock -n 9 || { echo "another launcher holds ${V}/${S}; exiting"; exit 0; }

BASE="$HOME/riverwatch2/gpu1080"
PY="$BASE/.venv/bin/python"
NHRUN="$BASE/.venv/bin/nh-run"
DD=data/mblstm/l51_dumps
LOG="logs/l53_nhar_${V}_s${S}.log"
mkdir -p logs "$DD" "$BASE/nh_cfgs" "$BASE/nh_runs"

EPOCHS=30; SMOKE=0; BASINS="$BASE/nh_data_multi/basins.txt"
case "$V" in
  nhar)      HIDDEN=128; HOLDOUT=0.5 ;;
  nhar256)   HIDDEN=256; HOLDOUT=0.5 ;;
  nhar0)     HIDDEN=128; HOLDOUT=0.0 ;;
  nharsmoke) HIDDEN=128; HOLDOUT=0.5; EPOCHS=2; SMOKE=1
             BASINS="$BASE/nh_cfgs/basins_smoke8.txt"; head -8 "$BASE/nh_data_multi/basins.txt" > "$BASINS" ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac
EXP="l53_${V}_s${S}"
CFG="$BASE/nh_cfgs/${EXP}.yml"

# GUARD 0: the NH multi-forcing dataset is exactly the 531 benchmark basins.
N=$($PY -c "
import json
ids=set(str(x).zfill(8) for x in json.load(open('data/camels_gauge_ids.json'))['531'])
have=set(l.strip() for l in open('$BASE/nh_data_multi/basins.txt') if l.strip())
print(f'{len(ids & have)}/{len(have)}')")
echo "[$(date +%F_%T)] $V s$S nh_data_multi -> $N benchmark/total (hidden=$HIDDEN holdout=$HOLDOUT epochs=$EPOCHS smoke=$SMOKE)" | tee -a "$LOG"
[ "$N" = "531/531" ] || { echo "ABORT: nh_data_multi is not the 531 set" | tee -a "$LOG"; exit 1; }

find_run () { ls -td "$BASE/nh_runs/${EXP}_"* 2>/dev/null | head -1; }
RUN="$(find_run)"
CK_LAST=$(printf "model_epoch%03d.pt" "$EPOCHS")

# ---- train -----------------------------------------------------------------
if [ -z "$RUN" ] || [ ! -f "$RUN/$CK_LAST" ]; then
  sed -e "s#__BASE__#$BASE#g" -e "s#__VARIANT__#$V#g" -e "s#__SEED__#$S#g" \
      -e "s#__HIDDEN__#$HIDDEN#g" -e "s#__HOLDOUT__#$HOLDOUT#g" \
      gpu1080/nh_arlstm_config.yml.tmpl > "$CFG"
  if [ "$SMOKE" = 1 ]; then
    sed -i -e "s#^epochs: 30#epochs: $EPOCHS#" -e "s#nh_data_multi/basins.txt#nh_cfgs/basins_smoke8.txt#g" \
           -e "s#^validate_every: 10#validate_every: 1#" "$CFG"
  fi
  echo "=== TRAIN $EXP $(date +%F_%T) ===" | tee -a "$LOG"
  # scripts/nh_arlstm_train.py == `nh-run train` with ARLSTM.forward replaced by
  # the cuDNN-step CUDA-graph loop of scripts/nh_arlstm_fast.py (bit-identical
  # outputs and gradients to NH's loop; 3x faster on the 1080). Same Config,
  # same start_training, same run dir + state_dict; FAST_LOOP.json marks it.
  echo "ARGV: $PY scripts/nh_arlstm_train.py --config-file $CFG" | tee -a "$LOG"
  nice -n 5 $PY scripts/nh_arlstm_train.py --config-file "$CFG" > "logs/l53_nhtrain_${V}_s${S}.log" 2>&1
  RC=$?
  echo "=== trained rc=$RC $(date +%T) ===" | tee -a "$LOG"
  RUN="$(find_run)"
fi
[ -n "$RUN" ] && [ -f "$RUN/$CK_LAST" ] || { echo "ABORT: no final checkpoint in ${RUN:-<none>}" | tee -a "$LOG"; exit 1; }
echo "[$(date +%T)] run dir $RUN" | tee -a "$LOG"

# GUARD 1: the protocol, read back from the run's own artifacts (not argv).
$PY - "$RUN" "$S" "$HIDDEN" "$HOLDOUT" "$EPOCHS" "$SMOKE" <<'PYG'
import sys, yaml, re
from pathlib import Path
run, seed, hidden, holdout, epochs, smoke = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
cfg = yaml.safe_load(open(Path(run) / "config.yml"))
want = {"train_start_date": "01/10/1999", "train_end_date": "30/09/2008",
        "validation_start_date": "01/10/1980", "validation_end_date": "30/09/1989",
        "test_start_date": "01/10/1989", "test_end_date": "30/09/1999"}
for k, v in want.items():
    assert str(cfg[k]) == v, f"{k}={cfg[k]!r} != {v!r}"
assert cfg["model"] == "arlstm", cfg["model"]
assert int(cfg["seed"]) == seed, (cfg["seed"], seed)
assert int(cfg["hidden_size"]) == hidden, cfg["hidden_size"]
assert cfg["autoregressive_inputs"] == ["q_mm_shift1"], cfg["autoregressive_inputs"]
assert cfg["lagged_features"] == {"q_mm": [1]}, cfg["lagged_features"]
ho = cfg["random_holdout_from_dynamic_features"]["q_mm_shift1"]
assert abs(float(ho["missing_fraction"]) - holdout) < 1e-9 and int(ho["mean_missing_length"]) == 5, ho
assert len(cfg["dynamic_inputs"]) == 15 and cfg["target_variables"] == ["q_mm"], "inputs"
assert cfg["loss"] == "NSE" and int(cfg["epochs"]) == epochs, (cfg["loss"], cfg["epochs"])
nb = int(cfg["number_of_basins"])
assert nb == (8 if smoke else 531), f"number_of_basins={nb}"
log = (Path(run) / "output.log").read_text()
assert re.search(rf"Epoch {epochs} average loss", log), f"epoch {epochs} not logged"
import json
fl = json.load(open(Path(run) / "FAST_LOOP.json"))
assert fl["fast_loop"] and fl["cuda_graph"] and fl["captures"] >= 1, fl
print(f"GUARD OK: {Path(run).name} model=arlstm hidden={hidden} holdout={holdout} epochs={epochs} "
      f"basins={nb} train={cfg['train_start_date']}..{cfg['train_end_date']} fast_loop={fl}")
PYG
[ $? -eq 0 ] || { echo "ABORT: guard failed" | tee -a "$LOG"; exit 1; }

# ---- evaluate (holdout OFF) + dump ----------------------------------------
dump () {  # <frame> <period>
  local FR="$1" PER="$2"
  local RES="$RUN/$PER/$CK_LAST"; RES="${RES%.pt}"; RES="$RES/${PER}_results.p"
  local OUT="$DD/camels531_l51_${V}_s${S}_${FR}.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date +%T)] $FR dump exists" | tee -a "$LOG"; return 0; }
  if [ ! -f "$RES" ]; then
    echo "=== EVAL $PER epoch $EPOCHS (holdout OFF) $(date +%F_%T) ===" | tee -a "$LOG"
    nice -n 10 $PY scripts/nh_arlstm_eval.py --run-dir "$RUN" --period "$PER" --epoch "$EPOCHS" \
      > "logs/l53_nheval_${V}_s${S}_${FR}.log" 2>&1 || { echo "ABORT: eval $PER failed" | tee -a "$LOG"; return 1; }
  fi
  [ -f "$RES" ] || { echo "ABORT: no results at $RES" | tee -a "$LOG"; return 1; }
  local EXTRA=""; [ "$SMOKE" = 1 ] && EXTRA="--allow-partial"
  # shellcheck disable=SC2086
  $PY scripts/nh_to_l51dump.py --results "$RES" --run-dir "$RUN" --member "$V" --seed "$S" \
      --frame "$FR" --epoch "$EPOCHS" --out "$OUT" $EXTRA 2>&1 | tee -a "$LOG"
  gzip -t "$OUT" || { echo "ABORT: $FR dump failed gzip -t" | tee -a "$LOG"; return 1; }
}
dump val1  validation   # the honest selection window, stride 1
dump test1 test         # the headline frame, every day, day-1
echo "=== L53 $V s$S ALL DONE $(date +%F_%T) ===" | tee -a "$LOG"
