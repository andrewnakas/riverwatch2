#!/usr/bin/env bash
# LEDGER 54 — AR-LSTM FAMILY: single-forcing members + capacity variant, all at
# missing_fraction 0.0 (Nearing's own 0.879 config, and the precondition for the
# no-NaN fused path that makes a 3-seed member ~7 h instead of ~52 h).
#
# A NEW FILE, not an edit of queue_l53_nhar.sh: LEDGER-53 lanes are executing that
# file and bash reads a script by byte offset ([[never-edit-a-running-shell-script]]).
#
# Same discipline as L53: protocol hard-coded here, then READ BACK from the run's
# own config.yml / output.log / FAST_LOOP.json before any dump is written.
#
# usage: queue_l54_nhar.sh <variant> <seed>
set -u
cd ~/riverwatch2
V="${1:?variant}"; S="${2:?seed}"
exec 9>"/tmp/rw2_l54_${V}_${S}.lock"
flock -n 9 || { echo "another launcher holds ${V}/${S}; exiting"; exit 0; }

BASE="$HOME/riverwatch2/gpu1080"; PY="$BASE/.venv/bin/python"
DD=data/mblstm/l51_dumps; LOG="logs/l54_${V}_s${S}.log"
mkdir -p logs "$DD" "$BASE/nh_cfgs" "$BASE/nh_runs"

EPOCHS=30; SMOKE=0; HOLDOUT=0.0
case "$V" in
  nhar0d)  DATA=nh_data/daymet; FORC=daymet; HID=128 ;;
  nhar0n)  DATA=nh_data/nldas;  FORC=nldas;  HID=128 ;;
  nhar0m)  DATA=nh_data/maurer; FORC=maurer; HID=128 ;;
  nhar0h256) DATA=nh_data_multi; FORC=daymet,nldas,maurer; HID=256 ;;
  nhar0h512) DATA=nh_data_multi; FORC=daymet,nldas,maurer; HID=512 ;;
  nhar0|nhar0x) DATA=nh_data_multi; FORC=daymet,nldas,maurer; HID=128 ;;  # continues the L53 nhar0 member
  nhar0dsmoke) DATA=nh_data/daymet; FORC=daymet; HID=128; EPOCHS=1; SMOKE=1 ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac
BFILE="$DATA/basins.txt"
if [ "$SMOKE" = 1 ]; then head -8 "$BASE/$DATA/basins.txt" > "$BASE/nh_cfgs/basins_smoke8_${V}.txt"
                        BFILE="nh_cfgs/basins_smoke8_${V}.txt"; fi
EXP="l54_${V}_s${S}"; CFG="$BASE/nh_cfgs/${EXP}.yml"

# GUARD 0: the dataset is exactly the 531 benchmark basins.
N=$($PY -c "
import json
ids=set(str(x).zfill(8) for x in json.load(open('data/camels_gauge_ids.json'))['531'])
have=set(l.strip() for l in open('$BASE/$DATA/basins.txt') if l.strip())
print(f'{len(ids & have)}/{len(have)}')")
echo "[$(date +%F_%T)] $V s$S $DATA -> $N benchmark/total (hidden=$HID holdout=$HOLDOUT epochs=$EPOCHS smoke=$SMOKE)" | tee -a "$LOG"
[ "$N" = "531/531" ] || { echo "ABORT: $DATA is not the 531 set" | tee -a "$LOG"; exit 1; }

find_run () { ls -td "$BASE/nh_runs/${EXP}_"* 2>/dev/null | head -1; }
RUN="$(find_run)"; CK_LAST=$(printf "model_epoch%03d.pt" "$EPOCHS")

# RESUME: a run dir with partial checkpoints is continued from its last epoch
# (the box stalled once already under memory pressure; lanes must self-heal).
if [ -n "$RUN" ] && [ ! -f "$RUN/$CK_LAST" ]; then
  LAST=$(ls "$RUN"/model_epoch*.pt 2>/dev/null | sed -E 's/.*model_epoch([0-9]+)\.pt/\1/' | sort -n | tail -1)
  LAST=$((10#${LAST:-0}))                      # "015" is OCTAL to bash arithmetic
  LASTP=$(printf "%03d" "$LAST")
  if [ "$LAST" -gt 0 ] && [ -f "$RUN/optimizer_state_epoch${LASTP}.pt" ]; then
    echo "=== RESUME $EXP from epoch $LAST $(date +%F_%T) ===" | tee -a "$LOG"
    nice -n 5 $PY scripts/nh_arlstm_train.py --continue-run-dir "$RUN" --target-epochs "$EPOCHS"       --num-workers 2 > "logs/l54_nhtrain_${V}_s${S}_resume_from${LASTP}.log" 2>&1
    echo "=== resumed segment rc=$? $(date +%T) ===" | tee -a "$LOG"
    NEST="$RUN/continue_training_from_epoch${LASTP}"
    if [ -d "$NEST" ]; then
      mv -n "$NEST"/model_epoch*.pt "$NEST"/optimizer_state_epoch*.pt "$RUN"/ 2>/dev/null
      { echo "=== resumed from epoch $LAST at $(date +%F_%T) ==="; cat "$NEST/output.log"; } >> "$RUN/output.log"
      cp -f "$NEST/FAST_LOOP.json" "$RUN/FAST_LOOP.json" 2>/dev/null
    fi
  fi
fi
RUN="$(find_run)"
if [ -z "$RUN" ] || [ ! -f "$RUN/$CK_LAST" ]; then
  $PY scripts/render_nh_arlstm_cfg.py --variant "$V" --seed "$S" --base "$BASE" \
      --data-dir "$DATA" --forcings "$FORC" --hidden "$HID" --holdout "$HOLDOUT" \
      --epochs "$EPOCHS" --basin-file "$BFILE" --out "$CFG" | tee -a "$LOG"
  echo "=== TRAIN $EXP $(date +%F_%T) ===" | tee -a "$LOG"
  echo "ARGV: $PY scripts/nh_arlstm_train.py --config-file $CFG" | tee -a "$LOG"
  nice -n 5 $PY scripts/nh_arlstm_train.py --config-file "$CFG" > "logs/l54_nhtrain_${V}_s${S}.log" 2>&1
  echo "=== trained rc=$? $(date +%T) ===" | tee -a "$LOG"
  RUN="$(find_run)"
fi
[ -n "$RUN" ] && [ -f "$RUN/$CK_LAST" ] || { echo "ABORT: no final checkpoint in ${RUN:-<none>}" | tee -a "$LOG"; exit 1; }

# GUARD 1: protocol read back from the run's own artifacts, never from argv.
$PY - "$RUN" "$S" "$HID" "$HOLDOUT" "$EPOCHS" "$SMOKE" "$FORC" <<'PYG'
import json, re, sys, yaml
from pathlib import Path
run, seed, hid, hold, ep, smoke, forc = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                         float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
cfg = yaml.safe_load(open(Path(run) / "config.yml"))
want = {"train_start_date": "01/10/1999", "train_end_date": "30/09/2008",
        "validation_start_date": "01/10/1980", "validation_end_date": "30/09/1989",
        "test_start_date": "01/10/1989", "test_end_date": "30/09/1999"}
for k, v in want.items():
    assert str(cfg[k]) == v, f"{k}={cfg[k]!r} != {v!r}"
assert cfg["model"] == "arlstm" and int(cfg["seed"]) == seed and int(cfg["hidden_size"]) == hid
assert cfg["autoregressive_inputs"] == ["q_mm_shift1"] and cfg["lagged_features"] == {"q_mm": [1]}
ho = cfg["random_holdout_from_dynamic_features"]["q_mm_shift1"]
assert abs(float(ho["missing_fraction"]) - hold) < 1e-9, ho
nf = len([f for f in forc.split(",") if f])
assert len(cfg["dynamic_inputs"]) == 5 * nf, (len(cfg["dynamic_inputs"]), nf)
assert cfg["loss"] == "NSE" and int(cfg["epochs"]) == ep and cfg["target_variables"] == ["q_mm"]
assert int(cfg["number_of_basins"]) == (8 if smoke else 531), cfg["number_of_basins"]
log = (Path(run) / "output.log").read_text()
assert re.search(rf"Epoch {ep} average loss", log), f"epoch {ep} not logged"
fl = json.load(open(Path(run) / "FAST_LOOP.json"))
assert fl["fast_loop"], fl
# the fused path must actually have been taken for a holdout-0 run
if hold == 0.0:
    assert fl.get("fused_batches", 0) > fl.get("step_batches", 0), \
        f"fused path did not dominate: {fl.get('fused_batches')} fused vs {fl.get('step_batches')} step"
print(f"GUARD OK: {Path(run).name} arlstm hidden={hid} holdout={hold} dyn={len(cfg['dynamic_inputs'])} "
      f"basins={cfg['number_of_basins']} fused={fl.get('fused_batches')} step={fl.get('step_batches')}")
PYG
[ $? -eq 0 ] || { echo "ABORT: guard failed" | tee -a "$LOG"; exit 1; }
$PY - "$RUN" >> "$LOG" <<'PYL'
import json, sys
from pathlib import Path
print(f"  marker: {json.load(open(Path(sys.argv[1]) / 'FAST_LOOP.json'))}")
PYL

dump () {  # <frame> <period>
  local FR="$1" PER="$2"
  local RES="$RUN/$PER/${CK_LAST%.pt}/${PER}_results.p"
  local OUT="$DD/camels531_l51_${V}_s${S}_${FR}.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date +%T)] $FR dump exists" | tee -a "$LOG"; return 0; }
  if [ ! -f "$RES" ]; then
    echo "=== EVAL $PER epoch $EPOCHS (holdout OFF) $(date +%F_%T) ===" | tee -a "$LOG"
    nice -n 10 $PY scripts/nh_arlstm_eval.py --run-dir "$RUN" --period "$PER" --epoch "$EPOCHS" \
      > "logs/l54_nheval_${V}_s${S}_${FR}.log" 2>&1 || { echo "ABORT: eval $PER failed" | tee -a "$LOG"; return 1; }
  fi
  local EXTRA=""; [ "$SMOKE" = 1 ] && EXTRA="--allow-partial"
  # shellcheck disable=SC2086
  $PY scripts/nh_to_l51dump.py --results "$RES" --run-dir "$RUN" --member "$V" --seed "$S" \
      --frame "$FR" --epoch "$EPOCHS" --out "$OUT" $EXTRA 2>&1 | tee -a "$LOG"
  gzip -t "$OUT" || { echo "ABORT: $FR dump failed gzip -t" | tee -a "$LOG"; return 1; }
}
dump val1  validation
dump test1 test
echo "=== L54 $V s$S ALL DONE $(date +%F_%T) ===" | tee -a "$LOG"
