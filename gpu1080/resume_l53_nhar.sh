#!/usr/bin/env bash
# LEDGER 53 — resume an interrupted NH AR-LSTM run from its last saved epoch,
# flatten NH's nested continue_training_from_epochNNN/ checkpoints back into
# the run dir (so queue_l53_nhar.sh's guard/eval/dump path is unchanged), then
# hand over to queue_l53_nhar.sh which skips training and evaluates.
# usage: resume_l53_nhar.sh <variant> <seed> [target_epochs=30]
set -u
cd ~/riverwatch2
V="${1:?variant}"; S="${2:?seed}"; TE="${3:-30}"
exec 8>"/tmp/rw2_l53_resume_${V}_${S}.lock"; flock -n 8 || { echo "resume already running for $V/$S"; exit 0; }
BASE="$HOME/riverwatch2/gpu1080"; PY="$BASE/.venv/bin/python"
LOG="logs/l53_nhar_${V}_s${S}.log"
RUN=$(ls -td "$BASE/nh_runs/l53_${V}_s${S}_"* 2>/dev/null | head -1)
[ -n "$RUN" ] || { echo "no run dir for $V s$S"; exit 1; }
LAST=$(ls "$RUN"/model_epoch*.pt | sed -E 's/.*model_epoch([0-9]+)\.pt/\1/' | sort -n | tail -1)
LAST=$((10#$LAST))            # "015" is OCTAL to bash arithmetic/printf; force base 10
LASTP=$(printf "%03d" "$LAST")
echo "[$(date +%F_%T)] RESUME $V s$S from epoch $LAST in $RUN (target $TE)" | tee -a "$LOG"
if [ "$LAST" -ge "$TE" ]; then echo "already complete"; exec bash gpu1080/queue_l53_nhar.sh "$V" "$S"; fi
[ -f "$RUN/optimizer_state_epoch${LASTP}.pt" ] || { echo "ABORT: no optimizer state for epoch $LAST" | tee -a "$LOG"; exit 1; }
echo "ARGV: $PY scripts/nh_arlstm_train.py --continue-run-dir $RUN --target-epochs $TE --num-workers 2" | tee -a "$LOG"
nice -n 5 $PY scripts/nh_arlstm_train.py --continue-run-dir "$RUN" --target-epochs "$TE" --num-workers 2 \
  > "logs/l53_nhtrain_${V}_s${S}_resume_from${LASTP}.log" 2>&1
RC=$?; echo "=== resumed segment rc=$RC $(date +%T) ===" | tee -a "$LOG"
NEST="$RUN/continue_training_from_epoch${LASTP}"
[ -d "$NEST" ] || { echo "ABORT: no nested dir $NEST" | tee -a "$LOG"; exit 1; }
TEP=$(printf "%03d" "$TE")
[ -f "$NEST/model_epoch${TEP}.pt" ] || { echo "ABORT: nested dir has no model_epoch${TEP}.pt" | tee -a "$LOG"; exit 1; }
# flatten: checkpoints + optimizer states up, log appended, provenance marker
mv -n "$NEST"/model_epoch*.pt "$NEST"/optimizer_state_epoch*.pt "$RUN"/
{ echo "=== resumed from epoch $LAST at $(date +%F_%T); nested log follows ==="; cat "$NEST/output.log"; } >> "$RUN/output.log"
$PY - "$RUN" "$LAST" "$TE" <<'PYM'
import json, sys, time
from pathlib import Path
run, last, te = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
m = {"resumed_from_epoch": last, "target_epochs": te, "when": time.strftime("%F %T"),
     "reason": "box-wide memory stall 2026-09-07 06:20 (weekly cron retrain_correction.sh, 3.9 GB) stalled and killed the trainers",
     "note": "optimizer state + weights restored from the last saved epoch; the AR holdout pattern is re-drawn for the "
             "resumed segment (numba RNG is unseeded in NH); lr schedule keys are absolute epochs"}
prev = json.loads((run / "RESUMED.json").read_text()) if (run / "RESUMED.json").exists() else []
(run / "RESUMED.json").write_text(json.dumps(prev + [m], indent=1))
print("RESUMED.json written")
PYM
ls "$RUN"/model_epoch*.pt | wc -l | xargs -I{} echo "[$(date +%T)] checkpoints in run dir: {}" | tee -a "$LOG"
exec bash gpu1080/queue_l53_nhar.sh "$V" "$S"
