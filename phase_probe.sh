#!/usr/bin/env bash
# Confirm (or kill) the phase hypothesis with a STRIDE-1 dump on a basin subset.
#
# The shared-error anatomy says 88% of ensemble MSE is phase error and 1% of
# days carry 93% of it. That diagnosis came from a decomposition, not from
# watching the hydrograph move -- and it CANNOT be confirmed on the existing
# dumps, because gpu_dumps_s14 is stride-14: consecutive rows are 14 days apart,
# so a one-row shift is a fourteen-day shift and every lag test collapses
# (0.9196 -> -0.38 at +/-1 row).
#
# Stride 1 over a subset gives consecutive DAYS, which is what a sub-daily or
# one-day timing error actually needs. 60 stations keeps it to roughly an hour
# rather than the ~20 h a full 531-basin stride-1 run would take.
#
# Runs the SAME 4 checkpoints as the record so the phase behaviour measured here
# is the phase behaviour of the member we actually shipped.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts
LOG=logs/phase_probe.log
mkdir -p logs benchmarks

echo "=== PHASE PROBE START $(date +%F_%T) ===" >> $LOG

# wait for the GPU queue: NH retrain, then s985 training
PAT_NH="nh_run"" train"
PAT_TR="train_mblstm.py --epo""chs 30"
while pgrep -f "$PAT_NH" >/dev/null 2>&1 || pgrep -f "$PAT_TR" >/dev/null 2>&1; do
  sleep 300
done
echo "[$(date +%T)] GPU free" >> $LOG

CKPTS="$CK/camels531_aorc_withq_s981.pt:$CK/camels531_aorc_withq_s982.pt:$CK/camels531_aorc_withq_s983.pt:$CK/camels531_aorc_withq_s984.pt"

nice -n 10 $PY scripts/backtest_mblstm.py \
  --ckpt "$CKPTS" \
  --label "aorc_phase_stride1" \
  --corpus-dir gpu1080/corpora671/camels_corpus_aorc_v2 \
  --start 1995-10-01 --end 1997-09-30 \
  --stride 1 --limit-stations 60 \
  --dump-windows "data/mblstm/gpu_dumps_s14/phase_probe_stride1.csv.gz" \
  > benchmarks/mblstm_backtest_phase_probe.json 2>> logs/backtest_phase_probe.log
echo "=== rc=$? $(date +%T) ===" >> $LOG
ls -la data/mblstm/gpu_dumps_s14/phase_probe_stride1.csv.gz >> $LOG 2>&1
echo "=== PHASE PROBE DONE $(date +%F_%T) ===" >> $LOG
