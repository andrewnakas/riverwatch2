#!/bin/zsh
# Per-lead flood-F1 on the 2026 window — the ONLY archive with daily forecast
# inits (gfs_fcst_2026: 2026-03-10..06-30 consecutive), so per-lead event
# series have no NaN gaps to split events (2025 gfs/ecmwf are weekly-init and
# unusable for per-lead F1; chained F1 on the CAMELS decade already shipped,
# EXPERIMENTS row 19). Queued behind the heavy chain (one torch job at a time).
#
# Caveat baked into the output: ~112 days is a SHORT window for flood return
# periods — event counts are thin, thresholds come from each gauge's full
# corpus record (not this window). This is a methods demo of the honest
# per-lead F1, not a large-sample number.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
LOG=logs/flood_f1_perlead.log
echo "=== perlead flood-f1 armed $(date)" >> $LOG
# Fire after the aifl527 flood follow-up (last in the armed chain) so we never
# run two torch backtests at once.
until [ -f benchmarks/flood_f1_aifl527_chained.json ]; do sleep 900; done
while pgrep -f "train_mblstm.p[y]" > /dev/null \
   || pgrep -f "backtest_mblstm.p[y]" > /dev/null \
   || pgrep -f "backtest_blend_2026.p[y]" > /dev/null; do sleep 600; done

CK="data/mblstm/model_h256_s101_cmalv2p.pt:data/mblstm/model_h256_s102_cmalv2p.pt:data/mblstm/model_h256_s103_cmalv2p.pt:data/mblstm/model_h256_s104_cmalv2p.pt"
echo "step 1: daily-init 2026 dump (gfs+hrrr overlay) $(date)" >> $LOG
# NOTE: 2026 window needs the gfs2026/hrrr2026 SRC_DIRS (the --gfs sugar maps
# to gfs_fcst = 2021/2025 weekly) AND corpus_openmeteo (only corpus with 2026
# encoder history) — the default corpus stops before 2026.
RW2_ENABLE_MBLSTM=1 caffeinate -i .venv/bin/python scripts/backtest_mblstm.py \
  --ckpt "$CK" --corpus-dir data/mblstm/corpus_openmeteo \
  --forcing-plan "gfs2026:1-14,hrrr2026?:1-2" --point mean \
  --start 2026-03-10 --end 2026-06-30 --stride 1 \
  --label flood2026_daily \
  --dump-windows data/mblstm/dumps/flood2026_daily.csv.gz \
  > logs/bt_flood2026_daily.log 2>&1

if [ -f data/mblstm/dumps/flood2026_daily.csv.gz ]; then
  echo "step 2: per-lead flood F1 $(date)" >> $LOG
  .venv/bin/python scripts/score_flood_f1.py \
    --dump data/mblstm/dumps/flood2026_daily.csv.gz \
    --corpus-dir data/mblstm/corpus_openmeteo \
    --mode per-lead --leads 1-10 --sim-thresholds matched-quantile \
    --point ymean --label flood2026_perlead \
    > logs/flood_f1_perlead_score.log 2>&1 \
    && echo "per-lead F1 done $(date)" >> $LOG \
    || echo "per-lead F1 FAILED $(date)" >> $LOG
else
  echo "step 1 produced no dump — FAILED $(date)" >> $LOG
fi
echo "=== perlead flood-f1 complete $(date)" >> $LOG
