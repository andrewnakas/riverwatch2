#!/bin/zsh
# B2 panel rerun (2026-07-06): grown cohort (~250 corpus_openmeteo stations vs
# 107 in EXPERIMENTS row 9) + full archive window (2026-03-09 -> 07-06, 93
# issue days). Two sequential runs so only one CPU job is live at a time (the
# B1 CAMELS trainer owns the GPU):
#   1. frozen gfsft ens4, median point  -- continuity with row 9
#   2. shipped cmalv2p ens4, mean point -- the headline config. Decoder
#      forcing stays gfs2026 + hrrr overlay for BOTH runs (no ECMWF-2026
#      archive fetched yet), so the ckpt/point swap is the only difference.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/b2_panel.log
ARCHIVE=/tmp/rw2-backtest/nwm-archive/archive
CMAL_CKPT="$PWD/data/mblstm/model_h256_s101_cmalv2p.pt:$PWD/data/mblstm/model_h256_s102_cmalv2p.pt:$PWD/data/mblstm/model_h256_s103_cmalv2p.pt:$PWD/data/mblstm/model_h256_s104_cmalv2p.pt"
echo "=== B2 panel armed $(date)" >> $LOG

# One CPU backtest at a time on this box: yield to any running mblstm eval.
while pgrep -f "backtest_mblstm.p[y]" > /dev/null; do sleep 600; done
echo "run 1: gfsft median (row-9 parity) $(date)" >> $LOG
RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_blend_2026.py \
  --archive-dir $ARCHIVE --stride-days 3 \
  --label panel > logs/b2_panel_gfsft.log 2>&1 \
  && echo "run 1 done $(date)" >> $LOG || echo "run 1 FAILED $(date)" >> $LOG

while pgrep -f "backtest_mblstm.p[y]" > /dev/null; do sleep 600; done
echo "run 2: cmalv2p mean (headline) $(date)" >> $LOG
RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_blend_2026.py \
  --archive-dir $ARCHIVE --stride-days 3 \
  --ckpt "$CMAL_CKPT" --point mean \
  --label panel_cmalv2p > logs/b2_panel_cmalv2p.log 2>&1 \
  && echo "run 2 done $(date)" >> $LOG || echo "run 2 FAILED $(date)" >> $LOG
echo "=== B2 panel complete $(date)" >> $LOG
