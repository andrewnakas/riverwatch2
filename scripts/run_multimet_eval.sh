#!/bin/zsh
# MultiMet HRES head-to-head vs AIFL/Google — queued behind the heavy chain.
# Corpus: multimet_corpus_531 (USGS obs + Open-Meteo weather, built from the
# 37 corpus_openmeteo-covered CAMELS gauges; grows as build_multimet_corpus.py
# fills the rest under the Open-Meteo quota). Window 2016-2020 (Caravan/CAMELS
# obs-limited); shipped cmalv2p, mean point. One torch job at a time.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
LOG=logs/multimet_eval.log
CORPUS=/Volumes/STORAGE_SD/riverwatch2_data/multimet_corpus_531
echo "=== multimet eval armed $(date)" >> $LOG
# Fire after the per-lead flood-f1 runner's marker (last torch job in the chain).
until [ -f benchmarks/flood_f1_flood2026_perlead.json ] \
   || [ -f logs/flood_f1_perlead_score.log ]; do sleep 900; done
while pgrep -f "train_mblstm.p[y]" > /dev/null \
   || pgrep -f "backtest_mblstm.p[y]" > /dev/null \
   || pgrep -f "backtest_blend_2026.p[y]" > /dev/null \
   || pgrep -f "backtest_multimet.p[y]" > /dev/null; do sleep 600; done

CK="data/mblstm/model_h256_s101_cmalv2p.pt:data/mblstm/model_h256_s102_cmalv2p.pt:data/mblstm/model_h256_s103_cmalv2p.pt:data/mblstm/model_h256_s104_cmalv2p.pt"
echo "running multimet eval $(date)" >> $LOG
RW2_ENABLE_MBLSTM=1 caffeinate -i .venv/bin/python scripts/backtest_multimet.py \
  --ckpt "$CK" --corpus-dir "$CORPUS" \
  --start 2016-01-01 --end 2020-12-31 --stride 14 --point mean \
  --label multimet_camels_$(ls "$CORPUS"/*.csv.gz 2>/dev/null | grep -vc '\._') \
  > logs/bt_multimet.log 2>&1 \
  && echo "multimet eval done $(date)" >> $LOG \
  || echo "multimet eval FAILED $(date)" >> $LOG
tail -3 logs/bt_multimet.log >> $LOG
echo "=== multimet eval complete $(date)" >> $LOG
