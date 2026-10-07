#!/usr/bin/env bash
# MODERN-1 — THE FINISH: ship-bar seeds for the final compositions, then the all-leads
# tables. Leaves the test read to a deliberate, separate act.
#
# Final compositions (backward elimination, val1, PREREG rule):
#   with-q (4): gridmetm9, fused4md49, fused4m9, nldasm9        trimmed 0.913216
#   no-q   (5): lstm_fused3md3, lstm_gridmetm, lstm_fused4md4,
#               lstm_fused4m, lstm_nldasm                        trimmed 0.850768
#
# gridmetm9 already has 5 seeds; everything else is at 3. Members live in different
# launchers, so the batches are split by M1_QUEUE rather than forcing one table to hold
# all of them.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
LOG=logs/m1_final_chain.log
exec >>"$LOG" 2>&1
echo "=== FINAL CHAIN START $(date +%F_%T) ==="

# ---- 1. SHIP-BAR SEEDS (5 is the bar; no record may be frozen below it) ----
echo "[$(date +%T)] with-q ship-bar: fused4md49, fused4m9 (queue_m1d)"
M1_QUEUE=queue_m1d_withq.sh M1_LOG=m1_batch_sb_d M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh fused4md49:504 fused4md49:505 fused4m9:504 fused4m9:505

echo "[$(date +%T)] with-q ship-bar: nldasm9 (queue_m1)"
M1_QUEUE=queue_m1_withq.sh M1_LOG=m1_batch_sb_n M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh nldasm9:504 nldasm9:505

echo "[$(date +%T)] no-q ship-bar: all five final members"
M1_QUEUE=queue_m1b_noq.sh M1_LOG=m1_batch_sb_noq M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh \
  lstm_fused4md4:504 lstm_fused4md4:505 lstm_fused4m:504 lstm_fused4m:505 \
  lstm_fused3md3:504 lstm_fused3md3:505 lstm_gridmetm:504 lstm_gridmetm:505 \
  lstm_nldasm:504 lstm_nldasm:505

# ---- 2. re-score both compositions AT THE SHIP BAR -------------------------
# ⚠️ Seed depth helps the MEMBER and can HURT the ensemble (measured in this campaign).
# The SHIP-BAR number is what gets quoted, whichever way it moves.
echo "[$(date +%T)] === with-q at the 5-seed ship bar ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members gridmetm9,fused4md49,fused4m9,nldasm9 --seeds 501,502,503,504,505 --corr \
    --out benchmarks/m1_withq_FINAL_5seed_val1.json
echo "[$(date +%T)] === no-q at the 5-seed ship bar ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members lstm_fused3md3,lstm_gridmetm,lstm_fused4md4,lstm_fused4m,lstm_nldasm \
    --seeds 501,502,503,504,505 --corr --out benchmarks/m1_noq_FINAL_5seed_val1.json

# ---- 3. ALL-LEADS dumps for the final rosters ------------------------------
# ⚠️ At 3 seeds, deliberately and stated: the table reports the DECAY SHAPE across leads,
# not a record, and the day-1 headline comes from the ship-bar composition. 54 dumps at
# ~8-22 min each is ~10 h; at 5 seeds it would be ~17 h for no added information about shape.
# (A 14-lead dump used to cost >12 h each before the dump path was vectorised 90x.)
echo "[$(date +%T)] all-leads dumps, final rosters, 3 seeds"
for S in 501 502 503; do
  for M in fused4md49 fused4m9; do M1_ALLLEADS=1 bash gpu1080/queue_m1d_withq.sh "$M" "$S"; done
  for M in gridmetm9 nldasm9;    do M1_ALLLEADS=1 bash gpu1080/queue_m1_withq.sh  "$M" "$S"; done
  for M in lstm_fused3md3 lstm_gridmetm lstm_fused4md4 lstm_fused4m lstm_nldasm; do
    M1_ALLLEADS=1 bash gpu1080/queue_m1b_noq.sh "$M" "$S"; done
done

echo "[$(date +%T)] === all-leads tables ==="
$PY analysis/m1_allleads.py --frame val14 --protocol modern-v3-9yr \
    --members gridmetm9,fused4md49,fused4m9,nldasm9 --seeds 501,502,503 --crosscheck \
    --out benchmarks/m1_withq_allleads_val14.json
$PY analysis/m1_allleads.py --frame val14 --protocol modern-v3-9yr \
    --members lstm_fused3md3,lstm_gridmetm,lstm_fused4md4,lstm_fused4m,lstm_nldasm \
    --seeds 501,502,503 --crosscheck --out benchmarks/m1_noq_allleads_val14.json
echo "=== FINAL CHAIN DONE $(date +%F_%T) ==="
echo "⇒ test read is a SEPARATE deliberate act: analysis/m1_freeze_record.py"
