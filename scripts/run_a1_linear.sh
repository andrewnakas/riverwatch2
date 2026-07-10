#!/bin/zsh
# CAMELS recipe-v2 screen A-1 (2026-07-10) — the first step toward beating the
# 0.83 record. Two changes vs lever 2 (camels531_3f_mse, day-1 0.684), both from
# the Kratzert-2021 recipe read: --q-transform LINEAR (basin-NSE loss on
# untransformed flow, vs lever 2's asinh which underweighted peaks; our median
# FHV was -32%) and +vapor_pressure input (camels1f = compat+vp, 6 var; lever 2
# dropped vp). Screened on DAYMET single-forcing because Li/Shen 2025 shows a
# per-forcing ensemble beats one fused model, and 6-var single-forcing is
# lighter on the 8 GB box than 15-var fused.
#
# Same swap-safe scaffold as run_b1_lever2_lean.sh: pre-seed swap guard,
# .partial+promote, one-heavy-job-at-a-time, sync+settle between seeds.
#
# Queued sequence:
#   A-0  baseline of the EXISTING lever-2 ens8, SAME protocol as the A-1 eval,
#        so the screen is apples-to-apples (rebaselines the gate)
#   A-1  train seeds 941/942 (camels1f + linear) + eval
# SCREEN PROTOCOL: --stride 14 --stride-stations 3 (177 of 531 basins,
# deterministic). Measured on this box: stride-5 over 27 basins took >8 min, so
# full-531 stride-1 is a multi-hour CPU backtest — too slow to gate a screen and
# it blocks the trainer behind it. stride-14 is what lever-2's 0.684 was measured
# at, so A-1 vs A-0 on stride-14 is a DIRECT, apples-to-apples comparison; the
# 177-basin subsample tracks the full corpus within Δ<=0.03 (prior campaign).
# A-0 on the subsample calibrates the subsample-vs-full offset. Full-531 stride-1
# is reserved for the Phase-C headline only.
# Gate (recorded by hand in EXPERIMENTS.md): A-1 2-seed Daymet day-1 clearly
# beats the A-0 lever-2 subsample number (target the recipe-v2 lift toward ~0.74
# single-forcing; a flat/regressed screen means split linear-loss from +vp).
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/a1_linear.log
SD=/Volumes/STORAGE_SD/riverwatch2_data
CORPUS=$SD/camels_corpus_daymet_v2   # per-forcing recipe-v2 (has vapor_pressure)
LEVER2_CORPUS=$SD/camels_corpus_3f    # for the A-0 baseline of the old ens8
# Gate on swap FREE (allocation headroom), not swap USED: macOS never shrinks the
# swap file once inflated, so an idle-but-large file (e.g. 7 GB left over from a
# prior job) is NOT active pressure — thrashing happens when new allocations
# exceed FREE swap. Require >= SWAP_FREE_MIN_MB headroom before starting a heavy
# job. (The old used<6000 gate could deadlock forever on a stale swap file.)
SWAP_FREE_MIN_MB=4000
mkdir -p data/mblstm/dumps benchmarks/dumps logs
echo "=== A-1 recipe-v2 screen armed $(date)" >> $LOG
[ -d "$CORPUS" ] || { echo "daymet_v2 corpus missing" >> $LOG; exit 1; }

swap_free_mb() { sysctl -n vm.swapusage | awk '{print int($6)}'; }
swap_used_mb() { sysctl -n vm.swapusage | awk '{print int($3)}'; }
wait_idle() {
  while pgrep -f "train_mblstm.p[y]" > /dev/null \
     || pgrep -f "backtest_mblstm.p[y]" > /dev/null \
     || pgrep -f "build_camels_corpus.p[y]" > /dev/null \
     || pgrep -f "build_oof_pred[s]" > /dev/null; do sleep 300; done
  tries=0
  while [ "$(swap_free_mb)" -lt "$SWAP_FREE_MIN_MB" ] && [ "$tries" -lt 90 ]; do
    sync; sleep 60; tries=$((tries+1))
  done
}

# --- A-0: honest stride-1 baseline of the existing lever-2 ens8 (CPU) ---
CK2=$(ls data/mblstm/camels531_3f_mse_s93*.pt 2>/dev/null | grep -v partial | paste -sd: -)
if [ -n "$CK2" ] && [ ! -f benchmarks/mblstm_backtest_camels531_3f_ens8_s14ss3.json ]; then
  wait_idle
  echo "A-0 stride-14 subsample baseline of lever-2 ens8 (swap free $(swap_free_mb)MB/used $(swap_used_mb)MB) $(date)" >> $LOG
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK2" --corpus-dir "$LEVER2_CORPUS" \
    --start 1989-10-01 --end 1999-09-30 --stride 14 --stride-stations 3 \
    --camels-subset 531 --label camels531_3f_ens8_s14ss3 \
    > logs/bt_camels531_3f_ens8_s14ss3.log 2>&1 \
    && echo "A-0 done $(date)" >> $LOG || echo "A-0 FAILED $(date)" >> $LOG
fi

# --- A-1: recipe-v2 seeds (camels1f + linear) on Daymet ---
for SEED in 941 942; do
  OUT=data/mblstm/camels531_daymet_v2_lin_s${SEED}.pt
  [ -f "$OUT" ] && continue
  wait_idle
  echo "training daymet_v2 linear seed ${SEED} (swap free $(swap_free_mb)MB/used $(swap_used_mb)MB) $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py \
    --corpus-dir "$CORPUS" --enc-vars camels1f --no-q-input \
    --point-loss mse --q-transform linear \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    --hidden 256 --epochs 16 --windows-per-station 1000 --batch 128 \
    --val-stride 20 --lr 2e-4 --seed ${SEED} --device mps \
    --out "$OUT.partial" > logs/camels531_daymet_v2_lin_s${SEED}.log 2>&1 \
    && mv "$OUT.partial" "$OUT" \
    && echo "seed ${SEED} done (swap free $(swap_free_mb)MB/used $(swap_used_mb)MB) $(date)" >> $LOG \
    || echo "seed ${SEED} FAILED/killed $(date)" >> $LOG
  rm -f "$OUT.partial"
  sync; sleep 30
done

# --- A-1 eval: stride-14 subsample of the 2-seed screen ---
CK=$(ls data/mblstm/camels531_daymet_v2_lin_s94*.pt 2>/dev/null | grep -v partial | paste -sd: -)
NDONE=$(ls data/mblstm/camels531_daymet_v2_lin_s94*.pt 2>/dev/null | grep -vc partial)
if [ -n "$CK" ] && [ "$NDONE" -ge 2 ] && [ ! -f benchmarks/mblstm_backtest_camels531_daymet_v2_lin_ens2.json ]; then
  wait_idle
  echo "A-1 eval daymet_v2 linear ens${NDONE} (swap free $(swap_free_mb)MB/used $(swap_used_mb)MB) $(date)" >> $LOG
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK" --corpus-dir "$CORPUS" \
    --start 1989-10-01 --end 1999-09-30 --stride 14 --stride-stations 3 \
    --camels-subset 531 --label camels531_daymet_v2_lin_ens2 \
    > logs/bt_camels531_daymet_v2_lin_ens2.log 2>&1 \
    && echo "A-1 eval done $(date)" >> $LOG || echo "A-1 eval FAILED $(date)" >> $LOG
fi
echo "=== A-1 recipe-v2 screen complete $(date)" >> $LOG
