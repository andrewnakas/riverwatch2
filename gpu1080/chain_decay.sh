#!/usr/bin/env bash
# MODERN-1 — registered decision (c): the day-1 forecast-forcing decay, measured.
#
# Dumps nldasm9 (the gridMET-free member of the corrected roster) under REAL archived 1-day-ahead
# GFS forcing substituted into the decoder, then pairs it against the existing perfect-forcing dump
# on identical (basin, t0) rows.
#
# The forcing is AREAL (scripts_modern/build_gfs_areal_lead1.py + gfs_to_forcing_schema.py), unlike
# the point-sampled archive on the unmounted drive which runs ~53% dry vs areal in mountain
# terrain. It passed its registration screen: precipitation peaks at k=0 with the day bookkeeping
# independently confirmed by tmean/tmax/srad/vp all peaking at k=0 (benchmarks/m1_gfs_areal_screen.json).
#
# ⚠️ This reads TEST-window rows (the archive starts 2021-05, after the val window ends). It selects
# nothing -- both arms are the same frozen checkpoints -- and is logged as a characterization.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_decay_chain.log
exec >>"$LOG" 2>&1
echo "=== DECAY CHAIN START $(date +%F_%T) ==="

QUIET=0
for i in $(seq 1 2880); do
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  T=$(ps -eo cmd --no-headers | awk 'index($0,"train_mblstm.py")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  C=$(ps -eo cmd --no-headers | awk 'index($0,"chain_decomp.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$B" -eq 0 ] && [ "$T" -eq 0 ] && [ "$C" -eq 0 ] && [ "$N" -eq 0 ]; then
    QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 consecutive polls"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: batch=$B train=$T decomp=$C gpu=$N quiet=$QUIET/3"
  sleep 120
done

PY=gpu1080/.venv/bin/python
M=nldasm9
NF=$(ls data/mblstm/gfs_fcst/*.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] forcing archive: $NF init dates"
[ "$NF" -lt 50 ] && { echo "ABORT: only $NF forcing dates"; exit 6; }

for S in 501 502 503 504 505; do
  CKPT=data/mblstm/gpu_ckpts/m1_${M}_s${S}.pt
  OUT=data/mblstm/l51_dumps/m1_${M}_s${S}_test1gfs.csv.gz
  [ -f "$CKPT" ] || { echo "[$(date +%T)] no ckpt for seed $S, skipping"; continue; }
  [ -f "$OUT" ] && { echo "[$(date +%T)] seed $S already dumped"; continue; }
  echo "[$(date +%T)] dumping $M s$S under REAL GFS forcing"
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
      --out "$CKPT.decaysink_$S" \
      --corpus-dir data/modern_corpus/v3/camels_corpus_gridmetfix3_m3_cohort \
      --val-start 2018-10-01 --val-end 2025-09-30 --val-stride 1 \
      --gfs-finetune --dump-day1 "$OUT"
  rc=$?
  echo "[$(date +%T)] seed $S dump rc=$rc"
  rm -f "$CKPT.decaysink_$S"
  # a sidecar so the dump is never provenance-less; this frame is NOT a record frame
  $PY - "$OUT" "$M" "$S" <<'PYS'
import gzip, json, os, sys
out, member, seed = sys.argv[1:4]
if os.path.exists(out):
    n = sum(1 for _ in gzip.open(out, "rt")) - 1
    json.dump({"member": member, "seed": int(seed), "frame": "test1gfs", "rows": n,
               "protocol": "modern-v3-9yr", "train_start": "2005-10-01",
               "train_end": "2014-09-30", "all_leads": False,
               "decoder_forcing": "archived AREAL GFS 1-day-ahead (real forecast error)",
               "frame_caveat": "sparse: archive starts 2021-05, 14-day stride; DELTA only",
               "not_a_record_frame": True, "ledger": "MODERN-1"},
              open(out.replace(".csv.gz", ".json"), "w"), indent=1)
    print(f"sidecar written: {n} rows")
PYS
done

echo "[$(date +%T)] ===== THE DAY-1 DECAY, paired on identical rows ====="
$PY analysis/m1_decay_measure.py --member $M --seeds 501,502,503,504,505 \
    --out benchmarks/m1_decay_day1_nldasm9.json 2>&1 | tail -24
echo "=== DECAY CHAIN DONE $(date +%F_%T) ==="
