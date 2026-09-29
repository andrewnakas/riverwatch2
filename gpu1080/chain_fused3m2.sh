#!/usr/bin/env bash
# MODERN-1 — when Daymet completes: build the 3-product corpus, run the REGISTERED
# STOP GATE (Gate A), and only then spend GPU on the P-MODERN family.
#
# Replaces chain_fused3m.sh, which exited on its own 24 h wall-clock deadline at
# 621/671 while the Daymet build was still perfectly healthy and progressing at
# ~5.5 basins/h. That deadline existed for a good reason (a dead GPU once went
# unnoticed for four days) but it was sized on an optimistic rate, so it silently
# abandoned a working build and left the card idle. ⇒ A deadline must key on LACK
# OF PROGRESS, not on elapsed time. Fixed below.
#
# Three ordering/verification changes from the version it replaces:
#   1. STALL-based wait: abort only if the file count has not moved in 3 h.
#   2. A fused-column check that compares the fused corpus against its SOURCE
#      products value-by-value -- the merge is column-keyed and the parts have
#      different column counts, so a positional bug is possible and would be
#      invisible for three GPU-hours.
#   3. GATE A RUNS FIRST and gates everything after it. It is the registered STOP
#      gate (PREREG_v2.md MODERN-1); the version it replaces launched nine
#      P-MODERN runs ahead of it, putting ~8 GPU-hours at risk of a broken
#      extraction the gate exists to catch.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
LOG=logs/m1_fused3m_chain2.log
exec >>"$LOG" 2>&1
echo "=== FUSED3M CHAIN 2 START $(date +%F_%T) ==="

# ---- 1. wait for the Daymet back-extension, STALL-keyed ---------------------
SRC=data/modern_corpus/_daymet_1980_2013
LAST=-1; STALL=0; STALL_LIMIT=90          # 90 polls x 120 s = 3 h with no progress
for i in $(seq 1 2880); do                # hard ceiling 96 h
  n=$(ls "$SRC"/*.csv.gz 2>/dev/null | wc -l)
  [ "$n" -ge 671 ] && { echo "[$(date +%T)] daymet complete: $n/671"; break; }
  if [ "$n" -gt "$LAST" ]; then LAST=$n; STALL=0; else STALL=$((STALL + 1)); fi
  if [ "$STALL" -ge "$STALL_LIMIT" ]; then
    echo "STALLED: daymet stuck at $n/671 for 3 h with no new basin - stopping"; exit 2
  fi
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: daymet $n/671 (stall $STALL/$STALL_LIMIT)"
  sleep 120
done
n=$(ls "$SRC"/*.csv.gz 2>/dev/null | wc -l)
[ "$n" -lt 671 ] && { echo "CEILING: daymet at $n/671 after 96 h - stopping"; exit 2; }

# ---- 2. assemble daymet_m3 WITH the precip leakage shift --------------------
# +1 day on precipitation_sum: V4R1 stores at date t the rain that falls on t+1,
# so serving it natively hands the decoder TOMORROW's rain at the step it is
# scored on. Validated: native peak lag +1 -> shifted 0, identical peak r.
if [ "$(ls "$V3/camels_corpus_daymet_m3"/*.csv.gz 2>/dev/null | wc -l)" -lt 671 ]; then
  echo "[$(date +%T)] assembling daymet_m3 (+1d precip shift)"
  $PY scripts_modern/assemble_m3.py --product daymet \
      --parts "$SRC",data/modern_corpus/camels_corpus_daymet_modern_areal \
      --shift-days 1 --shift-cols precipitation_sum \
      --out "$V3/camels_corpus_daymet_m3"
fi
nd=$(ls "$V3/camels_corpus_daymet_m3"/*.csv.gz 2>/dev/null | wc -l)
[ "$nd" -lt 671 ] && { echo "ARTIFACT FAIL: daymet_m3 has $nd/671"; exit 3; }
echo "[$(date +%T)] daymet_m3 OK: $nd basins"

# ---- 3. fuse the three products --------------------------------------------
if [ "$(ls "$V3/camels_corpus_fused3m_m3"/*.csv.gz 2>/dev/null | wc -l)" -lt 400 ]; then
  echo "[$(date +%T)] building fused3m (daymet+gridmet+nclimgrid)"
  $PY build_fused_corpus.py --src "$V3" --suffix _m3 \
      --products daymet,gridmet,nclimgrid \
      --out "$V3/camels_corpus_fused3m_m3" 2>&1 | tail -6
fi
nf=$(ls "$V3/camels_corpus_fused3m_m3"/*.csv.gz 2>/dev/null | wc -l)
[ "$nf" -lt 400 ] && { echo "ARTIFACT FAIL: fused3m has $nf basins"; exit 4; }
echo "[$(date +%T)] fused3m OK: $nf basins"

# ---- 3b. VERIFY THE FUSE AGAINST ITS SOURCES -------------------------------
# Cheap (seconds) and it guards a real failure mode: the merge is column-keyed
# because the build parts carry 8 vs 10 columns, so a positional regression would
# silently pair gridMET values with a Daymet column name. Three GPU-hours of
# Gate A would then "fail the pipeline" for the wrong reason.
echo "[$(date +%T)] verifying fused columns against source products"
$PY - "$V3" <<'PYV'
import glob, os, random, sys
import pandas as pd
v3 = sys.argv[1]
fused = os.path.join(v3, "camels_corpus_fused3m_m3")
prods = ["daymet", "gridmet", "nclimgrid"]
files = sorted(glob.glob(os.path.join(fused, "*.csv.gz")))
random.seed(0)
checked = 0
for f in random.sample(files, min(3, len(files))):
    g = os.path.basename(f)
    F = pd.read_csv(f, dtype={"date": str}).set_index("date")
    for p in prods:
        sp = os.path.join(v3, f"camels_corpus_{p}_m3", g)
        S = pd.read_csv(sp, dtype={"date": str}).set_index("date")
        common = F.index.intersection(S.index)
        assert len(common) > 3000, f"{g}/{p}: only {len(common)} common dates"
        for col in ("precipitation_sum", "temperature_2m_mean"):
            fc = f"{col}_{p}"
            assert fc in F.columns, f"{g}: fused is missing {fc}"
            d = (F.loc[common, fc].astype(float) - S.loc[common, col].astype(float)).abs().max()
            assert d < 1e-6, f"{g}: {fc} != {p}/{col} (max|diff|={d:.6g}) - MERGE IS MISALIGNED"
        checked += 1
    span = f"{F.index.min()}..{F.index.max()}"
print(f"FUSE VERIFY OK: {checked} basin-product pairs value-identical to source; span {span}")
PYV
[ $? -eq 0 ] || { echo "ABORT: fused corpus does not match its sources"; exit 44; }

# ---- 4. cohort symlink trees (fused + daymet) ------------------------------
for NAME in fused3m daymet; do
  DST="$V3/camels_corpus_${NAME}_m3_cohort"
  rm -rf "$DST"; mkdir -p "$DST"
  $PY - "$V3/camels_corpus_${NAME}_m3" "$DST" <<'PYS'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
ids = json.load(open("data/m1_cohort.json"))["cohort"]
n = 0
for g in ids:
    s = os.path.abspath(os.path.join(src, f"{g}.csv.gz"))
    if os.path.exists(s):
        os.symlink(s, os.path.join(dst, f"{g}.csv.gz")); n += 1
print(f"cohort tree {dst}: {n} symlinks")
PYS
  nc=$(ls "$DST"/*.csv.gz 2>/dev/null | wc -l)
  br=$(find "$DST"/ -xtype l | wc -l)
  echo "[$(date +%T)] $NAME cohort tree: $nc basins (broken: $br)"
  { [ "$nc" -lt 400 ] || [ "$br" -gt 0 ]; } && { echo "ARTIFACT FAIL: $NAME cohort tree"; exit 5; }
done

# ---- 5. GATE A -- the registered STOP gate, BEFORE the P-MODERN spend ------
echo "[$(date +%T)] GATE A: fused3mN on P-NEARING, 3 seeds"
M1_LOG=m1_batch_gateA bash gpu1080/run_m1_batch.sh fused3mN:501 fused3mN:502 fused3mN:503
echo "[$(date +%T)] GATE A training done; scoring against frozen fused3h1"
$PY analysis/m1_gate_a.py --frame test1 --cand fused3mN --ref fused3h1 --seeds 501,502,503
GA=$?
echo "[$(date +%T)] GATE A scorer rc=$GA"
if [ "$GA" -ne 0 ]; then
  echo "=== ⛔ GATE A FAILED ITS REGISTERED FALSIFIER - NOT launching the P-MODERN family."
  echo "    Re-run the Phase-1 lag/streamflow screens on the BUILT fused corpus."
  echo "=== FUSED3M CHAIN 2 HALTED $(date +%F_%T) ==="
  exit 9
fi

# ---- 6. the P-MODERN family, released only by a passing Gate A -------------
# fused3md3q is the best member class measured on the CAMELS analogue
# (solo +0.006384); fused3m is the operationally servable shape; fused3md3 is the
# all-product decoder without dec-q, which separates the two fixes.
echo "[$(date +%T)] GATE A PASSED - launching the fused3m family, 3-seed screen"
M1_LOG=m1_batch_fused3m bash gpu1080/run_m1_batch.sh \
  fused3md3q:501 fused3md3q:502 fused3md3q:503 \
  fused3m:501 fused3m:502 fused3m:503 \
  fused3md3:501 fused3md3:502 fused3md3:503

# ---- 7. daymet as the third single-product member --------------------------
echo "[$(date +%T)] launching daymetm (3rd single-product member), 3-seed screen"
M1_LOG=m1_batch_daymetm bash gpu1080/run_m1_batch.sh daymetm:501 daymetm:502 daymetm:503
echo "=== FUSED3M CHAIN 2 DONE $(date +%F_%T) ==="
