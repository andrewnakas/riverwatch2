#!/usr/bin/env bash
# When Daymet completes, build the 3-product corpus and launch the members that
# carry the measured value. Fully unattended, verified at every step.
#
# Keyed on the RESOURCE (the file count), never a log marker -- a gate keyed on a
# shared log's "BATCH DONE" matched a STALE line and fired instantly (2026-09-26).
# Every step verifies its ARTIFACT before the next runs, because an exit code has
# already lied twice today (rc=0 for a dead NH run; rc=0 for a gapped corpus).
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
LOG=logs/m1_fused3m_chain.log
exec >>"$LOG" 2>&1
echo "=== FUSED3M CHAIN START $(date +%F_%T) ==="

# ---- 1. wait for the Daymet back-extension ---------------------------------
for i in $(seq 1 720); do                      # up to 24 h
  n=$(ls data/modern_corpus/_daymet_1980_2013/*.csv.gz 2>/dev/null | wc -l)
  [ "$n" -ge 671 ] && { echo "[$(date +%T)] daymet 1980-2013 complete: $n/671"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: daymet $n/671"
  sleep 120
done
n=$(ls data/modern_corpus/_daymet_1980_2013/*.csv.gz 2>/dev/null | wc -l)
[ "$n" -lt 671 ] && { echo "DEADLINE: daymet stalled at $n/671 - stopping"; exit 2; }

# ---- 2. assemble daymet_m3 WITH the leakage shift --------------------------
# +1 day on precipitation_sum: V4R1 stores at date t the rain that falls on t+1,
# so serving it natively hands the decoder TOMORROW's rain at the step it is
# scored on. Validated: native peak lag +1 -> shifted 0, identical peak r.
echo "[$(date +%T)] assembling daymet_m3 (+1d precip shift)"
$PY scripts_modern/assemble_m3.py --product daymet \
    --parts data/modern_corpus/_daymet_1980_2013,data/modern_corpus/camels_corpus_daymet_modern_areal \
    --shift-days 1 --shift-cols precipitation_sum \
    --out "$V3/camels_corpus_daymet_m3"
nd=$(ls "$V3/camels_corpus_daymet_m3"/*.csv.gz 2>/dev/null | wc -l)
[ "$nd" -lt 671 ] && { echo "ARTIFACT FAIL: daymet_m3 has $nd/671 basins"; exit 3; }
echo "[$(date +%T)] daymet_m3 OK: $nd basins"

# ---- 3. fuse the three products --------------------------------------------
echo "[$(date +%T)] building fused3m (daymet+gridmet+nclimgrid)"
# build_fused_corpus.py takes --src + --products and appends a suffix; it had
# "_v2" hard-coded, so the modern corpora were unreachable until --suffix existed.
$PY build_fused_corpus.py --src "$V3" --suffix _m3 \
    --products daymet,gridmet,nclimgrid \
    --out "$V3/camels_corpus_fused3m_m3" 2>&1 | tail -6
nf=$(ls "$V3/camels_corpus_fused3m_m3"/*.csv.gz 2>/dev/null | wc -l)
if [ "$nf" -lt 400 ]; then
  echo "ARTIFACT FAIL: fused3m has $nf basins - build_fused_corpus may need different flags."
  echo "  Falling back to the generic merge is NOT automatic: inspect and run by hand."
  exit 4
fi
echo "[$(date +%T)] fused3m OK: $nf basins"

# ---- 4. cohort symlink tree ------------------------------------------------
DST="$V3/camels_corpus_fused3m_m3_cohort"
rm -rf "$DST"; mkdir -p "$DST"
$PY - "$V3/camels_corpus_fused3m_m3" "$DST" <<'PYS'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
ids = json.load(open("data/m1_cohort.json"))["cohort"]
n = 0
for g in ids:
    s = os.path.abspath(os.path.join(src, f"{g}.csv.gz"))
    if os.path.exists(s):
        os.symlink(s, os.path.join(dst, f"{g}.csv.gz")); n += 1
print(f"cohort tree: {n} symlinks")
PYS
nc=$(ls "$DST"/*.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] cohort tree: $nc basins (broken: $(find "$DST"/ -xtype l | wc -l))"
[ "$nc" -lt 400 ] && { echo "ARTIFACT FAIL: cohort tree only $nc"; exit 5; }

# ---- 5. launch the members that carry the value ----------------------------
# fused3md3q is the best member class measured on the CAMELS analogue
# (solo +0.006384); fused3m is the operationally servable shape; the three
# hybrids are the shape that MEASURED as admitting (+0.00021..+0.00026).
echo "[$(date +%T)] launching the fused3m family, 3-seed screen"
M1_LOG=m1_batch_fused3m bash gpu1080/run_m1_batch.sh \
  fused3md3q:501 fused3md3q:502 fused3md3q:503 \
  fused3m:501 fused3m:502 fused3m:503 \
  fused3md3:501 fused3md3:502 fused3md3:503
echo "=== FUSED3M CHAIN DONE $(date +%F_%T) ==="
