#!/usr/bin/env bash
# MODERN-1 — when NLDAS completes, make it USABLE: v3 corpus, the 4-product fuse,
# cohort trees. Then STOP.
#
# ⛔ DELIBERATELY LAUNCHES NO GPU WORK. chain_nh3 auto-claimed the card three minutes
# before the registered STOP gate wanted it, because its "is the box quiet" gate could
# not see an imminent consumer. A builder chain has no business scheduling GPU work;
# it prepares artifacts and reports what became runnable.
#
# Unlocks, once this finishes: fused4m / fused4md4 / fused4md4q (already defined in
# queue_m1_withq.sh against camels_corpus_fused4m_m3_cohort), nldasm / nldasmq, the
# nldas *_d4 hybrids, and lstm_nldasm in the no-q arm.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
SRC=data/modern_corpus/camels_corpus_nldas_modern_areal
MON=data/modern_corpus/_nldas_monthly
LOG=logs/m1_fuse4_chain.log
exec >>"$LOG" 2>&1
echo "=== FUSE4 CHAIN START $(date +%F_%T) ==="

nmon () { ls "$MON"/*.npz 2>/dev/null | sed 's/.*nldas_//;s/\.npz//' | awk '$1<201400' | wc -l; }
# Completion test is the ARTIFACT's span, not a file count: the dir already holds 671
# files for 2014-2025, so counting files would say "done" before the back-extension
# was assembled at all.
full_span () {
  local f
  f=$(ls "$SRC"/*.csv.gz 2>/dev/null | head -1) || return 1
  [ -z "$f" ] && return 1
  [ "$(zcat "$f" 2>/dev/null | sed -n 2p | cut -d, -f1)" = "1980-01-01" ]
}

# ---- 1. wait: stall-keyed on the month counter, completion on the span ------
LAST=-1; STALL=0
for i in $(seq 1 4320); do            # ceiling 144 h (fetch measured ~7 min/month)
  if full_span; then echo "[$(date +%T)] NLDAS assembled to 1980-01-01"; break; fi
  n=$(nmon)
  if [ "$n" -gt "$LAST" ]; then LAST=$n; STALL=0; else STALL=$((STALL + 1)); fi
  # ⚠️ SIZED FROM AN OBSERVED OUTAGE, not from the work rate. The 6 h window this
  # replaces fired correctly by its own rule and was still wrong: a network partition
  # froze the box for ~9 h (up 1d22h, so it never rebooted), NLDAS could not fetch, the
  # month counter stopped, and this chain killed itself while the build was healthy.
  # ⇒ A frozen counter with a LIVE producer is not a dead build -- it is a stall worth
  # waiting through. Declare failure only when the producer is actually gone, or after
  # a window far longer than any outage this box has shown.
  ALIVE=$(ps -eo cmd --no-headers | grep -c "build_nldas2_area[l]")
  if [ "$STALL" -ge 540 ] && [ "$ALIVE" -eq 0 ]; then
    echo "STALLED: months stuck at $n/408 for 18 h AND no fetcher running - stopping"; exit 2
  fi
  if [ "$STALL" -ge 1080 ]; then
    echo "STALLED: months stuck at $n/408 for 36 h even with a fetcher alive - stopping"; exit 2
  fi
  [ $((i % 60)) -eq 0 ] && echo "[$(date +%T)] waiting: months $n/408 (stall $STALL/180)"
  sleep 120
done
full_span || { echo "CEILING reached without a full-span NLDAS corpus"; exit 2; }

# ---- 2. v3 corpus for nldas (symlink tree over the assembled corpus) -------
DST="$V3/camels_corpus_nldas_m3"
rm -rf "$DST"; mkdir -p "$DST"
for f in "$SRC"/*.csv.gz; do ln -s "$(readlink -f "$f")" "$DST/$(basename "$f")"; done
NN=$(ls "$DST"/*.csv.gz 2>/dev/null | wc -l)
BR=$(find "$DST"/ -xtype l | wc -l)
echo "[$(date +%T)] $DST: $NN basins (broken: $BR)"
{ [ "$NN" -lt 671 ] || [ "$BR" -gt 0 ]; } && { echo "ARTIFACT FAIL: nldas_m3 tree"; exit 3; }

# ---- 3. the 4-product fuse -------------------------------------------------
if [ "$(ls "$V3/camels_corpus_fused4m_m3"/*.csv.gz 2>/dev/null | wc -l)" -lt 400 ]; then
  echo "[$(date +%T)] fusing 4 products (daymet,nldas,gridmet,nclimgrid)"
  $PY build_fused_corpus.py --src "$V3" --suffix _m3 \
      --products daymet,nldas,gridmet,nclimgrid \
      --out "$V3/camels_corpus_fused4m_m3" 2>&1 | tail -6
fi
NF=$(ls "$V3/camels_corpus_fused4m_m3"/*.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] fused4m: $NF basins"
[ "$NF" -lt 400 ] && { echo "ARTIFACT FAIL: fused4m has $NF"; exit 4; }

# ---- 3b. verify the fuse against its sources, all FOUR products ------------
# ⚠️ The inner join sets the span. NLDAS now covers 1980-2025 but report the result
# explicitly rather than discovering a 2008-style truncation at scoring time.
$PY - "$V3" <<'PYV'
import glob, os, random, sys
import pandas as pd
v3 = sys.argv[1]
fused = os.path.join(v3, "camels_corpus_fused4m_m3")
files = sorted(glob.glob(os.path.join(fused, "*.csv.gz")))
random.seed(0)
checked, span = 0, None
for f in random.sample(files, min(3, len(files))):
    g = os.path.basename(f)
    F = pd.read_csv(f, dtype={"date": str}).set_index("date")
    for p in ("daymet", "nldas", "gridmet", "nclimgrid"):
        S = pd.read_csv(os.path.join(v3, f"camels_corpus_{p}_m3", g),
                        dtype={"date": str}).set_index("date")
        common = F.index.intersection(S.index)
        assert len(common) > 3000, f"{g}/{p}: only {len(common)} common dates"
        for col in ("precipitation_sum", "temperature_2m_mean"):
            fc = f"{col}_{p}"
            assert fc in F.columns, f"{g}: fused missing {fc}"
            d = (F.loc[common, fc].astype(float) - S.loc[common, col].astype(float)).abs().max()
            assert d < 1e-6, f"{g}: {fc} != {p}/{col} (max|diff|={d:.6g}) - MERGE MISALIGNED"
        checked += 1
    span = f"{F.index.min()}..{F.index.max()}"
print(f"FUSE4 VERIFY OK: {checked} basin-product pairs value-identical to source")
print(f"FUSED4 SPAN (set by the inner join): {span}")
PYV
[ $? -eq 0 ] || { echo "ABORT: fused4m does not match its sources"; exit 44; }

# ---- 4. cohort trees -------------------------------------------------------
for NAME in nldas fused4m; do
  D="$V3/camels_corpus_${NAME}_m3_cohort"
  rm -rf "$D"; mkdir -p "$D"
  $PY - "$V3/camels_corpus_${NAME}_m3" "$D" <<'PYS'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
for g in json.load(open("data/m1_cohort.json"))["cohort"]:
    s = os.path.realpath(os.path.join(src, f"{g}.csv.gz"))
    if os.path.exists(s):
        os.symlink(s, os.path.join(dst, f"{g}.csv.gz")); n += 1
print(f"cohort tree {dst}: {n} symlinks")
PYS
  NC=$(ls "$D"/*.csv.gz 2>/dev/null | wc -l)
  BR=$(find "$D"/ -xtype l | wc -l)
  echo "[$(date +%T)] $NAME cohort tree: $NC (broken: $BR)"
  { [ "$NC" -lt 400 ] || [ "$BR" -gt 0 ]; } && { echo "ARTIFACT FAIL: $NAME cohort"; exit 5; }
done

echo "[$(date +%T)] ✅ FOUR-PRODUCT CORPUS READY. Now runnable (launch deliberately,"
echo "   never from this chain): fused4m / fused4md4 / fused4md4q, nldasm / nldasmq,"
echo "   daymetmd4 / nldasmd4 / gridmetmd4 / nclimgridmd4, and lstm_nldasm (no-q)."
echo "=== FUSE4 CHAIN DONE $(date +%F_%T) ==="
