#!/usr/bin/env bash
# MODERN-1 — NLDAS-2 as the FOURTH forcing product. Corrected replacement for
# chain_nldas4.sh, which died instantly on two bugs of mine:
#
# 1. It passed `--out`, which this builder does not have (unlike the Daymet builder).
#    I had read its argument list minutes earlier and added the flag anyway.
# 2. It logged `builder rc=0` when argparse had exited 2, because
#    `echo "[$(date +%T)] builder rc=$?"` runs the command substitution FIRST, which
#    resets $?. Verified in isolation: `false; echo "[$(date +%T)] rc=$?"` prints 0.
#    ⇒ capture the status on its own line, BEFORE anything else runs.
#
# ⚠️⚠️ THE REAL HAZARD this exposed: `OUT` in build_nldas2_areal.py is a FIXED
# constant pointing at data/modern_corpus/camels_corpus_nldas_modern_areal, which
# already holds the COMPLETE 671-basin 2014-2025 half. An `--assemble-only` over a
# 1980-2013 span would overwrite that corpus with a shorter one. The fetch phase does
# NOT assemble (it logs "run with --assemble-only to write the corpus"), so fetching
# is safe; the assemble is the dangerous step and is done ONCE, over the FULL
# 1980-2025 span, so the result is a superset. A snapshot is taken first regardless.
#
# Corrected state: of 155 cached months, 144 are 2014-2025 and only **11** are
# 1980-2013 -- the back-extension is ~2.7% done, not the 38% I first reported by
# counting every cached month against the 1980-2013 denominator.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
MON=data/modern_corpus/_nldas_monthly
OUT=data/modern_corpus/camels_corpus_nldas_modern_areal
BAK=data/modern_corpus/_nldas_2014_2025_snapshot
LOG=logs/m1_nldas4b_chain.log
exec >>"$LOG" 2>&1
echo "=== NLDAS4B CHAIN START $(date +%F_%T) ==="

nmon () { ls "$MON"/*.npz 2>/dev/null | sed 's/.*nldas_//;s/\.npz//' | awk -v a="$1" -v b="$2" '$1>=a && $1<=b' | wc -l; }
echo "[$(date +%T)] months cached: 1980-2013 $(nmon 198001 201312)/408 | 2014-2025 $(nmon 201401 202512)/144"

# ---- 0. snapshot the corpus the assemble step will overwrite ----------------
# Cheap insurance. Before overwriting, look at the target: this dir is the only copy
# of the 2014-2025 areal NLDAS half, and assemble() writes per-basin, so a failure
# midway would leave a mixed-span corpus that still looks complete by file count.
if [ ! -d "$BAK" ] && [ "$(ls "$OUT"/*.csv.gz 2>/dev/null | wc -l)" -ge 671 ]; then
  echo "[$(date +%T)] snapshotting $OUT -> $BAK"
  cp -al "$OUT" "$BAK" 2>/dev/null || cp -a "$OUT" "$BAK"
  echo "[$(date +%T)] snapshot: $(ls "$BAK"/*.csv.gz 2>/dev/null | wc -l) files"
fi

# ---- 1. FETCH 1980-2013 (safe: writes month checkpoints only) --------------
# workers 6, not the default 12: the batch controller holds a GPU job whenever
# MemAvailable drops below 3500 MiB, and Gate A plus the fused family are the
# critical path.
echo "[$(date +%T)] fetching 1980-2013 months, subset route, workers 6"
nice -n 10 $PY scripts_modern/build_nldas2_areal.py \
    --start-year 1980 --end-year 2013 --workers 6
RC=$?                      # ⭐ on its own line, before any expansion can clobber it
echo "[$(date +%T)] fetch rc=$RC"
N13=$(nmon 198001 201312)
echo "[$(date +%T)] 1980-2013 months now: $N13/408"
if [ "$N13" -lt 408 ]; then
  echo "INCOMPLETE FETCH: $N13/408 months. NOT assembling -- an assemble over a"
  echo "  partial span would overwrite the good 2014-2025 corpus with a shorter one."
  echo "  Re-running this chain resumes from the month cache."
  exit 3
fi

# ---- 2. ASSEMBLE the FULL span, once ---------------------------------------
N25=$(nmon 201401 202512)
if [ "$N25" -lt 144 ]; then
  echo "2014-2025 months incomplete ($N25/144) - refusing to assemble a gapped span"; exit 4
fi
echo "[$(date +%T)] assembling the FULL 1980-2025 span (552 months) -> $OUT"
nice -n 10 $PY scripts_modern/build_nldas2_areal.py \
    --assemble-only --start-year 1980 --end-year 2025
RC=$?
echo "[$(date +%T)] assemble rc=$RC"

# ---- 3. verify the ARTIFACT, and that it is a SUPERSET ---------------------
NB=$(ls "$OUT"/*.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] $OUT: $NB basins"
if [ "$NB" -lt 671 ]; then
  echo "ARTIFACT FAIL: $NB/671 basins. The snapshot at $BAK still holds the good"
  echo "  2014-2025 corpus -- restore from it before anything else uses this dir."
  exit 5
fi
$PY - "$OUT" <<'PYV'
import glob, gzip, os, random, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.csv.gz")))
random.seed(0)
bad = []
for f in random.sample(files, min(5, len(files))):
    rows = gzip.open(f, "rt").read().splitlines()
    first, last = rows[1].split(",")[0], rows[-1].split(",")[0]
    n = len(rows) - 1
    if not (first.startswith("1980-01-01") and last.startswith("2025-12-31")):
        bad.append((os.path.basename(f), first, last, n))
    print(f"  {os.path.basename(f)}: {first} .. {last} ({n} rows)")
assert not bad, f"span is NOT the full 1980-2025 superset: {bad}"
print("SPAN VERIFY OK: 1980-01-01..2025-12-31 on every sampled basin")
PYV
[ $? -eq 0 ] || { echo "ABORT: span verify failed; snapshot is at $BAK"; exit 6; }
echo "[$(date +%T)] NLDAS is a complete 4th product. Next: v3 corpus + cohort tree,"
echo "  then the nldasm / lstm_nldasm members and a 4-product fuse."
echo "=== NLDAS4B CHAIN DONE $(date +%F_%T) ==="
