#!/usr/bin/env bash
# Verify the vectorised dump path reproduces an EXISTING all-leads dump exactly.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
REF=data/mblstm/l51_dumps/m1_gridmetm9_s501_val14.csv.gz
NEW=/tmp/verify_val14_new.csv.gz
LOG=logs/m1_dumpfast_verify.log
exec >>"$LOG" 2>&1
echo "=== DUMP-FAST VERIFY START $(date +%F_%T) ==="
[ -f "$REF" ] || { echo "no reference dump"; exit 2; }
echo "reference: $(zcat "$REF" | wc -l) lines, $(stat -c %s "$REF") bytes"
T0=$(date +%s)
nice -n 15 $PY scripts/train_mblstm.py --epochs 0 \
  --init-ckpt data/mblstm/gpu_ckpts/m1_gridmetm9_s501.pt --out /tmp/verify_sink.pt \
  --head quantile --enc-vars camels1f --q-transform linear \
  --static-set camels --hidden 256 \
  --corpus-dir data/modern_corpus/v3/camels_corpus_gridmet_m3_cohort \
  --val-start 2014-10-01 --val-end 2018-09-30 --val-stride 1 \
  --dump-day1 "$NEW" --dump-all-leads
RC=$?
T1=$(date +%s)
rm -f /tmp/verify_sink.pt
echo "new dump rc=$RC in $(( (T1-T0)/60 )) min $(( (T1-T0)%60 )) s"
[ -f "$NEW" ] || { echo "VERIFY FAIL: no new dump produced"; exit 3; }
echo "new: $(zcat "$NEW" | wc -l) lines, $(stat -c %s "$NEW") bytes"
# compare DECOMPRESSED content: gzip headers carry a timestamp, so the .gz bytes
# legitimately differ even when the content is identical.
A=$(zcat "$REF" | md5sum | cut -d" " -f1)
B=$(zcat "$NEW" | md5sum | cut -d" " -f1)
echo "ref  md5: $A"
echo "new  md5: $B"
if [ "$A" = "$B" ]; then
  echo "✅ BYTE-IDENTICAL content. The vectorised path is a pure speedup."
else
  echo "⛔ CONTENT DIFFERS -- showing the first divergences:"
  diff <(zcat "$REF") <(zcat "$NEW") | head -12
  echo "  (line counts: ref=$(zcat "$REF" | wc -l) new=$(zcat "$NEW" | wc -l))"
fi
echo "=== DUMP-FAST VERIFY DONE $(date +%F_%T) ==="
