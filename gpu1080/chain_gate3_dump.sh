#!/usr/bin/env bash
# GATE 3 — dump val1 for the 130 SCOREABLE out-of-cohort basins.
#
# Registered in PREREG_v2.md §GATE 3 plus its ADDENDUM 2026-10-16, which fixes
# the scoreable subset (benchmarks/m1_gate3_coverage.json) BEFORE any score.
# This script only produces the treatment arm; the control arm already exists
# as data/mblstm/l51_dumps/m1_nldasm9_s50?_val1.csv.gz over the 490 cohort, so
# nothing about the control is recomputed here.
#
# ⭐ RUNS ON THE CPU, deliberately. The 1080 is held by an unrelated serving
# process (7.6 of 8.2 GiB, for hours), and inference for 130 basins x 1461 days
# does not need it: measured 14.7 s for 4 basins x 60 days nice'd on 4 cores.
# `start_when_free` exists for GPU jobs; this one must not queue behind a
# process that never exits.
set -u
cd ~/riverwatch2
exec 9>/tmp/rw2_gate3_dump.lock
flock -n 9 || { echo "another GATE 3 dumper holds the lock"; exit 0; }
PY=gpu1080/.venv/bin/python
DD=data/mblstm/l51_dumps
CORP=gpu1080/corpora671/m1_g3_outcohort130
MEMBER=nldasm9
SEEDS="501 502 503 504 505"
VS=2014-10-01; VE=2018-09-30
LOG=logs/m1_gate3_dump.log
exec >>"$LOG" 2>&1
echo "=== GATE 3 dump start $(date -u +%FT%TZ)"

# The corpus must be exactly the registered treatment set. Gate on the
# RESOURCE: a corpus that is merely present tells you nothing about which
# basins are in it, and a 12-basin corpus would score silently.
WANT=$($PY -c "import json;print(len(json.load(open('benchmarks/m1_gate3_coverage.json'))['groups']['scoreable_full']['ids']))")
HAVE=$(ls "$CORP"/*.csv.gz 2>/dev/null | wc -l)
if [ "$HAVE" != "$WANT" ]; then
  echo "ABORT: corpus has $HAVE basins, the registered scoreable set has $WANT"
  exit 1
fi
$PY - "$CORP" <<'PYG'
import json, os, sys
corp = sys.argv[1]
want = set(json.load(open("benchmarks/m1_gate3_coverage.json"))
           ["groups"]["scoreable_full"]["ids"])
have = {f.split(".")[0] for f in os.listdir(corp) if f.endswith(".csv.gz")}
extra, missing = have - want, want - have
assert not extra and not missing, f"extra={sorted(extra)[:5]} missing={sorted(missing)[:5]}"
# And none of them may be in the training cohort, or this is not a transfer test.
cohort = set(json.load(open("data/m1_cohort.json"))["cohort"])
leak = have & cohort
assert not leak, f"{len(leak)} treatment basins are IN the training cohort: {sorted(leak)[:5]}"
print(f"CORPUS GATE OK: {len(have)} basins, none in the 490 cohort")
PYG
[ $? -eq 0 ] || { echo "ABORT: corpus gate failed"; exit 1; }

for S in $SEEDS; do
  CKPT="data/mblstm/gpu_ckpts/m1_${MEMBER}_s${S}.pt"
  OUT="$DD/g3oc_${MEMBER}_s${S}_val1.csv.gz"
  [ -f "$CKPT" ] || { echo "ABORT: no checkpoint $CKPT"; exit 1; }
  [ -f "$OUT" ] && { echo "[$(date -u +%T)] s$S exists, skipping"; continue; }
  echo "[$(date -u +%T)] dumping s$S -> $OUT"
  nice -n 15 $PY scripts/train_mblstm.py --epochs 0 --device cpu \
    --init-ckpt "$CKPT" --out "$CKPT.dumpsink_g3" \
    --head quantile --enc-vars camels1f --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start "$VS" --val-end "$VE" --val-stride 1 \
    --dump-day1 "$OUT" >> "logs/m1_gate3_dump_s${S}.log" 2>&1
  RC=$?   # before anything else runs: a $(date) substitution resets $?
  echo "[$(date -u +%T)] s$S builder rc=$RC"
  rm -f "$CKPT.dumpsink_g3"
  [ "$RC" = "0" ] || { echo "ABORT: s$S dump rc=$RC"; exit 1; }
  gzip -t "$OUT" || { echo "ABORT: s$S failed gzip -t"; exit 1; }
  # Sidecar, same schema and same derived row floor as the GATE 1 chain.
  $PY - "$OUT" "$CKPT" "$MEMBER" "$S" "$VS" "$VE" "$CORP" <<'PYS'
import datetime as dt, gzip, hashlib, json, os, sys
out, ck, member, seed, a, b, corp = sys.argv[1:8]
import torch as _t
cfg = _t.load(ck, map_location="cpu", weights_only=False)["cfg"]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
basins = len([f for f in os.listdir(corp) if f.endswith(".csv.gz")])
days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1
floor = int(0.40 * basins * days)
assert n >= floor, f"{n} rows < derived floor {floor} ({basins} basins x {days} d)"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
json.dump({"member": member, "seed": int(seed), "frame": "val1", "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": cfg.get("train_start"), "train_end": cfg.get("train_end"),
           "protocol": "modern-v3-9yr", "all_leads": False,
           "bytes": os.path.getsize(out), "ledger": "MODERN-1-GATE3",
           "corpus": corp, "basins_in_corpus": basins,
           "arm": "out_of_cohort_130",
           "enc_lead": int(cfg.get("enc_lead", 0) or 0),
           "dec_lead": int(cfg.get("dec_lead", 0) or 0)},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {out}: {n} rows (floor {floor})")
PYS
  [ $? -eq 0 ] || { echo "ABORT: s$S sidecar failed"; exit 1; }
done

# COMPLETENESS before the verdict. A gate over present artifacts cannot see the
# missing ones: this project once printed "FILE GATE: PASS" over 48 of 115.
HAVE=$(ls "$DD"/g3oc_${MEMBER}_s*_val1.csv.gz 2>/dev/null | wc -l)
echo "treatment dumps: $HAVE of 5"
[ "$HAVE" = "5" ] || { echo "ABORT: incomplete treatment arm"; exit 1; }
echo "[$(date -u +%T)] scoring"
$PY analysis/m1_gate3_verdict.py
echo "=== GATE 3 dump done $(date -u +%FT%TZ) rc=$?"
