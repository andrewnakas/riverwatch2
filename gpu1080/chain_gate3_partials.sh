#!/usr/bin/env bash
# GATE 3 — score the TWO partial-coverage basins, separately, as registered.
#
# PREREG_v2.md GATE 3 ADDENDUM 2026-10-16 says the 2 basins with 25-90% of the
# val window observed are "scored and reported SEPARATELY, and are not pooled
# into the deciding median". The main run reported n_scored = 0 for them, and
# correctly so -- the treatment corpus was built from the scoreable_full list
# of 130, so the dump never covered them. That is a deviation from the
# registration, not a result, and it is cheap to close: 2 basins.
#
# They stay OUT of the deciding median. A basin contributing ~1,100 days is not
# comparable to one contributing 1,461, and a median whose rows differ between
# arms has misled this project before.
set -u
cd ~/riverwatch2
exec 9>/tmp/rw2_gate3_partials.lock
flock -n 9 || { echo "another partials run holds the lock"; exit 0; }
PY=gpu1080/.venv/bin/python
DD=data/mblstm/l51_dumps
LOG=logs/m1_gate3_partials.log
exec >>"$LOG" 2>&1
echo "=== GATE 3 partials start $(date -u +%FT%TZ)"

CORP=gpu1080/corpora671/m1_g3_partial2
mkdir -p "$CORP"; rm -f "$CORP"/*.csv.gz
$PY - "$CORP" <<'PYG'
import json, os, sys
corp = sys.argv[1]
ids = json.load(open("benchmarks/m1_gate3_coverage.json"))["groups"]["partial"]["ids"]
src = "data/modern_corpus/camels_corpus_nldas_modern_areal"
cohort = set(json.load(open("data/m1_cohort.json"))["cohort"])
assert not (set(ids) & cohort), "a partial basin is in the training cohort"
for sid in ids:
    p = os.path.join(src, f"{sid}.csv.gz")
    assert os.path.exists(p), p
    os.symlink(os.path.realpath(p), os.path.join(corp, f"{sid}.csv.gz"))
print(f"PARTIAL CORPUS OK: {len(ids)} basins {ids}, none in the 490 cohort")
PYG
[ $? -eq 0 ] || { echo "ABORT: corpus gate failed"; exit 1; }

for S in 501 502 503 504 505; do
  CKPT="data/mblstm/gpu_ckpts/m1_nldasm9_s${S}.pt"
  OUT="$DD/g3pt_nldasm9_s${S}_val1.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date -u +%T)] s$S exists"; continue; }
  echo "[$(date -u +%T)] dumping s$S"
  nice -n 15 $PY scripts/train_mblstm.py --epochs 0 --device cpu \
    --init-ckpt "$CKPT" --out "$CKPT.dumpsink_g3pt" \
    --head quantile --enc-vars camels1f --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start 2014-10-01 --val-end 2018-09-30 --val-stride 1 \
    --dump-day1 "$OUT" >> logs/m1_gate3_partials_dump.log 2>&1
  RC=$?
  rm -f "$CKPT.dumpsink_g3pt"
  [ "$RC" = "0" ] || { echo "ABORT: s$S rc=$RC"; exit 1; }
  gzip -t "$OUT" || { echo "ABORT: s$S failed gzip -t"; exit 1; }
  $PY - "$OUT" "$CKPT" "$S" "$CORP" <<'PYS'
import gzip, hashlib, json, os, sys
out, ck, seed, corp = sys.argv[1:5]
import torch as _t
cfg = _t.load(ck, map_location="cpu", weights_only=False)["cfg"]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
json.dump({"member": "nldasm9", "seed": int(seed), "frame": "val1", "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": cfg.get("train_start"), "train_end": cfg.get("train_end"),
           "protocol": "modern-v3-9yr", "all_leads": False,
           "bytes": os.path.getsize(out), "ledger": "MODERN-1-GATE3-PARTIAL",
           "corpus": corp, "basins_in_corpus": 2,
           "arm": "out_of_cohort_partial_coverage",
           "enc_lead": int(cfg.get("enc_lead", 0) or 0),
           "dec_lead": int(cfg.get("dec_lead", 0) or 0)},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {out}: {n} rows")
PYS
  [ $? -eq 0 ] || { echo "ABORT: s$S sidecar failed"; exit 1; }
done

HAVE=$(ls "$DD"/g3pt_nldasm9_s*_val1.csv.gz 2>/dev/null | wc -l)
echo "partial dumps: $HAVE of 5"
[ "$HAVE" = "5" ] || { echo "ABORT: incomplete"; exit 1; }
$PY analysis/m1_gate3_partials.py
echo "=== GATE 3 partials done $(date -u +%FT%TZ) rc=$?"
