#!/usr/bin/env bash
# LEDGER 51 GATE 2 (ANCHOR) — re-dump the 12 GUARDED July with-q checkpoints on
# the new frames, so the clean rebuild can be compared against a member whose
# training window was already correct.
#
# ⚠️ These checkpoints are guarded in TRAINING but were best-epoch selected on
# --val 1998-10-01..1999-09-30, i.e. INSIDE the test decade. So 0.9058 is itself
# mildly optimistic, and a LEDGER-51 member scoring a little below it is not
# automatically a regression. Stated before the comparison is made.
#
# usage: dump_l51_anchor.sh <forcing>   (daymet|maurer|nldas)
set -u
cd ~/riverwatch2
F="${1:?forcing}"
PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts
DD=data/mblstm/l51_dumps
CORP="gpu1080/corpora671/camels_corpus_${F}_v2_531"
mkdir -p "$DD" logs

for S in 981 982 983 984; do
  CKPT="$CK/camels531_${F}_withq_s${S}.pt"
  [ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
  $PY - "$CKPT" <<'PYG'
import sys, torch
cfg = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["cfg"]
assert cfg["train_end"] == "2008-09-30", f"{sys.argv[1]}: train_end={cfg['train_end']} — NOT guarded"
assert cfg["n_stations"] == 531 and cfg["no_q_input"] is False
PYG
  [ $? -eq 0 ] || exit 1
  for SPEC in "test14 1989-10-01 1999-09-30 14" "test1 1989-10-01 1999-09-30 1"; do
    set -- $SPEC
    OUT="$DD/camels531_l51_anchor${F}_s${S}_${1}.csv.gz"
    [ -f "$OUT" ] && continue
    echo "=== anchor $F s$S $1 $(date +%F_%T) ===" >> logs/l51_anchor_${F}.log
    nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
      --out "$CKPT.dumpsink" \
      --head quantile --enc-vars camels1f --q-transform linear \
      --static-set camels --hidden 256 --corpus-dir "$CORP" \
      --val-start "$2" --val-end "$3" --val-stride "$4" \
      --dump-day1 "$OUT" >> logs/l51_anchor_${F}.log 2>&1
    rm -f "$CKPT.dumpsink"
    gzip -t "$OUT" || { echo "ABORT: $OUT failed gzip -t"; exit 1; }
    $PY - "$OUT" "$CKPT" "$1" "anchor$F" "$S" <<'PYS'
import gzip, hashlib, json, os, sys
out, ck, tag, member, seed = sys.argv[1:6]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows"
assert gzip.open(out, "rt").readline().strip() == "station_id,t0,h,truth,ylo,ymed,yhi"
json.dump({"member": member, "seed": int(seed), "frame": tag, "rows": n, "ckpt": ck,
           "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": "1999-10-01", "train_end": "2008-09-30",
           "protocol": "nearing2022-july-guarded",
           "caveat": "best-epoch selected on 1998-10..1999-09, inside the test decade",
           "bytes": os.path.getsize(out)}, open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"ANCHOR DUMP OK {member} s{seed} {tag}: {n} rows")
PYS
  done
done
echo "=== ANCHOR $F DONE $(date +%F_%T) ==="
