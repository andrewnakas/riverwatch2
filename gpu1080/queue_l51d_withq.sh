#!/usr/bin/env bash
# LEDGER 51 — with-q rebuild on the NEARING 2022 protocol.
#
# WHY THIS SCRIPT EXISTS. Every August 2026 with-q member was trained WITHOUT
# --train-start, so windows from the 1989-10-01..1999-09-30 TEST decade were in
# the training set (6.7-8.1M train windows vs ~2.2M guarded). The 0.9253 and
# 0.9203 figures are in-sample. The August launchers each copied the July
# --val-* line and dropped the --train-start line next to it, so the guard is
# hard-coded here and ASSERTED after the fact, not passed by the caller.
#
# Protocol (Nearing et al. 2022, HESS 26:5493, fetched 2026-09-03):
#   train 1999-10-01..2008-09-30 | val 1980-10-01..1989-09-30 | test 1989-10-01..1999-09-30
#   531 basins, day-1 nowcast, median NSE across basins.
# The 365-day encoder reaches ~1 year before train-start (standard: NH does the
# same). Targets never touch the test decade.
#
# usage: queue_l51_withq.sh <member> <seed>
#   member = daymet | maurer | nldas | aorc | fused3 | fused3mse | fused3nm | fused4
set -u
cd ~/riverwatch2
M="${1:?member}"; S="${2:?seed}"

exec 9>"/tmp/rw2_l51_${M}_${S}.lock"
flock -n 9 || { echo "another launcher holds ${M}/${S}; exiting"; exit 0; }

PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts
DD=data/mblstm/l51_dumps
LOG=logs/l51_${M}_s${S}.log
mkdir -p logs "$CK" "$DD" benchmarks

case "$M" in
  daymet|maurer|nldas|aorc) CORP="gpu1080/corpora671/camels_corpus_${M}_v2_531"; ENC="camels1f"; EXTRA="" ;;
  daymeth1|maurerh1|nldash1|aorch1)
             CORP="gpu1080/corpora671/camels_corpus_${M%h1}_v2_531"; ENC="camels1f"; EXTRA="--h1-weight 0.5" ;;
  fused3h1b) CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2"; EXTRA="--h1-weight 0.9" ;;
  fused3h1nm)  CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2"; EXTRA="--h1-weight 0.5 --ar-mask-p 0.0" ;;
  fused3h1mse) CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2"; EXTRA="--h1-weight 0.5 --point-loss mse" ;;
  fused4h1)  CORP="gpu1080/corpora671/camels_corpus_fused4_v2_531"; ENC="camels4fv2"; EXTRA="--h1-weight 0.5" ;;
  fused3)    CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2"; EXTRA="" ;;
  fused3mse) CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2"; EXTRA="--point-loss mse" ;;
  fused3nm)  CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2"; EXTRA="--ar-mask-p 0.0" ;;
  fused3h1)  CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2"; EXTRA="--h1-weight 0.5" ;;
  fused4)    CORP="gpu1080/corpora671/camels_corpus_fused4_v2_531"; ENC="camels4fv2"; EXTRA="" ;;
  *) echo "unknown member $M"; exit 1 ;;
esac
CKPT="$CK/l51_${M}_s${S}.pt"

# GUARD 0: the corpus must be exactly the 531 benchmark basins.
N=$($PY -c "
import json,glob,os
ids=set(str(x).zfill(8) for x in json.load(open('data/camels_gauge_ids.json'))['531'])
have=set(os.path.basename(f)[:8] for f in glob.glob('$CORP/*.csv.gz'))
print(f'{len(ids & have)}/{len(have)}')")
echo "[$(date +%F_%T)] $M s$S corpus $CORP -> $N benchmark/total" | tee -a "$LOG"
[ "$N" = "531/531" ] || { echo "ABORT: corpus is not the 531 set" | tee -a "$LOG"; exit 1; }

# ---- train -----------------------------------------------------------------
if [ ! -f "$CKPT" ]; then
  echo "=== TRAIN $M s$S $(date +%F_%T) ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  nice -n 5 $PY scripts/train_mblstm.py \
    --epochs 30 --seed "$S" --out "$CKPT" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1980-10-01 --val-end 1989-09-30 \
    $EXTRA > "logs/l51_train_${M}_s${S}.log" 2>&1
  RC=$?
  echo "=== trained rc=$RC $(date +%T) ===" | tee -a "$LOG"
fi
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint" | tee -a "$LOG"; exit 1; }

# GUARD 1: the training window, read back from the artifacts (not the caller).
$PY - "$CKPT" "logs/l51_train_${M}_s${S}.log" <<'PYG'
import re, sys, torch
ck, log = sys.argv[1], sys.argv[2]
cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
assert cfg["train_start"] == "1999-10-01", f"train_start={cfg['train_start']!r}"
assert cfg["train_end"] == "2008-09-30", f"train_end={cfg['train_end']!r}"
assert cfg["val_range"] == ["1980-10-01", "1989-09-30"], cfg["val_range"]
assert cfg["no_q_input"] is False, "with-q member must see discharge"
assert cfg["n_stations"] == 531, cfg["n_stations"]
txt = open(log).read()
m = re.search(r"windows: train=(\d+) val=(\d+)", txt)
assert m, "no window line in log"
tr = int(m.group(1))
assert tr < 3_000_000, f"train windows {tr} — the guard did not bite"
assert "usable stations: 531" in txt, "not 531 stations"
print(f"GUARD OK: train_windows={tr} val_windows={m.group(2)} stations=531 "
      f"train={cfg['train_start']}..{cfg['train_end']}")
PYG
[ $? -eq 0 ] || { echo "ABORT: guard failed" | tee -a "$LOG"; exit 1; }

# ---- dumps -----------------------------------------------------------------
# One forward pass per frame. Physical cfs, all three quantile slots; seeds are
# averaged downstream so each seed keeps its own dump.
dump () {  # <tag> <vstart> <vend> <stride>
  local TAG="$1" VS="$2" VE="$3" ST="$4"
  local OUT="$DD/camels531_l51_${M}_s${S}_${TAG}.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date +%T)] $TAG dump exists" | tee -a "$LOG"; return 0; }
  echo "=== DUMP $TAG ($VS..$VE stride $ST) $(date +%F_%T) ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
    --out "$CKPT.dumpsink" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start "$VS" --val-end "$VE" --val-stride "$ST" \
    --dump-day1 "$OUT" >> "logs/l51_dump_${M}_s${S}_${TAG}.log" 2>&1
  rm -f "$CKPT.dumpsink"
  gzip -t "$OUT" || { echo "ABORT: $TAG dump failed gzip -t" | tee -a "$LOG"; return 1; }
  $PY - "$OUT" "$CKPT" "$TAG" "$M" "$S" <<'PYS'
import gzip, hashlib, json, sys, os
out, ck, tag, member, seed = sys.argv[1:6]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows — the dump path emitted nothing"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
side = out.replace(".csv.gz", ".json")
json.dump({"member": member, "seed": int(seed), "frame": tag, "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": "1999-10-01", "train_end": "2008-09-30",
           "protocol": "nearing2022", "bytes": os.path.getsize(out)},
          open(side, "w"), indent=1)
print(f"DUMP OK {tag}: {n} rows -> {out}")
PYS
}

dump test14 1989-10-01 1999-09-30 14   # continuity with the old record frame
dump val7   1980-10-01 1989-09-30 7    # honest window for readout + weights
dump test1  1989-10-01 1999-09-30 1    # THE HEADLINE FRAME: every day, day-1
echo "=== L51 $M s$S ALL DONE $(date +%F_%T) ===" | tee -a "$LOG"
